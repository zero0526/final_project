import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from matrix_source.agents.ppo_networks import (
    MultiInstanceActor, MultiInstanceCritic, MFNetwork
)
from matrix_source.agents.buffer.policy_replay_buffer import MultiAgentPolicyBuffer
from matrix_source.agents.abstract_agent import AbstractAgent
from matrix_source.visualize.tracking import TrainingDashboard
from matrix_source.utils.helper import get_grad_norm
from matrix_source.utils.math_utils import compute_gae


class PPOAgent(AbstractAgent):
    def __init__(self, node_id, node_type, state_dim, action_dim, u_action_dim,
                 mf_hidden_sizes, mf_lr, buffer_min_size,
                 hidden_sizes=(128, 64),
                 lr=1e-4, critic_lr=5e-4, gamma=0.99, alpha=0.005,
                 buffer_size=100000, batch_size=64,
                 lam=0.95, clip_eps=0.4, k_epochs=5, entropy_coef=0.05,
                 exclude_zero=False, num_instances=1, device=None,
                 dashboard: TrainingDashboard = None):

        super().__init__()

        self.node_id = node_id
        self.node_type = node_type

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.num_instances = num_instances
        self.action_dim = action_dim
        self.u_action_dim = u_action_dim
        self.exclude_zero = exclude_zero

        self.initial_entropy_coef = entropy_coef
        self.entropy_coef = entropy_coef

        # PPO Hyperparameters
        self.gamma = gamma
        self.lmbda = lam
        self.eps_clip = clip_eps
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.min_batch_size = buffer_min_size
        self.alpha = alpha

        # Networks — no targets
        self.actor = MultiInstanceActor(
            state_dim, self.action_dim, self.u_action_dim,
            hidden_sizes, num_instances
        ).to(self.device)

        self.critic = MultiInstanceCritic(
            state_dim, self.action_dim, hidden_sizes, num_instances
        ).to(self.device)

        self.mf_net = MFNetwork(
            state_dim + self.action_dim, self.action_dim,
            mf_hidden_sizes, num_instances
        ).to(self.device)

        # Optimizers
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.mf_optimizer = optim.Adam(self.mf_net.parameters(), lr=mf_lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.memory = MultiAgentPolicyBuffer(
            num_instances, buffer_size, state_dim,
            self.action_dim, self.u_action_dim, self.device
        )
        self.dashboard = dashboard
        self.learn_step_counter = 0

    # ── Action Selection ──

    def choose_action(self, state, prev_mf, epsilon, mask=None,
                      agent_idx=0, zeta=1.0):
        idx_tensor = torch.tensor([agent_idx], device=self.device)
        actions, _, _ = self.choose_action_batch(
            state.unsqueeze(0) if not torch.is_tensor(state) else state.detach().unsqueeze(0),
            prev_mf.unsqueeze(0) if not torch.is_tensor(prev_mf) else prev_mf.detach().unsqueeze(0),
            masks_batch=mask.unsqueeze(0) if mask is not None else None,
            agent_indices=idx_tensor,
            zeta=zeta
        )
        return int(actions[0])

    def choose_action_batch(self, states, mfs, masks_batch=None,
                             agent_indices=None, deterministic=False, zeta=1.0):
        batch_size = states.shape[0]
        if agent_indices is None:
            agent_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        else:
            agent_indices = agent_indices.to(self.device).view(-1)

        states = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        mfs = torch.as_tensor(mfs, device=self.device, dtype=torch.float32)

        if masks_batch is not None:
            masks_batch = masks_batch.to(self.device)

        with torch.no_grad():
            # MF prediction — online only
            pred_mfs = self.mf_net(
                torch.cat([states, mfs], dim=-1), indices=agent_indices)

            logits = self.actor(states, pred_mfs, indices=agent_indices)
            values = self.critic(states, pred_mfs, indices=agent_indices)

            if masks_batch is not None:
                logits = logits.masked_fill(masks_batch == 0, -1e9)

            if self.exclude_zero and self.u_action_dim > 1:
                zero_mask = torch.zeros_like(logits, dtype=torch.bool)
                zero_mask[:, 0] = True
                logits = logits.masked_fill(zero_mask, -1e9)

            # ζ scaling
            if zeta != 1.0:
                logits = logits * zeta

            if deterministic:
                actions = logits.argmax(dim=-1)
                log_probs = torch.zeros(batch_size, device=self.device)
            else:
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

        return actions, log_probs.detach(), values.detach()

    # ── Store + MF Train ──

    def store_transition_train_mf_batch(self, states, prev_mfs, curr_mfs,
                                         actions, rewards, next_states, dones,
                                         agent_ids, log_prob, value, masks=None):
        self.learn_mf_batch(states, prev_mfs, curr_mfs, agent_ids)
        self.memory.add_batch(
            states, prev_mfs, curr_mfs, actions, rewards,
            next_states, dones, log_prob, value, agent_ids, masks=masks
        )
        return 0.0

    def learn_mf_batch(self, states, prev_mfs, ground_truth_mfs, agent_ids):
        s = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        pmf = torch.as_tensor(prev_mfs, device=self.device, dtype=torch.float32)
        gt_mf = torch.as_tensor(ground_truth_mfs, device=self.device, dtype=torch.float32)

        pred_mf = self.mf_net(torch.cat([s, pmf], dim=-1), indices=agent_ids)
        loss = self.loss_fn(pred_mf, gt_mf)

        self.mf_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mf_net.parameters(), max_norm=5.0)
        self.mf_optimizer.step()
        return loss.item()

    # ── Learn — Standard PPO, no targets ──

    def learn(self, zeta=1.0, agents_ids=None, **kwargs):
        if agents_ids is not None:
            agents_ids = agents_ids.to(self.device).view(-1)

        data = self.memory.get_all_ready(
            min_size=self.min_batch_size, agent_ids_pool=agents_ids)
        if data is None:
            return None

        states, prev_mfs, curr_mfs, actions, rewards, next_states, \
            dones, old_log_probs, old_values, masks, agent_ids = data

        actions = actions.squeeze(-1)
        old_log_probs = old_log_probs.squeeze(-1)
        old_values = old_values.squeeze(-1)
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)

        dataset_size = states.shape[0]
        # ── 1. GAE with ONLINE critic (no target) ──
        with torch.no_grad():
            next_pred_mfs = self.mf_net(
                torch.cat([next_states, curr_mfs], dim=-1), indices=agent_ids)
            next_values = self.critic(
                next_states, next_pred_mfs, indices=agent_ids)

            advantages = compute_gae(
                rewards, next_values, old_values, dones,
                agent_ids, self.gamma, self.lmbda)
            returns = advantages + old_values

            if advantages.shape[0] > 1:
                advantages = (advantages - advantages.mean()) / \
                             (advantages.std() + 1e-8)

        # ── 2. PPO epochs ──
        epoch_v_loss = 0
        total_batches = 0

        for _ in range(self.k_epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                batch_states = states[idx]
                batch_prev_mfs = prev_mfs[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                batch_agent_ids = agent_ids[idx]
                batch_masks = masks[idx]

                # MF detach for actor/critic update
                batch_pred_mfs = self.mf_net(
                    torch.cat([batch_states, batch_prev_mfs], dim=-1),
                    indices=batch_agent_ids
                ).detach()

                # Actor update
                log_probs, entropy = self.actor.evaluate(
                    batch_states, batch_pred_mfs, batch_actions,
                    masks=batch_masks, indices=batch_agent_ids,
                    exclude_zero=self.exclude_zero, zeta=zeta
                )

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.eps_clip, 1 + self.eps_clip
                ) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() \
                             - self.entropy_coef * entropy.mean()

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.optimizer_actor.step()

                # Critic update
                values = self.critic(
                    batch_states, batch_pred_mfs, indices=batch_agent_ids)
                critic_loss = F.mse_loss(values, batch_returns)

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.optimizer_critic.step()

                epoch_v_loss += critic_loss.item()

                # Dashboard — single logging point
                if self.dashboard is not None:
                    with torch.no_grad():
                        self.dashboard.log({
                            'v_loss': critic_loss.item(),
                            'actor_loss': actor_loss.item(),
                            'entropy': entropy.mean().item(),
                            'entropy_coef': self.entropy_coef,
                            'avg_reward': rewards.mean().item(),
                            'advantage_mean': batch_advantages.mean().item(),
                            'advantage_std': batch_advantages.std().item(),
                            'ratio_mean': ratio.mean().item(),
                            'approx_kl': (batch_old_log_probs - log_probs.detach()).mean().item(),
                            'clip_fraction': ((ratio - 1).abs() > self.eps_clip).float().mean().item(),
                            'grad_norm_actor': get_grad_norm(self.actor),
                            'grad_norm_critic': get_grad_norm(self.critic),
                            'avg_value_pred': values.mean().item(),
                            'avg_returns': batch_returns.mean().item(),
                            'temperature': 1.0 / max(zeta, 1e-8),
                            'zeta': zeta,
                            'lr_actor': self.optimizer_actor.param_groups[0]['lr'],
                            'lr_critic': self.optimizer_critic.param_groups[0]['lr'],
                            'k_epochs': self.k_epochs,
                        })

                total_batches += 1

        # ── 3. Post-update hooks ──
        avg_reward = rewards.mean().item()
        self.update_entropy(avg_reward)
        self.step_lr_schedulers()

        self.learn_step_counter += 1

        log_freq = 10 if self.node_type == "Edge_Group" else 100
        if self.learn_step_counter % log_freq == 0:
            avg_v_loss = epoch_v_loss / total_batches if total_batches > 0 else 0
            avg_v = old_values.mean().item()
            phase = self.get_entropy_phase()
            print(
                f"[{self.node_type} PPO] Step {self.learn_step_counter:5d} | "
                f"Phase: {phase:7s} | VLoss: {avg_v_loss:.5f} | "
                f"AvgV: {avg_v:.3f} | EntCoef: {self.entropy_coef:.5f}")

        self.memory.clear()
        return epoch_v_loss / total_batches if total_batches > 0 else 0

    # ── Save/Load — simplified, no targets ──

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'mf_net': self.mf_net.state_dict(),
            'actor_opt': self.optimizer_actor.state_dict(),
            'critic_opt': self.optimizer_critic.state_dict(),
            'mf_opt': self.mf_optimizer.state_dict(),
            'learn_step': self.learn_step_counter,
            'entropy_coef': self.entropy_coef
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.mf_net.load_state_dict(ckpt['mf_net'])
        self.optimizer_actor.load_state_dict(ckpt['actor_opt'])
        self.optimizer_critic.load_state_dict(ckpt['critic_opt'])
        self.mf_optimizer.load_state_dict(ckpt['mf_opt'])
        self.learn_step_counter = ckpt.get('learn_step', 0)
        self.entropy_coef = ckpt.get('entropy_coef', self.initial_entropy_coef)
