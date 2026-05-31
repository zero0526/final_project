import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from matrix_source.agents.buffer.policy_replay_buffer import MultiAgentPolicyBuffer
from matrix_source.visualize.metric_tracking import PPOTracker


class MultiInstanceLinear(nn.Module):
    """
    Parallel linear layer for multiple independent agent models.
    Supports batched inference where each sample in the batch can
    use a specific agent's weights.
    """

    def __init__(self, num_instances, in_features, out_features, bias=True):
        super().__init__()
        self.num_instances = num_instances
        self.in_features = in_features
        self.out_features = out_features

        # Weights: (num_instances, in, out)
        self.weight = nn.Parameter(torch.Tensor(num_instances, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_instances, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Orthogonal initialization per instance
        for i in range(self.num_instances):
            nn.init.orthogonal_(self.weight[i], gain=1.0)
            if self.bias is not None:
                nn.init.zeros_(self.bias[i])

    def forward(self, x, indices=None):
        if indices is None:
            indices = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        w = self.weight[indices]
        out = torch.bmm(x.unsqueeze(1), w).squeeze(1)

        if self.bias is not None:
            out += self.bias[indices]
        return out


class MultiInstanceRMSNorm(nn.Module):
    """
    Instance-specific Root Mean Square Layer Normalization.
    """

    def __init__(self, num_instances, normalized_shape, eps=1e-6):
        super().__init__()
        self.num_instances = num_instances
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_instances, normalized_shape))

    def forward(self, x, indices):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        gamma = self.weight[indices]
        return x_norm * gamma


class MultiInstanceActor(nn.Module):
    def __init__(self, state_dim, mf_dim, action_dim, hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        self.num_instances = num_instances

        # Shared Feature Extractor
        self.fc1 = MultiInstanceLinear(num_instances, state_dim + mf_dim, h1)
        self.norm1 = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2 = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2 = MultiInstanceRMSNorm(num_instances, h2)

        # Actor head (returns logits for Categorical distribution)
        self.actor_logits = MultiInstanceLinear(num_instances, h2, action_dim)

    def forward(self, state, mf, indices=None):
        x = torch.cat([state, mf], dim=-1)
        x = F.silu(self.norm1(self.fc1(x, indices), indices))
        x = F.silu(self.norm2(self.fc2(x, indices), indices))
        return self.actor_logits(x, indices)

    def evaluate(self, state, mf, action, masks=None, indices=None, exclude_zero=False, zeta=1.0):
        logits = self.forward(state, mf, indices)

        # Apply zeta (temperature scaling)
        # if zeta != 1.0:
        #     logits = logits * zeta

        if masks is not None:
            logits = logits.masked_fill(masks == 0, -1e9)

        # --- APPLY EXCLUDE_ZERO LOGIC ---
        if exclude_zero and logits.shape[-1] > 1:
            zero_mask = torch.zeros_like(logits, dtype=torch.bool)
            zero_mask[:, 0] = True
            logits = logits.masked_fill(zero_mask, -1e9)
        # --------------------------------

        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


class MultiInstanceCritic(nn.Module):
    def __init__(self, state_dim, mf_dim, hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        self.num_instances = num_instances

        self.fc1 = MultiInstanceLinear(num_instances, state_dim + mf_dim, h1)
        self.norm1 = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2 = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2 = MultiInstanceRMSNorm(num_instances, h2)

        # Critic head (returns state value)
        self.critic = MultiInstanceLinear(num_instances, h2, 1)

    def forward(self, state, mf, indices=None):
        x = torch.cat([state, mf], dim=-1)
        x = F.silu(self.norm1(self.fc1(x, indices), indices))
        x = F.silu(self.norm2(self.fc2(x, indices), indices))
        return self.critic(x, indices).squeeze(-1)


class MFNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        self.fc1 = MultiInstanceLinear(num_instances, input_dim, h1)
        self.norm = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2 = MultiInstanceLinear(num_instances, h1, h2)
        self.out = MultiInstanceLinear(num_instances, h2, output_dim)

    def forward(self, x, indices=None):
        x = F.silu(self.norm(self.fc1(x, indices), indices))
        x = F.silu(self.fc2(x, indices))
        return torch.sigmoid(self.out(x, indices))  # Constrain MF to [0, 1] range


class PPOAgent:
    def __init__(self, node_id, node_type, state_dim, action_dim, u_action_dim,
                 mf_hidden_sizes, mf_lr, buffer_min_size,
                 hidden_sizes=(128, 64),
                 lr=1e-4,critic_lr=5e-4, gamma=0.99, alpha=0.005, buffer_size=100000, batch_size=64,
                 lam=0.95, clip_eps=0.4, k_epochs=5, entropy_coef=0.05,
                 exclude_zero=False, num_instances=1, device=None):

        self.node_id = node_id
        self.node_type = node_type

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.num_instances = num_instances

        # Dimensions
        self.action_dim = action_dim  # Dimensions of mean field
        self.u_action_dim = u_action_dim  # Number of discrete actions
        self.exclude_zero = exclude_zero
        
        # --- ENTROPY DECAY THAY THẾ AUTO-TUNING ---
        self.initial_entropy_coef = entropy_coef  # Lưu lại giá trị ban đầu (ví dụ 0.05)
        self.entropy_coef = entropy_coef          # Giá trị đang dùng hiện tại
        self.entropy_decay_rate = 0.99          # Tốc độ giảm sau mỗi lần learn (thử 0.999 - 0.9999)
        self.min_entropy_coef = 0.001             # Giá trị nhỏ nhất cho phép (không để nó bằng 0 hoàn toàn)
        self.zeta= 0.6
        self.zeta_decay_rate = 0.99
        self.max_zeta = 5
        # PPO Hyperparameters
        self.gamma = gamma
        self.lmbda = lam
        self.eps_clip = clip_eps
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.min_batch_size = buffer_min_size
        self.entropy_coef = entropy_coef
        self.alpha = alpha

        # Networks
        self.actor = MultiInstanceActor(state_dim, self.action_dim, self.u_action_dim, hidden_sizes, num_instances).to(
            self.device)
        self.critic = MultiInstanceCritic(state_dim, self.action_dim, hidden_sizes, num_instances).to(self.device)
        self.mf_net = MFNetwork(state_dim + self.action_dim, self.action_dim, mf_hidden_sizes, num_instances).to(
            self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.mf_optimizer = optim.Adam(self.mf_net.parameters(), lr=mf_lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.memory = MultiAgentPolicyBuffer(num_instances, buffer_size, state_dim, self.action_dim, self.u_action_dim,
                                             self.device)
        self.learn_step_counter = 0
        self.tracker = PPOTracker(name=f"ppo_{self.node_type.lower()}")

    def choose_action(self, state, prev_mf, epsilon, mask=None, agent_idx=0, zeta=1.0):
        # Single agent usage
        idx_tensor = torch.tensor([agent_idx], device=self.device)
        actions, _, _ = self.choose_action_batch(
            state.unsqueeze(0) if not torch.is_tensor(state) else state.detach().unsqueeze(0),
            prev_mf.unsqueeze(0) if not torch.is_tensor(prev_mf) else prev_mf.detach().unsqueeze(0),
            masks_batch=mask.unsqueeze(0) if mask is not None else None,
            agent_indices=idx_tensor,
            zeta=zeta
        )
        return int(actions[0])

    def choose_action_batch(self, states, mfs, masks_batch=None, agent_indices=None, deterministic=False, zeta=1.0):
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
            # 1. Predict current MF
            pred_mfs = self.mf_net(torch.cat([states, mfs], dim=-1), indices=agent_indices)

            # 2. Get Actor Logits and Critic Values
            logits = self.actor(states, pred_mfs, indices=agent_indices)
            values = self.critic(states, pred_mfs, indices=agent_indices)

            # 3. Apply masks
            if masks_batch is not None:
                logits = logits.masked_fill(masks_batch == 0, -1e9)

            if self.exclude_zero and self.u_action_dim > 1:
                # Tạo mask cho cột 0
                zero_mask = torch.zeros_like(logits, dtype=torch.bool)
                zero_mask[:, 0] = True
                logits = logits.masked_fill(zero_mask, -1e9)

            # Apply zeta (temperature scaling)
            if zeta != 1.0:
                logits = logits * self.zeta

            # 4. Sample actions
            if deterministic:
                actions = logits.argmax(dim=-1)
                log_probs = torch.zeros(batch_size, device=self.device)
            else:
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

        return actions, log_probs.detach(), values.detach()

    def store_transition_train_mf_batch(self, states, prev_mfs, curr_mfs, actions, rewards, next_states, dones,
                                        agent_ids, log_prob, value, masks=None):
        # 1. Train MF (supervised learning)
        loss_mf = self.learn_mf_batch(states, prev_mfs, curr_mfs, agent_ids)

        # 2. Store in Buffer
        # Ensure log_prob and value are handled as lists/tensors correctly
        self.memory.add_batch(states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, log_prob,
                              value, agent_ids, masks=masks)
        return loss_mf

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

    def learn(self, agents_ids: torch.Tensor = None, zeta=1.0):
        from matrix_source.trainers.ppo_stategy import compute_gae

        if agents_ids is not None:
            agents_ids = agents_ids.to(self.device).view(-1)

        data = self.memory.get_all_ready(min_size=self.min_batch_size, agent_ids_pool=agents_ids)
        if data is None:
            return None
        self.zeta = min(self.zeta * 1/self.zeta_decay_rate, self.max_zeta)
        self.entropy_coef = max(self.entropy_coef * self.entropy_decay_rate, self.min_entropy_coef)
        # ============================================================

        states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, old_log_probs, old_values, masks, agent_ids = data

        actions = actions.squeeze(-1)
        old_log_probs = old_log_probs.squeeze(-1)
        old_values = old_values.squeeze(-1)
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)

        # Track rewards
        reward_mean = rewards.mean().item()
        reward_std = rewards.std().item()

        dataset_size = states.shape[0]

        # 1. Compute Advantages and Targets
        with torch.no_grad():
            # --- SỬA BUG CRITICAL Ở ĐÂY: Re-predict MF ---
            next_pred_mfs = self.mf_net(torch.cat([next_states, curr_mfs], dim=-1), indices=agent_ids)
            # ---------------------------------------------

            next_values = self.critic(next_states, next_pred_mfs, indices=agent_ids)  # Dùng next_pred_mfs
            advantages = compute_gae(rewards, next_values, old_values, dones, agent_ids, self.gamma, self.lmbda)
            returns = advantages + old_values

            if advantages.shape[0] > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 2. PPO Mini-batch Update Epochs
        epoch_v_loss = 0
        total_batches = 0
        total_entropy = 0
        total_max_prob = 0

        for _ in range(self.k_epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                batch_states = states[idx]
                batch_prev_mfs = prev_mfs[idx]
                batch_curr_mfs = curr_mfs[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                batch_agent_ids = agent_ids[idx]
                batch_masks = masks[idx]

                # --- Re-predict MF cho mini-batch ---
                # Detach để tránh backward qua MF net 2 lần (do Actor và Critic đều dùng)
                batch_pred_mfs = self.mf_net(torch.cat([batch_states, batch_prev_mfs], dim=-1), indices=batch_agent_ids).detach()
                # -------------------------------------

                # Dùng batch_pred_mfs thay vì batch_curr_mfs
                log_probs, entropy = self.actor.evaluate(
                    batch_states, batch_pred_mfs, batch_actions, masks=batch_masks,
                    indices=batch_agent_ids, exclude_zero=self.exclude_zero,
                    zeta=self.zeta
                )
                values = self.critic(batch_states, batch_pred_mfs, indices=batch_agent_ids)

                # Calculate Max Probability for tracking
                with torch.no_grad():
                    batch_logits = self.actor(batch_states, batch_pred_mfs, indices=batch_agent_ids)
                    if batch_masks is not None:
                        batch_logits = batch_logits.masked_fill(batch_masks == 0, -1e9)
                    probs = F.softmax(batch_logits * self.zeta, dim=-1)
                    max_prob = probs.max(dim=-1)[0].mean().item()
                    total_max_prob += max_prob
                
                total_entropy += entropy.mean().item()

                ratio = torch.exp(log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.optimizer_actor.step()

                critic_loss = F.mse_loss(values, batch_returns)

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.optimizer_critic.step()

                epoch_v_loss += critic_loss.item()
                total_batches += 1

        self.learn_step_counter += 1
        
        # Record Metrics
        avg_entropy = total_entropy / total_batches if total_batches > 0 else 0
        avg_max_prob = total_max_prob / total_batches if total_batches > 0 else 0
        # Normalized Entropy: H / log(|A|)
        # Note: log is natural log in numpy/torch by default
        log_dim = np.log(max(self.u_action_dim, 2)) 
        norm_entropy = avg_entropy / log_dim if log_dim > 0 else 0
        
        self.tracker.record(
            reward_mean=reward_mean,
            reward_std=reward_std,
            entropy=avg_entropy,
            norm_entropy=norm_entropy,
            zeta=self.zeta,
            alpha=self.entropy_coef,
            max_prob=avg_max_prob
        )

        log_freq = 10 if self.node_type == "Edge_Group" else 100
        if self.learn_step_counter % log_freq == 0:
            self.tracker.plot() # Save Plot
            avg_v_loss = epoch_v_loss / total_batches if total_batches > 0 else 0
            avg_v = old_values.mean().item()
            print(
                f"[{self.node_type} PPO] Step {self.learn_step_counter:5d} | "
                f"VLoss: {avg_v_loss:.5f} | AvgV: {avg_v:.3f} | "
                f"EntCoef: {self.entropy_coef:.5f}")

        self.memory.clear()
        return epoch_v_loss / total_batches if total_batches > 0 else 0

    def save(self, path):
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'mf_net': self.mf_net.state_dict(),
            'actor_opt': self.optimizer_actor.state_dict(),
            'critic_opt': self.optimizer_critic.state_dict(),
            'mf_opt': self.mf_optimizer.state_dict(),
            'learn_step': self.learn_step_counter,
            'entropy_coef': self.entropy_coef
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.mf_net.load_state_dict(checkpoint['mf_net'])
        self.optimizer_actor.load_state_dict(checkpoint['actor_opt'])
        self.optimizer_critic.load_state_dict(checkpoint['critic_opt'])
        self.mf_optimizer.load_state_dict(checkpoint['mf_opt'])
        self.learn_step_counter = checkpoint.get('learn_step', 0)
        self.entropy_coef = checkpoint.get('entropy_coef', self.initial_entropy_coef)

    def set_lr_factor(self, factor):
        """
        Scales the learning rate of all optimizers by the given factor.
        """
        for opt in [self.optimizer_actor, self.optimizer_critic, self.mf_optimizer]:
            for param_group in opt.param_groups:
                param_group['lr'] *= factor
        print(
            f"[{self.node_type}] Learning rate scaled by {factor}. New Actor LR: {self.optimizer_actor.param_groups[0]['lr']:.6f}")