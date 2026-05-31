import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from matrix_source.agents.buffer.policy_replay_buffer import MultiAgentPolicyBuffer
from matrix_source.trainers.ppo_stategy import compute_gae


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





class GroupedMFNetwork(nn.Module):
    """MF Network shared per Group (Edge). Each Edge node has its own model weights."""
    def __init__(self, num_groups, input_dim, output_dim, hidden_sizes):
        super().__init__()
        h1, h2 = hidden_sizes
        self.fc1 = MultiInstanceLinear(num_groups, input_dim, h1)
        self.norm = MultiInstanceRMSNorm(num_groups, h1)
        self.fc2 = MultiInstanceLinear(num_groups, h1, h2)
        self.out = MultiInstanceLinear(num_groups, h2, output_dim)

    def forward(self, x, group_indices):
        x = F.silu(self.norm(self.fc1(x, group_indices), group_indices))
        x = F.silu(self.fc2(x, group_indices))
        return torch.sigmoid(self.out(x, group_indices))


class CriticBackbone(nn.Module):
    def __init__(self, state_dim, mf_dim, hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        self.fc1 = MultiInstanceLinear(num_instances, state_dim + mf_dim, h1)
        self.norm1 = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2 = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2 = MultiInstanceRMSNorm(num_instances, h2)

    def forward(self, state, mf, indices=None):
        x = torch.cat([state, mf], dim=-1)
        x = F.silu(self.norm1(self.fc1(x, indices), indices))
        x = F.silu(self.norm2(self.fc2(x, indices), indices))
        return x

class CriticHead(nn.Module):
    def __init__(self, h2, num_instances=1):
        super().__init__()
        self.value_head = MultiInstanceLinear(num_instances, h2, 1)

    def forward(self, x, indices=None):
        return self.value_head(x, indices).squeeze(-1)

class FedRepCritic(nn.Module):
    def __init__(self, state_dim, mf_dim, hidden_sizes, num_instances):
        super().__init__()
        _, h2 = hidden_sizes
        self.backbone = CriticBackbone(state_dim, mf_dim, hidden_sizes, num_instances)
        self.head = CriticHead(h2, num_instances)

    def forward(self, state, mf, agent_indices):
        latent = self.backbone(state, mf, agent_indices)
        return self.head(latent, agent_indices)

class PPOSCAFFOLDREPAgent:
    def __init__(self, node_id, node_type, state_dim, action_dim, u_action_dim,
                 mf_hidden_sizes, mf_lr, buffer_min_size, num_groups,
                 hidden_sizes=(128, 64),
                 lr=3e-4, gamma=0.99, alpha=0.005, buffer_size=100000, batch_size=64,
                 lam=0.95, clip_eps=0.2, k_epochs=5, entropy_coef=0.01,
                 exclude_zero=False, num_instances=1, device=None):

        self.node_id = node_id
        self.node_type = node_type
        self.num_groups = num_groups
        self.scaffold_beta = 0.9

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.num_instances = num_instances
        self.action_dim = action_dim
        self.u_action_dim = u_action_dim
        self.exclude_zero = exclude_zero
        
        # Simplified Entropy Control
        self.initial_entropy_coef = entropy_coef
        self.entropy_coef = entropy_coef
        self.entropy_decay_rate = 0.99
        self.min_entropy_coef = 0.001
        self.zeta= 0.6
        self.zeta_decay_rate = 0.99
        self.max_zeta = 5

        self.gamma = gamma
        self.lmbda = lam
        self.eps_clip = clip_eps
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.min_batch_size = buffer_min_size
        self.entropy_coef = entropy_coef
        self.alpha = alpha

        # Networks
        self.actor = MultiInstanceActor(state_dim, action_dim, u_action_dim, hidden_sizes, num_instances).to(self.device)
        self.critic = FedRepCritic(state_dim, action_dim, hidden_sizes, num_instances).to(self.device)
        self.mf_net = GroupedMFNetwork(num_groups, state_dim + action_dim, action_dim, mf_hidden_sizes).to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.bone_optimizer = optim.Adam(self.critic.backbone.parameters(), lr=lr)
        self.head_optimizer  = optim.Adam(self.critic.head.parameters(), lr=lr)
        self.mf_optimizer    = optim.Adam(self.mf_net.parameters(), lr=mf_lr)

        self.loss_fn = nn.SmoothL1Loss()

        self.memory = MultiAgentPolicyBuffer(num_instances, buffer_size, state_dim, action_dim, u_action_dim, self.device)
        self.learn_step_counter = 0

        # SCAFFOLD Control Variates for Critic Backbone (Encoder)
        self.bone_params = list(self.critic.backbone.parameters())
        self.c_b_local  = [torch.zeros_like(p, device=self.device) for p in self.bone_params]
        self.c_b_global = [torch.zeros_like(p, device=self.device) for p in self.bone_params]
        self.grad_b_sum = [torch.zeros_like(p, device=self.device) for p in self.bone_params]
        self.steps_in_round = torch.zeros(num_instances, dtype=torch.long, device=self.device)

    def choose_action(self, state, prev_mf, epsilon, mask=None, agent_idx=0, group_idx=0, zeta=1.0):
        # Wraps batch method for single agent
        idx_tensor = torch.tensor([agent_idx], device=self.device)
        g_idx_tensor = torch.tensor([group_idx], device=self.device)
        actions, _, _ = self.choose_action_batch(
            state.unsqueeze(0) if not torch.is_tensor(state) else state.detach().unsqueeze(0),
            prev_mf.unsqueeze(0) if not torch.is_tensor(prev_mf) else prev_mf.detach().unsqueeze(0),
            masks_batch=mask.unsqueeze(0) if mask is not None else None,
            agent_indices=idx_tensor, group_indices=g_idx_tensor,
            zeta=zeta
        )
        return int(actions[0])

    def choose_action_batch(self, states, mfs, masks_batch=None, agent_indices=None, group_indices=None, deterministic=False, zeta=1.0):
        batch_size = states.shape[0]
        if agent_indices is None:
            agent_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        if group_indices is None:
            group_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        agent_indices = agent_indices.to(self.device).view(-1)
        group_indices = group_indices.to(self.device).view(-1)
        states = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        mfs = torch.as_tensor(mfs, device=self.device, dtype=torch.float32)

        if masks_batch is not None:
            masks_batch = masks_batch.to(self.device)

        with torch.no_grad():
            pred_mfs = self.mf_net(torch.cat([states, mfs], dim=-1), group_indices=group_indices)
            logits = self.actor(states, pred_mfs, indices=agent_indices)
            logits = logits * self.zeta
            values = self.critic(states, pred_mfs, agent_indices=agent_indices)

            if masks_batch is not None:
                logits = logits.masked_fill(masks_batch == 0, -1e9)

            if self.exclude_zero and self.u_action_dim > 1:
                zero_mask = torch.zeros_like(logits, dtype=torch.bool)
                zero_mask[:, 0] = True
                logits = logits.masked_fill(zero_mask, -1e9)

            if deterministic:
                actions = logits.argmax(dim=-1)
                log_probs = torch.zeros(batch_size, device=self.device)
            else:
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

        return actions, log_probs.detach(), values.detach()

    def store_transition_train_mf_batch(self, states, prev_mfs, curr_mfs, actions, rewards, next_states, dones,
                                        agent_ids, group_ids, log_prob, value, masks=None):
        self.memory.add_batch(states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, log_prob,
                              value, agent_ids, masks=masks)

    def learn_mf_batch(self, states, prev_mfs, ground_truth_mfs, group_ids):
        s = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        pmf = torch.as_tensor(prev_mfs, device=self.device, dtype=torch.float32)
        gt_mf = torch.as_tensor(ground_truth_mfs, device=self.device, dtype=torch.float32)
        g_ids = group_ids.to(self.device).view(-1)

        pred_mf = self.mf_net(torch.cat([s, pmf], dim=-1), group_indices=g_ids)
        loss = F.mse_loss(pred_mf, gt_mf)

        self.mf_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mf_net.parameters(), max_norm=5.0)
        self.mf_optimizer.step()
        return loss.item()

    def _compute_gae(self, states, prev_mfs, curr_mfs, rewards, next_states, dones, old_values, agent_ids, group_ids):
        """Compute GAE, Returns and prediction MF for the whole dataset."""
        with torch.no_grad():
            pred_mfs = self.mf_net(torch.cat([states, prev_mfs], dim=-1), group_indices=group_ids)
            next_pred_mfs = self.mf_net(torch.cat([next_states, curr_mfs], dim=-1), group_indices=group_ids)

            next_values = self.critic(next_states, next_pred_mfs, agent_indices=agent_ids)
            advantages = compute_gae(rewards, next_values, old_values, dones, agent_ids, self.gamma, self.lmbda)
            returns = advantages + old_values
            
            if advantages.shape[0] > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
        return pred_mfs, advantages, returns

    def save_base_initial(self):
        """Reset accumulators at the start of each federated round."""
        with torch.no_grad():
            for g in self.grad_b_sum:
                g.zero_()
            self.steps_in_round.zero_()

    def get_backbone_state(self, instance_ids: torch.Tensor):
        return [p.data[instance_ids].clone() for p in self.critic.backbone.parameters()]

    def set_backbone_state(self, instance_ids: torch.Tensor, param_list):
        with torch.no_grad():
            for p, new_val in zip(self.critic.backbone.parameters(), param_list):
                p.data[instance_ids] = new_val

    def get_c_b_local(self, instance_ids):
        return [c[instance_ids].clone() for c in self.c_b_local]

    def set_c_b_global(self, instance_ids, c_b_global_list):
        with torch.no_grad():
            for c_g, new_val in zip(self.c_b_global, c_b_global_list):
                c_g[instance_ids] = new_val

    def update_local_cvariates(self, instance_ids: torch.Tensor):
        with torch.no_grad():
            active_mask = self.steps_in_round[instance_ids] > 0
            if not active_mask.any():
                return
            active_ids = instance_ids[active_mask]
            steps = self.steps_in_round[active_ids].float()

            for i in range(len(self.c_b_local)):
                K = steps.view(-1, *([1] * (self.grad_b_sum[i].dim() - 1)))
                self.c_b_local[i][active_ids] = (
                    self.c_b_local[i][active_ids]
                    - self.c_b_global[i][active_ids]
                    + self.grad_b_sum[i][active_ids] / K
                )

    def learn(self, agents_ids: torch.Tensor = None, round_idx: int = 0, group_ids: torch.Tensor = None, zeta=1.0):
        ready_pool = (self.memory.buffer_sizes >= self.min_batch_size).nonzero(as_tuple=True)[0]
        if agents_ids is not None:
            agents_ids = agents_ids.to(self.device).view(-1)
            target_agents = agents_ids[torch.isin(agents_ids, ready_pool)]
            target_agents = torch.unique(target_agents)
        else:
            target_agents = ready_pool

        if len(target_agents) == 0: return None

        data = self.memory.get_all_ready(min_size=self.min_batch_size, agent_ids_pool=target_agents)
        if data is None: return None

        states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, old_log_probs, old_values, masks, agent_ids = data
        
        # Squeeze
        actions = actions.squeeze(-1)
        old_log_probs = old_log_probs.squeeze(-1)
        old_values = old_values.squeeze(-1)
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)
        
        batch_group_ids = group_ids[agent_ids].to(self.device)

        # 0. Save Base Initial: Reset grad accumulators and steps at start of round
        self.save_base_initial()

        # 1. Compute GAE
        pred_mfs, advantages, returns = self._compute_gae(
            states, prev_mfs, curr_mfs, rewards, next_states, dones, old_values, agent_ids, batch_group_ids
        )

        dataset_size = states.shape[0]
        total_v_loss = 0
        total_batches = 0

        # 2. PPO Mini-batch Update Epochs
        for _ in range(self.k_epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                b_states = states[idx]
                b_prev_mfs = prev_mfs[idx]
                b_actions = actions[idx]
                b_old_log_probs = old_log_probs[idx]
                b_advantages = advantages[idx]
                b_returns = returns[idx]
                b_agent_ids = agent_ids[idx]
                b_masks = masks[idx]
                b_group_ids = batch_group_ids[idx]

                # Re-predict MF for mini-batch
                b_pred_mfs = self.mf_net(torch.cat([b_states, b_prev_mfs], dim=-1), group_indices=b_group_ids).detach()

                # --- Critic Update (Delayed Backbone, Direct Head) ---
                self.bone_optimizer.zero_grad()
                self.head_optimizer.zero_grad()
                
                v_curr = self.critic(b_states, b_pred_mfs, b_agent_ids)
                v_loss = F.mse_loss(v_curr, b_returns)
                v_loss.backward()

                # A. Accumulate raw backbone gradients (DO NOT STEP)
                with torch.no_grad():
                    u_ids = torch.unique(b_agent_ids)
                    for i, p in enumerate(self.bone_params):
                        if p.grad is not None:
                            # TÍCH LŨY GRADIENT THÔ (CHƯA SỬA)
                            self.grad_b_sum[i][u_ids] += p.grad.data[u_ids]
                    self.steps_in_round[u_ids] += 1
                
                # B. Update Head (STEP HEAD NORMALLY)
                torch.nn.utils.clip_grad_norm_(self.critic.head.parameters(), max_norm=1.0)
                self.head_optimizer.step()
                
                # Reset backbone grads so they don't accumulate in .grad attribute
                self.bone_optimizer.zero_grad()

                # --- Actor Update (PPO) ---
                log_probs, entropy = self.actor.evaluate(b_states, b_pred_mfs, b_actions, masks=b_masks, indices=b_agent_ids, exclude_zero=self.exclude_zero, zeta=self.zeta)
                ratio = torch.exp(log_probs - b_old_log_probs)
                
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.optimizer_actor.step()

                total_v_loss += v_loss.item()
                total_batches += 1

        # 3. Post-PPO SCAFFOLD Update for Backbone (CHẠY 1 LẦN DUY NHẤT SAU LOOP)
        self.update_local_cvariates(target_agents) # Cập nhật c_local dựa trên grad tích lũy

        with torch.no_grad():
            active_mask = self.steps_in_round > 0
            if active_mask.any():
                active_ids = torch.where(active_mask)[0]
                steps = self.steps_in_round[active_ids].float()
                
                # Tính gradient trung bình: g_avg = grad_sum / K
                self.bone_optimizer.zero_grad()
                for i, p in enumerate(self.bone_params):
                    K = steps.view(-1, *([1] * (p.dim() - 1)))
                    avg_grad = self.grad_b_sum[i][active_ids] / K
                    
                    # Áp dụng công thức SCAFFOLD: g_corrected = g_avg + (c_global - c_local)
                    p.grad = torch.zeros_like(p)
                    p.grad.data[active_ids] = avg_grad + (self.c_b_global[i][active_ids] - self.c_b_local[i][active_ids])
                
                # STEP BACKBONE
                torch.nn.utils.clip_grad_norm_(self.bone_params, max_norm=1.0)
                self.bone_optimizer.step()

        # 4. Simplified Entropy Decay
        self.entropy_coef = max(self.entropy_coef * self.entropy_decay_rate, self.min_entropy_coef)
        self.zeta = min(self.zeta * 1/self.zeta_decay_rate, self.max_zeta)
        self.memory.clear()
        self.learn_step_counter += 1
        return total_v_loss / total_batches if total_batches > 0 else 0.0

    def save(self, path):
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic_backbone': self.critic.backbone.state_dict(),
            'critic_head': self.critic.head.state_dict(),
            'mf_net': self.mf_net.state_dict(),
            'c_b_local': self.c_b_local,
            'c_b_global': self.c_b_global,
            'learn_step': self.learn_step_counter
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.backbone.load_state_dict(checkpoint['critic_backbone'])
        self.critic.head.load_state_dict(checkpoint['critic_head'])
        self.mf_net.load_state_dict(checkpoint['mf_net'])
        self.c_b_local = checkpoint['c_b_local']
        self.c_b_global = checkpoint['c_b_global']
        self.learn_step_counter = checkpoint.get('learn_step', 0)

    def set_lr_factor(self, factor):
        for opt in [self.optimizer_actor, self.head_optimizer, self.bone_optimizer, self.mf_optimizer]:
            for param_group in opt.param_groups:
                param_group['lr'] *= factor
