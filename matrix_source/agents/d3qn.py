import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple
from matrix_source.agents.ReplayBuffer import ReplayBuffer, MultiAgentReplayBuffer
from matrix_source.configs.configs import cfg

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
        # x: (Batch, In) or (Batch, 1, In)
        # indices: (Batch) long tensor
        if indices is None:
            # If no indices, default to shared (instance 0)
            indices = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            
        # Select weights and biases for the batch
        # w: (Batch, In, Out), b: (Batch, Out)
        w = self.weight[indices]
        
        # x.unsqueeze(1): (Batch, 1, In)
        # torch.bmm( (Batch, 1, In), (Batch, In, Out) ) -> (Batch, 1, Out)
        out = torch.bmm(x.unsqueeze(1), w).squeeze(1)
        
        if self.bias is not None:
            out += self.bias[indices]
        return out

class MultiInstanceRMSNorm(nn.Module):
    """
    Instance-specific Root Mean Square Layer Normalization.
    Each agent instance has its own learnable weight (gamma).
    """
    def __init__(self, num_instances, normalized_shape, eps=1e-6):
        super().__init__()
        self.num_instances = num_instances
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_instances, normalized_shape))

    def forward(self, x, indices):
        # x: (Batch, normalized_shape)
        # indices: (Batch)
        # RMS = sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        
        # Select gamma for each instance in the batch
        gamma = self.weight[indices]
        return x_norm * gamma

class DuelingNetwork(nn.Module):
    def __init__(self, state_dim, mf_dim, action_dim, hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        self.num_instances = num_instances

        # Independent normalized representation for each agent
        self.l1 = MultiInstanceLinear(num_instances, state_dim + mf_dim, h1)
        self.norm1 = MultiInstanceRMSNorm(num_instances, h1)
        self.l2 = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2 = MultiInstanceRMSNorm(num_instances, h2)

        # Value stream
        self.v1 = MultiInstanceLinear(num_instances, h2, h2)
        self.v2 = MultiInstanceLinear(num_instances, h2, 1)

        # Advantage stream
        self.a1 = MultiInstanceLinear(num_instances, h2, h2)
        self.a2 = MultiInstanceLinear(num_instances, h2, action_dim)

    def forward(self, state, pred_mf, indices=None):
        x = torch.cat([state, pred_mf], dim=-1)
        
        # Independent normalization and activation
        x = F.silu(self.norm1(self.l1(x, indices), indices))
        x = F.silu(self.norm2(self.l2(x, indices), indices))
        
        # Heads
        V = self.v2(F.silu(self.v1(x, indices)), indices)
        A = self.a2(F.silu(self.a1(x, indices)), indices)
        
        return V + (A - A.mean(dim=-1, keepdim=True))

    def get_base_params(self):
        # Used for SCAFFOLD or weight access
        return list(self.parameters())

class MF(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, num_instances=1):
        super().__init__()
        h = hidden_sizes[0]
        self.num_instances = num_instances
        
        self.l1 = MultiInstanceLinear(num_instances, input_size, h)
        self.norm = MultiInstanceRMSNorm(num_instances, h)
        self.l2 = MultiInstanceLinear(num_instances, h, output_size)

    def forward(self, x, indices=None):
        x = F.silu(self.norm(self.l1(x, indices), indices))
        return torch.sigmoid(self.l2(x, indices)) # Constrain MF to [0, 1] range

class D3QNAgent:
    def __init__(self, node_id, node_type, state_dim, action_dim, u_action_dim, mf_hidden_sizes, mf_lr, buffer_min_size,
                 hidden_sizes=(128, 64), lr=1e-4, gamma=0.99, alpha=0.005, buffer_size=100000, batch_size=64,
                 exclude_zero=False, num_instances=1, device=None):
        self.node_id = node_id
        self.node_type = node_type
        self.action_dim = action_dim
        self.u_action_dim = u_action_dim
        self.gamma = gamma
        self.alpha = float(alpha)
        self.batch_size = batch_size
        self.exclude_zero = exclude_zero
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.num_instances = num_instances
        
        self.min_batch_size = buffer_min_size

        # Use num_instances to create parallel independent internal models
        self.eval_net = DuelingNetwork(state_dim, action_dim, u_action_dim, hidden_sizes, num_instances).to(self.device)
        self.target_net = DuelingNetwork(state_dim, action_dim, u_action_dim, hidden_sizes, num_instances).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.mf_net = MF(state_dim + action_dim, action_dim, mf_hidden_sizes, num_instances).to(self.device)
        
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)
        self.mf_optimizer = optim.Adam(self.mf_net.parameters(), lr=mf_lr)
        self.loss_fn = nn.SmoothL1Loss() # Huber Loss is more robust to large reward scales
        
        # Each agent instance gets its own partitioned buffer to prevent "noise" and ensure fair training
        self.memory = MultiAgentReplayBuffer(num_instances,node_type, buffer_size, state_dim, action_dim, self.device)

        # Logging
        self.prev_loss = 0.0
        self.learn_step_counter = 0

    def choose_action(self, state, prev_mf, epsilon, zeta, mask=None, agent_idx=0):
        # Single agent usage (fallback or legacy)
        idx_tensor = torch.tensor([agent_idx], device=self.device)
        actions = self.choose_action_batch(
            state.unsqueeze(0) if not torch.is_tensor(state) else state.detach().unsqueeze(0),
            prev_mf.unsqueeze(0) if not torch.is_tensor(prev_mf) else prev_mf.detach().unsqueeze(0),
            zeta, 
            masks_batch=mask.unsqueeze(0) if mask is not None else None,
            agent_indices=idx_tensor
        )
        return int(actions[0])

    def choose_action_batch(self, states_batch, prev_mfs_batch, zeta, masks_batch=None, agent_indices=None):
        batch_size = states_batch.shape[0]
        if agent_indices is None:
            agent_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        else:
            agent_indices = agent_indices.to(self.device).view(-1)
            
        if masks_batch is not None:
            masks_batch = masks_batch.to(self.device)
            
        # Ensure states and mfs are on the correct device
        if not torch.is_tensor(states_batch):
            states_batch = torch.as_tensor(states_batch, dtype=torch.float32, device=self.device)
        else:
            states_batch = states_batch.to(self.device)
            
        if not torch.is_tensor(prev_mfs_batch):
            prev_mfs_batch = torch.as_tensor(prev_mfs_batch, dtype=torch.float32, device=self.device)
        else:
            prev_mfs_batch = prev_mfs_batch.to(self.device)

        # 1. Per-Agent Cold-Start Check
        is_policy_agent = torch.tensor([
            self.memory.get_len(aid.item()) >= self.min_batch_size 
            for aid in agent_indices
        ], device=self.device)
        
        final_actions = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # 2. Handle Cold-Start Agents (Random Exploration)
        cold_mask = ~is_policy_agent
        if cold_mask.any():
            indices = cold_mask.nonzero(as_tuple=True)[0]
            if masks_batch is not None:
                # Random choice within valid mask
                m = masks_batch[indices]
                probs = m / m.sum(dim=1, keepdim=True).clamp(min=1e-8)
                final_actions[indices] = torch.multinomial(probs, 1).squeeze(1)
            else:
                final_actions[indices] = torch.randint(0, self.u_action_dim, (len(indices),), device=self.device)

        # 3. Handle Policy Agents (Neural Network)
        policy_mask = is_policy_agent
        if policy_mask.any():
            indices = policy_mask.nonzero(as_tuple=True)[0]
            s_subset = states_batch[indices]
            mf_subset = prev_mfs_batch[indices]
            aid_subset = agent_indices[indices]

            with torch.no_grad():
                pred_mf = self.mf_net(torch.cat([s_subset, mf_subset], dim=-1), indices=aid_subset)
                q_values = self.eval_net(s_subset, pred_mf, indices=aid_subset)

                if masks_batch is not None:
                    m = masks_batch[indices]
                    q_values = q_values + (m - 1.0) * 1e10
                
                if self.exclude_zero and self.u_action_dim > 1:
                    q_values[:, 0] -= 1e10
                
                # zeta factor controls the exploration temperature
                probs = torch.softmax(q_values * zeta, dim=1)
                final_actions[indices] = torch.multinomial(probs, 1).squeeze(1)

        return final_actions.tolist()

    def store_transition_train_mf_batch(self, states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, agent_ids, masks=None, next_masks=None):
        # MF learning update
        loss = self.learn_mf_batch(states, prev_mfs, curr_mfs, agent_ids)
        # Store in buffer
        self.memory.add_batch(states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, agent_ids, masks=masks, next_masks=next_masks)
        return loss

    def learn_mf_batch(self, states_batch, prev_mf_batch, ground_truth_mf_batch, agent_ids):
        s = states_batch.to(self.device) if torch.is_tensor(states_batch) else torch.FloatTensor(states_batch).to(self.device)
        pmf = prev_mf_batch.to(self.device) if torch.is_tensor(prev_mf_batch) else torch.FloatTensor(prev_mf_batch).to(self.device)
        gt_mf = ground_truth_mf_batch.to(self.device) if torch.is_tensor(ground_truth_mf_batch) else torch.FloatTensor(ground_truth_mf_batch).to(self.device)
        
        pred_mf = self.mf_net(torch.cat([s, pmf], dim=-1), indices=agent_ids)
        loss = self.loss_fn(pred_mf, gt_mf)
        
        self.mf_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mf_net.parameters(), max_norm=2.0)
        self.mf_optimizer.step()
        return loss.item()

    def learn(self, agents_ids: torch.Tensor = None):
        # 1. Identify the pool of agents that have enough data
        ready_pool = (self.memory.buffer_sizes >= self.min_batch_size).nonzero(as_tuple=True)[0]
        
        if agents_ids is not None:
            # Filter specifically for the requested agents that are also ready
            if not isinstance(agents_ids, torch.Tensor):
                agents_ids = torch.tensor(agents_ids, device=self.device)
            
            # Move to correct device for comparison
            agents_ids = agents_ids.to(self.device).view(-1)
            mask = torch.isin(agents_ids, ready_pool)
            target_agents = agents_ids[mask]
        else:
            # Default to all ready agents
            target_agents = ready_pool

        if len(target_agents) == 0:
            return None

        # 2. Sample data: states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, agent_ids, masks, next_masks
        states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, agent_ids, masks, next_masks = \
            self.memory.sample(self.batch_size, agent_ids=target_agents)

        # 1. Train MF (prediction and current state)
        pred_curr_mfs = self.mf_net(torch.cat([states, prev_mfs], dim=-1), indices=agent_ids)

        # 2. DQN update
        q_eval = self.eval_net(states, pred_curr_mfs.detach(), indices=agent_ids).gather(1, actions)

        with torch.no_grad():
            next_pred_mfs = self.mf_net(torch.cat([next_states, curr_mfs], dim=-1), indices=agent_ids)
            q_next_pre = self.eval_net(next_states, next_pred_mfs, indices=agent_ids)
            
            # Application of NEXT-state masks for action selection in Q-target
            if next_masks is not None:
                q_next_pre = q_next_pre + (next_masks - 1.0) * 1e10
            
            next_actions = q_next_pre.argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_states, next_pred_mfs, indices=agent_ids).gather(1, next_actions)
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = self.loss_fn(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update logging/target
        self.learn_step_counter += 1
        step=10
        if self.node_type=="Terminal_Group":step=100
        if self.learn_step_counter % step == 0:
            avg_q = q_eval.mean().item()
            print(f"[{self.node_type} Group] Step {self.learn_step_counter:5d} | TD Loss: {loss.item():.5f} | Avg Q: {avg_q:.3f}")
        
        self._soft_update()
        return loss.item()

    def _soft_update(self):
        with torch.no_grad():
            for target_param, eval_param in zip(self.target_net.parameters(), self.eval_net.parameters()):
                target_param.data.copy_(self.alpha * eval_param.data + (1.0 - self.alpha) * target_param.data)



