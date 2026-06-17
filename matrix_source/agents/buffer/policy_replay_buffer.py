import torch
import numpy as np


class PolicyReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, u_action_dim, device="cpu"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device
        self.u_action_dim = u_action_dim

        # Pre-allocate with torch tensors
        self.state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.prev_mf = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        self.curr_mf = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        self.action = torch.zeros((max_size, 1), dtype=torch.int64, device=device)
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.next_state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.done = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.log_prob = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.value = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.mask = torch.zeros((max_size, u_action_dim), dtype=torch.float32, device=device)
        self.agent_id = torch.zeros((max_size, 1), dtype=torch.int64, device=device)

    def _to_tensor(self, x, dtype):
        if torch.is_tensor(x):
            return x.detach().to(device=self.device, dtype=dtype)
        return torch.tensor(x, dtype=dtype, device=self.device)

    def add(self, state, prev_mf, curr_mf, action, reward, next_state, done, log_prob, value, agent_id=0, mask=None):
        self.state[self.ptr] = self._to_tensor(state, torch.float32)
        self.prev_mf[self.ptr] = self._to_tensor(prev_mf, torch.float32)
        self.curr_mf[self.ptr] = self._to_tensor(curr_mf, torch.float32)
        self.action[self.ptr] = self._to_tensor(action, torch.int64)
        self.reward[self.ptr] = self._to_tensor(reward, torch.float32)
        self.next_state[self.ptr] = self._to_tensor(next_state, torch.float32)
        self.done[self.ptr] = self._to_tensor(done, torch.float32)
        self.log_prob[self.ptr] = self._to_tensor(log_prob, torch.float32)
        self.value[self.ptr] = self._to_tensor(value, torch.float32)
        if mask is not None:
            self.mask[self.ptr] = self._to_tensor(mask, torch.float32)
        else:
            self.mask[self.ptr] = 1.0  # Default all valid
        self.agent_id[self.ptr] = self._to_tensor(agent_id, torch.int64)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def clear(self):
        self.ptr = 0
        self.size = 0

    def get_all(self):
        if self.size == 0:
            return None
        return (
            self.state[:self.size],
            self.prev_mf[:self.size],
            self.curr_mf[:self.size],
            self.action[:self.size],
            self.reward[:self.size],
            self.next_state[:self.size],
            self.done[:self.size],
            self.log_prob[:self.size],
            self.value[:self.size],
            self.mask[:self.size],
            self.agent_id[:self.size].squeeze(1)
        )


class MultiAgentPolicyBuffer:
    def __init__(self, num_agents, max_size_per_agent, state_dim, action_dim, u_action_dim, device="cpu"):
        self.num_agents = num_agents
        self.device = device
        self.buffers = [
            PolicyReplayBuffer(max_size_per_agent, state_dim, action_dim, u_action_dim, device)
            for _ in range(num_agents)
        ]
        self.buffer_sizes = torch.zeros(num_agents, dtype=torch.long, device=device)
        self.total_size = 0

    def add_batch(self, states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, log_probs, values, agent_ids,
                  masks=None):
        a_ids = agent_ids.view(-1)
        for i in range(len(a_ids)):
            a_id = int(a_ids[i])
            m = masks[i] if masks is not None else None
            self.buffers[a_id].add(
                states[i], prev_mfs[i], curr_mfs[i], actions[i], rewards[i], next_states[i], dones[i],
                log_probs[i], values[i], a_id, mask=m
            )
            self.buffer_sizes[a_id] = self.buffers[a_id].size
        self.total_size = self.buffer_sizes.sum().item()

    def clear(self):
        for b in self.buffers:
            b.clear()
        self.buffer_sizes.zero_()
        self.total_size = 0

    def get_all_ready(self, min_size=1, agent_ids_pool=None):
        """Returns all transitions for specified agents that meet the min_size."""
        if agent_ids_pool is None:
            agent_ids_pool = torch.arange(self.num_agents, device=self.device)

        ready_mask = (self.buffer_sizes >= min_size)
        ready_agents = agent_ids_pool[ready_mask[agent_ids_pool]]

        if len(ready_agents) == 0:
            return None

        samples = [self.buffers[int(a_id)].get_all() for a_id in ready_agents]
        collated = []
        for i in range(11):  # 11 fields
            collated.append(torch.cat([s[i] for s in samples if s is not None], dim=0))

        return tuple(collated)

    def __len__(self):
        return self.buffer_sizes.min().item()