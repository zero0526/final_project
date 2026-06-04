import torch
from typing import List, Tuple, Optional


class COMARolloutBuffer:
    """
    Buffer cho 1 agent trong khung Proposal-Refine.
    Lưu trữ đầy đủ thông tin cho cả Phase 1 (Proposal) và Phase 2 (Refine/Critic).
    """

    def __init__(self, max_size: int, node_type: str,
                 service_state_dim: int, action_dim: int, h_dim: int,
                 device: str = "cpu"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.node_type = node_type
        self.device = device

        # ── Fixed-size fields (Per-group/Per-step) ──
        self.service_state = torch.zeros((max_size, service_state_dim), dtype=torch.float32, device=device)
        self.next_service_state = torch.zeros((max_size, service_state_dim), dtype=torch.float32, device=device)
        self.prev_mf = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        self.curr_mf = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)

        # Metrics cho Reward & Done
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.done = torch.zeros((max_size, 1), dtype=torch.float32, device=device)

        # Metrics cho Training (PPO/COMA)
        self.log_prob = torch.zeros((max_size, 1), dtype=torch.float32, device=device)  # Log prob của final action
        self.value = torch.zeros((max_size, 1), dtype=torch.float32, device=device)  # V(s) hoặc Q baseline

        # ── Contextual Fields cho Refine & Critic (Per-group) ──
        # Lưu h_node (tải trọng kỳ vọng của nhóm + MF) để đưa vào Critic
        self.h_node = torch.zeros((max_size, h_dim), dtype=torch.float32, device=device)

        # Agent ID để phân biệt nếu dùng chung buffer cho nhiều loại agent
        self.agent_id = torch.zeros((max_size, 1), dtype=torch.int64, device=device)

        # ── Variable-length fields (Per-task within a group) ──
        self.task_state = [None] * max_size  # (N_i, task_dim)
        self.proposal_logits = [None] * max_size  # (N_i, action_dim) - Cần cho Q_p và Confidence
        self.actions = [None] * max_size  # (N_i,) - Final actions sau khi refine
        self.masks = [None] * max_size  # (N_i, u_action_dim) - Action masking

    def _to_tensor(self, x, dtype):
        if isinstance(x, torch.Tensor):
            return x.detach().to(device=self.device, dtype=dtype)
        return torch.tensor(x, dtype=dtype, device=self.device)

    def add(self, service_state, task_state, prev_mf, curr_mf,
            proposal_logits, actions, reward, next_service_state, done,
            log_prob, value, h_node, agent_id=0, mask=None):
        """
        Add single transition (one group offloading step).
        """
        idx = self.ptr

        # Fixed-size
        self.service_state[idx] = self._to_tensor(service_state, torch.float32)
        self.next_service_state[idx] = self._to_tensor(next_service_state, torch.float32)
        self.prev_mf[idx] = self._to_tensor(prev_mf, torch.float32)
        self.curr_mf[idx] = self._to_tensor(curr_mf, torch.float32)
        self.reward[idx] = self._to_tensor(reward, torch.float32).view(1)
        self.done[idx] = self._to_tensor(done, torch.float32).view(1)
        self.log_prob[idx] = self._to_tensor(log_prob, torch.float32).view(1)
        self.value[idx] = self._to_tensor(value, torch.float32).view(1)
        self.h_node[idx] = self._to_tensor(h_node, torch.float32)  # Critical for COMA Critic
        self.agent_id[idx] = torch.tensor([agent_id], dtype=torch.int64, device=self.device)

        # Variable-length
        if task_state is not None:
            self.task_state[idx] = self._to_tensor(task_state, torch.float32)
        if proposal_logits is not None:
            self.proposal_logits[idx] = self._to_tensor(proposal_logits, torch.float32)
        if actions is not None:
            self.actions[idx] = self._to_tensor(actions, torch.int64)
        if mask is not None:
            self.masks[idx] = self._to_tensor(mask, torch.float32)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_all(self):
        """
        Return ALL stored data.
        Variable-length fields are returned as concatenated tensors + lengths.
        """
        if self.size == 0:
            return None

        n = self.size

        # ── Fixed-size fields ──
        svc = self.service_state[:n].clone()
        next_svc = self.next_service_state[:n].clone()
        prev_mf = self.prev_mf[:n].clone()
        curr_mf = self.curr_mf[:n].clone()
        reward = self.reward[:n].clone()
        done = self.done[:n].clone()
        log_prob = self.log_prob[:n].clone()
        value = self.value[:n].clone()
        h_node = self.h_node[:n].clone()  # (n, action_dim)
        agent_id = self.agent_id[:n].clone().squeeze(-1)

        # ── Variable-length fields processing ──
        task_list, p_logits_list, action_list, mask_list = [], [], [], []
        task_lens, action_lens = [], []

        for i in range(n):
            if self.task_state[i] is not None:
                t = self.task_state[i]
                task_list.append(t)
                task_lens.append(t.shape[0])

                # Đảm bảo proposal_logits và actions khớp với số lượng task
                if self.proposal_logits[i] is not None:
                    p_logits_list.append(self.proposal_logits[i])
                if self.actions[i] is not None:
                    action_list.append(self.actions[i])
                    action_lens.append(self.actions[i].shape[0])
                if self.masks[i] is not None:
                    mask_list.append(self.masks[i])

        task_lens_t = torch.tensor(task_lens, dtype=torch.long, device=self.device)
        action_lens_t = torch.tensor(action_lens, dtype=torch.long,
                                     device=self.device) if action_lens else torch.zeros_like(task_lens_t)

        task_batch_cat = torch.cat(task_list, dim=0) if task_list else torch.zeros(0, 0, device=self.device)
        p_logits_cat = torch.cat(p_logits_list, dim=0) if p_logits_list else torch.zeros(0, 0, device=self.device)
        actions_cat = torch.cat(action_list, dim=0) if action_list else torch.zeros(0, dtype=torch.int64,
                                                                                    device=self.device)

        # Masks giữ nguyên dạng list để dễ xử lý padding sau này nếu cần
        masks_final = mask_list

        return (
            svc,  # 0  (n, svc_dim)
            task_batch_cat,  # 1  (total_tasks, task_dim)
            task_lens_t,  # 2  (n,)
            p_logits_cat,  # 3  (total_tasks, action_dim) - Cho Critic tính Q_p
            actions_cat,  # 4  (total_tasks,)       - Final actions
            action_lens_t,  # 5  (n,)
            prev_mf,  # 6  (n, mf_dim)
            curr_mf,  # 7  (n, mf_dim)
            reward,  # 8  (n, 1)
            next_svc,  # 9  (n, svc_dim)
            done,  # 10 (n, 1)
            log_prob,  # 11 (n, 1)
            value,  # 12 (n, 1)
            h_node,  # 13 (n, action_dim)     - Cho Critic COMA
            masks_final,  # 14 List[n]
            agent_id,  # 15 (n,)
        )

    def clear(self):
        self.ptr = 0
        self.size = 0
        self.task_state = [None] * self.max_size
        self.proposal_logits = [None] * self.max_size
        self.actions = [None] * self.max_size
        self.masks = [None] * self.max_size

    def __len__(self):
        return self.size


class MultiAgentCOMARolloutBuffer:
    def __init__(self, num_agents: int, node_type: str,
                 max_size_per_agent: int, service_state_dim: int,
                 action_dim: int,h_dim:int, device):
        self.num_agents = num_agents
        self.node_type = node_type
        self.device = device

        self.buffers = [
            COMARolloutBuffer(max_size_per_agent, node_type,
                          service_state_dim, action_dim, h_dim, device=device)
            for _ in range(num_agents)
        ]
        self.buffer_sizes = torch.zeros(num_agents, dtype=torch.long)
        self.total_size = 0

    def add_batch(self, service_states, task_states,
                  prev_mfs, curr_mfs, proposal_logits, actions,
                  rewards, next_service_states, dones,
                  log_probs, values, h_nodes, agent_ids, masks=None):
        """Add batch of transitions."""
        a_ids = agent_ids.view(-1)
        for i in range(len(a_ids)):
            a_id = int(a_ids[i])
            self.buffers[a_id].add(
                service_states[i], task_states[i],
                prev_mfs[i], curr_mfs[i],
                proposal_logits[i], actions[i],
                rewards[i], next_service_states[i], dones[i],
                log_probs[i], values[i],
                h_nodes[i], a_id, mask=masks[i] if masks else None
            )
            self.buffer_sizes[a_id] = self.buffers[a_id].size
        self.total_size = int(self.buffer_sizes.sum().item())

    def get_all_ready(self, min_size=1, agent_ids_pool=None):
        """Collect data from ready agents."""
        if agent_ids_pool is not None:
            pool = set(agent_ids_pool.cpu().numpy().tolist())
        else:
            pool = set(range(self.num_agents))

        ready_agents = sorted([
            a for a in pool
            if a < self.num_agents and self.buffers[a].size >= min_size
        ])

        if not ready_agents:
            return None

        all_samples = []
        for a_id in ready_agents:
            s = self.buffers[a_id].get_all()
            if s is not None:
                all_samples.append(s)

        if not all_samples:
            return None

        return self._collate(all_samples)

    def _collate(self, samples):
        """Collate list of get_all() outputs into single batch."""
        result = []
        # Mapping indices based on new get_all return tuple (0-15)
        concat_indices = [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15]
        list_indices = [14]  # masks
        len_indices = [2, 5]  # task_lens, action_lens

        for field_idx in range(16):
            if field_idx in concat_indices:
                result.append(torch.cat([s[field_idx] for s in samples], dim=0))
            elif field_idx in len_indices:
                result.append(torch.cat([s[field_idx] for s in samples], dim=0))
            elif field_idx in list_indices:
                all_masks = []
                for s in samples:
                    all_masks.extend(s[field_idx])
                result.append(all_masks)

        # Ensure order matches the expected output structure if needed
        # Currently returning in index order 0..15
        return tuple(result)

    def clear(self, agent_ids=None):
        if agent_ids is None:
            for buf in self.buffers:
                buf.clear()
            self.buffer_sizes.zero_()
            self.total_size = 0
        else:
            if not isinstance(agent_ids, torch.Tensor):
                agent_ids = torch.tensor(agent_ids, dtype=torch.long)
            for a_id in agent_ids.view(-1):
                a_id = int(a_id)
                self.buffers[a_id].clear()
                self.buffer_sizes[a_id] = 0
            self.total_size = int(self.buffer_sizes.sum().item())

    def __len__(self):
        return self.total_size