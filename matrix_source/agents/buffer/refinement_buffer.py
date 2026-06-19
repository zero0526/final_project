import torch
from typing import List, Tuple, Optional


class RolloutBuffer:
    """
    Buffer cho 1 agent. Lưu cả fixed-size và variable-length fields.

    Fixed-size: service_state, next_service_state, prev_mf, curr_mf,
                reward, done, log_prob, value, agent_id
    Variable:   task_state, action, mask (per-task, khác nhau giữa các sample)
    """

    def __init__(self, max_size: int, node_type: str,
                 service_state_dim: int, action_dim: int,
                 device: str = "cpu"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.node_type = node_type
        self.device = device

        # ── Fixed-size fields ──
        self.service_state    = torch.zeros((max_size, service_state_dim),
                                            dtype=torch.float32, device=device)
        self.next_service_state = torch.zeros((max_size, service_state_dim),
                                              dtype=torch.float32, device=device)
        self.prev_mf  = torch.zeros((max_size, action_dim),
                                    dtype=torch.float32, device=device)
        self.curr_mf  = torch.zeros((max_size, action_dim),
                                    dtype=torch.float32, device=device)
        self.reward   = torch.zeros((max_size, 1),
                                    dtype=torch.float32, device=device)
        self.done     = torch.zeros((max_size, 1),
                                    dtype=torch.float32, device=device)
        self.log_prob = torch.zeros((max_size, 1),
                                    dtype=torch.float32, device=device)
        self.value    = torch.zeros((max_size, 1),
                                    dtype=torch.float32, device=device)
        self.agent_id = torch.zeros((max_size, 1),
                                    dtype=torch.int64, device=device)
        self.service_idx = torch.zeros((max_size, 1),
                                     dtype=torch.int64, device=device)
        
        # ── Histogram (Mean Field h_t) ──
        # M = service_state_dim // 2
        M = service_state_dim // 2

        # ── Variable-length fields (list of tensors) ──
        self.task_state  = [None] * max_size   # (N_i, task_dim) mỗi entry
        self.batch_sizes = [None] * max_size   # (N_i,) mỗi entry (batch_size của từng task)
        self.action      = [None] * max_size   # (N_i,) mỗi entry
        self.mask        = [None] * max_size   # (N_i, u_action_dim) hoặc None

    def _to_tensor(self, x, dtype):
        if isinstance(x, torch.Tensor):
            return x.detach().to(device=self.device, dtype=dtype)
        return torch.tensor(x, dtype=dtype, device=self.device)

    def _to_scalar(self, x, dtype, reduce='mean'):
        """Convert potential tensor inputs to scalar tensors."""
        t = self._to_tensor(x, dtype)
        if t.numel() > 1:
            if reduce == 'mean':
                t = t.mean()
            elif reduce == 'sum':
                t = t.sum()
        return t.view(1)

    def add(self, service_state, task_state, prev_mf, curr_mf,
            action, reward, next_service_state, done,
            log_prob, value, agent_id=0, service_idx=0, mask=None, task_batch_sizes=None):
        """Add single transition."""

        # Fixed-size
        self.service_state[self.ptr]     = self._to_tensor(service_state, torch.float32)
        self.next_service_state[self.ptr] = self._to_tensor(next_service_state, torch.float32)
        self.prev_mf[self.ptr]  = self._to_tensor(prev_mf, torch.float32)
        self.curr_mf[self.ptr]  = self._to_tensor(curr_mf, torch.float32)
        # Scalar fields — Use MEAN aggregation for training stability
        self.reward[self.ptr]   = self._to_scalar(reward,   torch.float32, 'mean')
        self.done[self.ptr]     = self._to_scalar(done,     torch.float32, 'mean')
        self.log_prob[self.ptr] = self._to_scalar(log_prob, torch.float32, 'mean')
        self.value[self.ptr]    = self._to_scalar(value,    torch.float32, 'mean')
        self.agent_id[self.ptr] = self._to_tensor(agent_id, torch.int64).view(1)
        self.service_idx[self.ptr] = self._to_tensor(service_idx, torch.int64).view(1)

        # Variable-length — store as-is
        if task_state is not None:
            self.task_state[self.ptr] = self._to_tensor(task_state, torch.float32)
        if task_batch_sizes is not None:
            self.batch_sizes[self.ptr] = self._to_tensor(task_batch_sizes, torch.float32)
        if action is not None:
            self.action[self.ptr] = self._to_tensor(action, torch.int64)
        if mask is not None:
            self.mask[self.ptr] = self._to_tensor(mask, torch.float32)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_all(self):
        """
        Return ALL stored data as batch.
        Variable-length fields returned as concatenated tensor + lengths.

        Returns:
            Tuple of 14 items:
            0  service_states    (size, service_state_dim)
            1  task_batch_cat    (total_tasks, task_dim)
            2  task_lens         (size,)
            3  actions_cat       (total_tasks,)       — per-task actions
            4  action_lens       (size,)              — same as task_lens
            5  prev_mfs          (size, action_dim)
            6  curr_mfs          (size, action_dim)
            7  rewards           (size, 1)
            8  next_service_states (size, service_state_dim)
            9  dones             (size, 1)
            10 log_probs         (size, 1)
            11 values            (size, 1)
            12 masks             List[size] of (N_i, u_action_dim) or None
            13 agent_ids         (size,)
            14 batch_sizes_cat   (total_tasks,)       — per-task batch sizes
            15 service_indices   (size,)
        """
        if self.size == 0:
            return None

        n = self.size

        # ── Fixed-size fields ──
        svc       = self.service_state[:n].clone()
        next_svc  = self.next_service_state[:n].clone()
        prev_mf   = self.prev_mf[:n].clone()
        curr_mf   = self.curr_mf[:n].clone()
        reward    = self.reward[:n].clone()
        done      = self.done[:n].clone()
        log_prob  = self.log_prob[:n].clone()
        value     = self.value[:n].clone()
        agent_id  = self.agent_id[:n].clone().squeeze(-1)  # (n,)
        svc_idx   = self.service_idx[:n].clone().squeeze(-1) # (n,)

        # ── Task states → concat + lens ──
        task_list = []
        task_lens_list = []
        for i in range(n):
            t = self.task_state[i]
            if t is not None:
                task_list.append(t)
                task_lens_list.append(t.shape[0])
            else:
                task_lens_list.append(0)

        task_lens = torch.tensor(task_lens_list, dtype=torch.long, device=self.device)
        if task_list:
            task_batch_cat = torch.cat(task_list, dim=0)
        else:
            task_batch_cat = torch.zeros(0, 4, device=self.device)

        # ── Actions → concat + lens (same structure as tasks) ──
        action_list = []
        action_lens_list = []
        for i in range(n):
            a = self.action[i]
            if a is not None:
                action_list.append(a)
                action_lens_list.append(a.shape[0])
            else:
                action_lens_list.append(0)

        action_lens = torch.tensor(action_lens_list, dtype=torch.long, device=self.device)
        if action_list:
            actions_cat = torch.cat(action_list, dim=0)
        else:
            actions_cat = torch.zeros(0, dtype=torch.int64, device=self.device)

        # ── Batch Sizes → concat ──
        bs_list = []
        for i in range(n):
            bs = self.batch_sizes[i]
            if bs is not None:
                bs_list.append(bs)
            else:
                # Fill with ones if missing? Or zeros? Let's use zeros consistent with lengths
                if task_lens_list[i] > 0:
                    bs_list.append(torch.ones(task_lens_list[i], device=self.device))
        
        if bs_list:
            batch_sizes_cat = torch.cat(bs_list, dim=0)
        else:
            batch_sizes_cat = torch.zeros(0, device=self.device)

        # ── Masks → list ──
        masks = [self.mask[i] for i in range(n)]

        return (
            svc,            # 0  (n, svc_dim)
            task_batch_cat, # 1  (total_tasks, task_dim)
            task_lens,      # 2  (n,)
            actions_cat,    # 3  (total_tasks,)
            action_lens,    # 4  (n,)
            prev_mf,        # 5  (n, mf_dim)
            curr_mf,        # 6  (n, mf_dim)
            reward,         # 7  (n, 1)
            next_svc,       # 8  (n, svc_dim)
            done,           # 9  (n, 1)
            log_prob,       # 10 (n, 1)
            value,          # 11 (n, 1)
            masks,          # 12 List[n]
            agent_id,       # 13 (n,)
            batch_sizes_cat, # 14 (total_tasks,)
            svc_idx,        # 15 (n,)
        )

    def clear(self):
        self.ptr = 0
        self.size = 0
        self.task_state  = [None] * self.max_size
        self.batch_sizes = [None] * self.max_size
        self.action      = [None] * self.max_size
        self.mask        = [None] * self.max_size

    def __len__(self):
        return self.size


class MultiAgentRolloutBuffer:
    def __init__(self, num_agents: int, node_type: str,
                 max_size_per_agent: int, service_state_dim: int,
                 action_dim: int, device):
        self.num_agents = num_agents
        self.node_type  = node_type
        self.device     = device

        self.buffers = [
            RolloutBuffer(max_size_per_agent, node_type,
                          service_state_dim, action_dim, device=device)
            for _ in range(num_agents)
        ]
        self.buffer_sizes = torch.zeros(num_agents, dtype=torch.long)
        self.total_size   = 0
        self.total_adds   = 0
        self.log_interval = (5000 * num_agents
                             if node_type == "Terminal_Group"
                             else 500 * num_agents)

    def add_batch(self, service_states, task_states,
                  prev_mfs, curr_mfs, actions, rewards,
                  next_service_states, dones, log_probs,
                  values, agent_ids, service_indices=None, masks=None, task_batch_sizes=None):
        """Add batch of transitions, one per agent."""
        a_ids = agent_ids.view(-1)
        s_ids = service_indices.view(-1) if service_indices is not None else [0] * len(a_ids)
        for i in range(len(a_ids)):
            a_id = int(a_ids[i])
            s_id = int(s_ids[i])
            t_s  = task_states[i] if task_states is not None else None
            m    = masks[i] if masks is not None else None

            self.buffers[a_id].add(
                service_states[i], t_s,
                prev_mfs[i], curr_mfs[i],
                actions[i], rewards[i],
                next_service_states[i], dones[i],
                log_probs[i], values[i],
                a_id, s_id, mask=m,
                task_batch_sizes=task_batch_sizes[i] if task_batch_sizes is not None else None
            )
            self.buffer_sizes[a_id] = self.buffers[a_id].size

        self.total_size = int(self.buffer_sizes.sum().item())
        self.total_adds += len(a_ids)

        if self.total_adds >= self.log_interval:
            self.total_adds = 0
            self.print_reward()
            if self.node_type == "Terminal_Group":
                self.print_stats()

    # ═══════════════════════════════════════════════════════
    # get_all_ready — dùng trong learn()
    # ═══════════════════════════════════════════════════════

    def get_all_ready(self, min_size=1, agent_ids_pool=None):
        """
        Collect ALL data from agents that have >= min_size samples.

        Args:
            min_size:       minimum samples per agent
            agent_ids_pool: (K,) LongTensor — chỉ lấy từ pool này.
                            Nếu None, lấy tất cả agents.

        Returns:
            Tuple of 14 items (same as RolloutBuffer.get_all):
            0  service_states     (D, svc_dim)
            1  task_batch_cat     (total_tasks, task_dim)
            2  task_lens          (D,)
            3  actions_cat        (total_tasks,)
            4  action_lens        (D,)
            5  prev_mfs           (D, mf_dim)
            6  curr_mfs           (D, mf_dim)
            7  rewards            (D, 1)
            8  next_service_states(D, svc_dim)
            9  dones              (D, 1)
            10 log_probs          (D, 1)
            11 values             (D, 1)
            12 masks              List[D]
            13 agent_ids          (D,)
            14 batch_sizes_cat    (total_tasks,)
            15 service_indices    (D,)  
            hoặc None nếu không có agent nào sẵn sàng.
        """
        # Xác định pool
        if agent_ids_pool is not None:
            pool = set(agent_ids_pool.cpu().numpy().tolist())
        else:
            pool = set(range(self.num_agents))

        # Tìm agents sẵn sàng
        ready_agents = sorted([
            a for a in pool
            if a < self.num_agents and self.buffers[a].size >= min_size
        ])

        if not ready_agents:
            return None

        # Thu thập data từ mỗi agent
        all_samples = []
        for a_id in ready_agents:
            s = self.buffers[a_id].get_all()
            if s is not None:
                all_samples.append(s)

        if not all_samples:
            return None

        # Gộp lại
        return self._collate(all_samples)

    def _collate(self, samples):
        """
        Collate list of get_all() outputs into single batch.

        Each sample = tuple of 16 items from RolloutBuffer.get_all()
        """
        result = []

        for field_idx in range(16):
            if field_idx == 1:
                # task_batch_cat: concat
                result.append(torch.cat([s[1] for s in samples], dim=0))

            elif field_idx == 2:
                # task_lens: concat
                result.append(torch.cat([s[2] for s in samples], dim=0))

            elif field_idx == 3:
                # actions_cat: concat
                result.append(torch.cat([s[3] for s in samples], dim=0))

            elif field_idx == 4:
                # action_lens: concat
                result.append(torch.cat([s[4] for s in samples], dim=0))

            elif field_idx == 12:
                # masks: flatten list of lists
                all_masks = []
                for s in samples:
                    all_masks.extend(s[12])
                result.append(all_masks)

            elif field_idx == 14:
                # batch_sizes_cat: concat
                result.append(torch.cat([s[14] for s in samples], dim=0))

            elif field_idx == 15:
                # service_indices: concat
                result.append(torch.cat([s[15] for s in samples], dim=0))

            else:
                # All other fields: standard concat
                result.append(torch.cat([s[field_idx] for s in samples], dim=0))

        return tuple(result)

    # ═══════════════════════════════════════════════════════
    # clear
    # ═══════════════════════════════════════════════════════

    def clear(self, agent_ids=None):
        """
        Clear buffers.
        Nếu agent_ids=None: clear tất cả.
        Nếu agent_ids: chỉ clear các agent đó.
        """
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

    # ═══════════════════════════════════════════════════════
    # Stats / Logging
    # ═══════════════════════════════════════════════════════

    def print_stats(self):
        all_s = [b.service_state[:b.size] for b in self.buffers if b.size > 0]
        if all_s:
            combined = torch.cat(all_s).view(-1)
            sq = torch.quantile(combined, torch.tensor([0.25, 0.5, 0.75]))
            print(f"[Terminal State] Mean: {combined.mean():.3f} | "
                  f"Std: {combined.std():.3f}")
            print(f"   Range: [{combined.min():.3f}, {combined.max():.3f}]")
            print(f"   Q25:{sq[0]:.3f} Q50:{sq[1]:.3f} Q75:{sq[2]:.3f}")
        print("─" * 50)

    def print_reward(self):
        if self.total_size == 0:
            return
        buf = self.buffers[0]
        if buf.size > 0:
            r = buf.reward[:buf.size].view(-1)
            rq = torch.quantile(r, torch.tensor([0.25, 0.5, 0.75]))
            print(f"\n[Buffer Stats] Total: {self.total_size}")
            print(f"[Agent 0 Reward] Mean: {r.mean():.3f} | Std: {r.std():.3f}")
            print(f"   Min: {r.min():.3f} | Max: {r.max():.3f}")
            print(f"   Q25:{rq[0]:.3f} Q50:{rq[1]:.3f} Q75:{rq[2]:.3f}")

    def get_len(self, agent_id=None):
        if agent_id is None:
            return self.total_size
        return self.buffer_sizes[agent_id].item()

    def __len__(self):
        return self.total_size
