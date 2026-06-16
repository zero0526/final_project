import torch
import numpy as np
from matrix_source.agents.ppo import PPOAgent
from matrix_source.agents.group_tasks import SequentialGroupGRU_PPOAgent
from matrix_source.trainers.strategies import AlgorithmStrategy
from matrix_source.trainers.train import log_transform
from matrix_source.utils.math_utils import to_binary
from tqdm import tqdm
import os
from datetime import datetime


# ══════════════════════════════════════════════════════════════
# 0. PLACEHOLDER cho code ngoài scope
# ══════════════════════════════════════════════════════════════
def compute_gae(rewards, next_values, values, dones, agent_ids, gamma, lmbda):
    device = rewards.device
    num_steps = rewards.size(0)
    deltas = rewards + gamma * next_values * (1 - dones) - values
    advantages = torch.zeros_like(deltas)
    masks = (1 - dones) * (gamma * lmbda)
    boundary_mask = torch.ones(num_steps, device=device)
    if num_steps > 1:
        boundary_mask[:-1] = (agent_ids[:-1] == agent_ids[1:]).float()
    combined_mask = masks * boundary_mask
    curr_advantage = 0
    for t in reversed(range(num_steps)):
        curr_advantage = deltas[t] + curr_advantage * (combined_mask[t] if t < num_steps - 1 else 0)
        advantages[t] = curr_advantage
    return advantages


# ══════════════════════════════════════════════════════════════
# 1. STRATEGY TỐI ƯU
# ══════════════════════════════════════════════════════════════

class GroupGRUPPOSCAFFOLDREPStrategy(AlgorithmStrategy):
    def __init__(self):
        super().__init__()
        self.lower_train_num = 0
        self.upper_train_num = 0
        self.lower_mf_prev = None

        # ── Collect thresholds ──
        self.upper_collect_size = 512   # upper cần đủ 512 transitions
        self.lower_collect_size = 5120  # lower cần đủ 5120 transitions (= 10×)

        # ── Train hyper-params ──
        self.upper_cfg = {'batch': 64, 'epochs': 5}
        self.lower_cfg = {'batch': 128, 'epochs': 8}

        self.max_cycles = 20
        self.cycle_num = 1

        self.is_evaluating = False
        self.model_workloads = None

        # [OPT] Cache tĩnh — khởi tạo 1 lần
        self._wl_tensor = None          # Pre-computed model workloads
        self._placement_cache = {}      # Cache service masks
        self._pw2 = None                # Binary decode weights

    def initialize_agents(self, trainer):
        self.model_workloads = trainer.env.metadata["model_workloads"]

        # [OPT-1] Pre-convert model_workloads dict → tensor lookup
        # Tránh Python dict access trong tight loop
        self._precompute_workload_tensor(trainer)

        # ==========================================
        # UPPER AGENT (GIỮ NGUYÊN)
        # ==========================================
        trainer.shared_upper_agent = PPOAgent(
            node_id=-2, node_type="Edge_Group",
            state_dim=trainer.upper_state_dim,
            action_dim=trainer.upper_action_dim,
            u_action_dim=trainer.upper_u_action_dim,
            mf_hidden_sizes=tuple(trainer.config.hyper_neural["MF_HIDDEN_LAYER"]),
            mf_lr=float(trainer.config.hyper_neural['MF_LR']),
            buffer_min_size=self.upper_cfg['min_size'],
            hidden_sizes=trainer.config.hyper_neural['AGENT_HIDDEN_LAYER'],
            lr=float(trainer.config.hyper_neural['UPPER_LR']),
            gamma=trainer.config.hyper_neural['DISCOUNT_FACTOR'],
            lam=trainer.config.hyper_neural.get('LAMBDA', 0.95),
            clip_eps=trainer.config.hyper_neural.get('CLIP_EPS', 0.2),
            k_epochs=self.upper_cfg['epochs'], batch_size=self.upper_cfg['batch'],
            num_instances=trainer.num_edge_agents, increase_rate_zeta=1.001, device=trainer.device
        )

        # ==========================================
        # LOWER AGENT
        # ==========================================
        lower_hidden_dim = trainer.config.hyper_neural['AGENT_HIDDEN_LAYER'][0]
        lower_mf_dim = trainer.num_nodes + trainer.max_models
        critic_global_dim = trainer.num_nodes * 4 + lower_mf_dim
        actor_state_dim = trainer.lower_state_dim + trainer.num_nodes

        trainer.shared_lower_agent = SequentialGroupGRU_PPOAgent(
            agent_id=-1, node_type="Service_Sequential",
            actor_state_dim=actor_state_dim,
            critic_global_dim=critic_global_dim,
            mf_action_dim=lower_mf_dim,
            action_dim=trainer.lower_u_action_dim,
            mf_hidden_sizes=tuple(trainer.config.hyper_neural["MF_HIDDEN_LAYER"]),
            mf_lr=float(trainer.config.hyper_neural['MF_LR']),
            hidden_dim=lower_hidden_dim,
            critic_hidden=(256, 128),
            lr=float(trainer.config.hyper_neural['LOWER_LR']),
            clip_eps=trainer.config.hyper_neural.get('CLIP_EPS', 0.2),
            k_epochs=self.lower_cfg['epochs'],
            entropy_coef=0.01,
            num_instances=trainer.num_edge_agents,
            device=trainer.device
        )

        if self.lower_mf_prev is None:
            self.lower_mf_prev = torch.zeros(
                trainer.num_edge_agents, trainer.num_nodes, lower_mf_dim,
                device=trainer.device
            )

        if self.phase == 'LOWER_ONLY' and self.lower_warmup_steps == 0:
            self.phase = 'UPPER_ONLY'

    # ──────────────────────────────────────────────────
    # [OPT-1] Pre-compute workloads → tensor
    # ──────────────────────────────────────────────────
    def _precompute_workload_tensor(self, trainer):
        """
        Chuyển self.model_workloads (dict of dict) → tensor (S, M_max).
        Tránh Python dict access + tensor creation trong loop.
        """
        S = trainer.num_services
        M = trainer.max_models
        wl = torch.zeros(S, M, device=trainer.device)
        for s in range(S):
            for m, val in enumerate(self.model_workloads[s]):
                wl[s, m] = val
        self._wl_tensor = wl  # (S, M)

    # ──────────────────────────────────────────────────
    # [OPT-2] Cache placement masks
    # ──────────────────────────────────────────────────
    def _get_service_mask(self, trainer, s_idx):
        if s_idx not in self._placement_cache:
            self._placement_cache[s_idx] = \
                trainer.env.engine.placement_matrix[:, s_idx].float()
        return self._placement_cache[s_idx]

    def _invalidate_placement_cache(self):
        """Gọi khi placement_matrix thay đổi (sau step_upper)"""
        self._placement_cache.clear()

    # ──────────────────────────────────────────────────
    # UPPER LEVEL (GIỮ NGUYÊN)
    # ──────────────────────────────────────────────────
    def get_upper_actions(self, trainer, current_upper_state, obs_upper):
        act_matrix = torch.zeros(
            (trainer.num_nodes, trainer.num_services), device=trainer.device
        )
        mf_global = obs_upper.get(
            'mean_fields',
            torch.zeros((trainer.num_nodes, trainer.num_services),
                        device=trainer.device)
        )

        edge_states = current_upper_state[trainer.edge_node_ids]
        edge_mfs = mf_global[trainer.edge_node_ids]
        instance_indices = torch.tensor(
            [trainer.node_to_instance[nid] for nid in trainer.edge_node_ids],
            device=trainer.device
        )
        is_det = self.is_evaluating

        batch_a_ids, log_probs, values = \
            trainer.shared_upper_agent.choose_action_batch(
                edge_states, edge_mfs,
                agent_indices=instance_indices, deterministic=is_det
            )

        if self._pw2 is None or self._pw2.shape[0] != trainer.num_services:
            self._pw2 = 2 ** torch.arange(
                trainer.num_services - 1, -1, -1,
                device=trainer.device
            ).float()

        for i, nid in enumerate(trainer.edge_node_ids):
            act_matrix[nid] = torch.tensor(
                to_binary(batch_a_ids[i], trainer.num_services),
                device=trainer.device
            )
        for nid in trainer.env.static_matrices.get("cloud_ids", []):
            act_matrix[nid] = torch.ones(
                trainer.num_services, device=trainer.device
            )
        return act_matrix, log_probs, values

    def store_upper_transitions(self, trainer, s_all, ns_all, current_res,
                                next_res, acts_matrix, log_probs, values,
                                is_done):
        reward = next_res['reward_global']
        rew_divisor = trainer.config.norm_upper_rw
        norm_rew = log_transform(
            reward / (rew_divisor if rew_divisor != 0 else 1.0)
        )
        avg_mf_loss = 0.0
        edge_states = (s_all[trainer.edge_node_ids]
                       if s_all is not None else None)

        if not self.is_evaluating:
            edge_next_states = ns_all[trainer.edge_node_ids]
            dones = torch.full(
                (trainer.num_edge_agents,),
                1.0 if is_done else 0.0,
                dtype=torch.float32, device=trainer.device
            )
            next_raw_mf = next_res['mean_fields']
            raw_mf = current_res["mean_fields"]

            edge_c_mfs = raw_mf[trainer.edge_node_ids]
            edge_n_mfs = next_raw_mf[trainer.edge_node_ids]
            edge_acts = acts_matrix[trainer.edge_node_ids]
            edge_a_ids = (edge_acts * self._pw2).sum(dim=1).long()
            rewards = torch.full(
                (trainer.num_edge_agents,), norm_rew,
                dtype=torch.float32, device=trainer.device
            )
            instance_indices = torch.tensor(
                [trainer.node_to_instance[nid]
                 for nid in trainer.edge_node_ids],
                device=trainer.device
            )
            avg_mf_loss = trainer.shared_upper_agent.store_transition_train_mf_batch(
                edge_states, edge_c_mfs,
                edge_n_mfs,
                edge_a_ids, rewards, edge_next_states,
                dones, agent_ids=instance_indices,
                log_prob=log_probs, value=values
            )

        agg_state = (edge_states[0]
                     if (edge_states is not None and len(edge_states) > 0)
                     else None)
        trainer.aggregator.add_upper(next_res, mf_loss=avg_mf_loss,
                                     state=agg_state)

        if self.is_evaluating:
            return {
                'reward': next_res['reward_global'],
                'backlog': next_res['obs']['backlog'].sum().item(),
                'energy': next_res['info'].get('energy', 0.0)
            }
        return None
    # ══════════════════════════════════════════════════
    # LOWER LEVEL HELPERS
    # ══════════════════════════════════════════════════
    def _build_simulated_state_batch(self, trainer, task_reqs,
                                     sim_backlog_batch,
                                     sim_capacity_batch,
                                     workload_sent_batch):
        """
        task_reqs đã được normalize bên ngoài trước khi gọi hàm này.
        Chỉ cần normalize sim_backlog và sim_capacity (GFLOPs).
        Layout: [task_reqs(4) | backlog(num_nodes) | capacity(num_nodes) | workload_sent(num_nodes)]
        """
        st = torch.cat([task_reqs, sim_backlog_batch,
                        sim_capacity_batch, workload_sent_batch], dim=-1)
        # backlog(num_nodes) + capacity(num_nodes) + workload_sent(num_nodes) đều là GFLOPs
        if st.shape[1] > 4:
            st[:, 4:] /= trainer.config.norm_gflop
        return st

    def _build_critic_global_state(self, trainer, mask_s, q_final_s,
                                   f_s, workload_sent_s, mean_mf_s):
        return torch.cat([mask_s, q_final_s, f_s,
                          workload_sent_s, mean_mf_s])

    def _compute_node_wise_mfs(self, trainer, t_idx, inst_idx, n_idx, m_idx,
                               src_node_indices):
        B = len(t_idx)
        mf_dim = trainer.num_nodes + trainer.max_models
        device = trainer.device

        two_hot = torch.zeros(B, mf_dim, device=device)
        two_hot[torch.arange(B), n_idx.long()] = 1.0
        two_hot[torch.arange(B), trainer.num_nodes + m_idx.long()] = 1.0

        num_instances = trainer.num_edge_agents
        num_nodes = trainer.num_nodes

        group_sums_en = torch.zeros(num_instances, num_nodes, mf_dim,
                                    device=device)
        group_counts_en = torch.zeros(num_instances, num_nodes,
                                      device=device)

        # inst_idx là instance ID của source node (0..num_edge_agents-1)
        instance_node_flat_idx = inst_idx * num_nodes + src_node_indices
        group_sums_en.view(-1, mf_dim).index_add_(
            0, instance_node_flat_idx.long(), two_hot
        )
        group_counts_en.view(-1).index_add_(
            0, instance_node_flat_idx.long(),
            torch.ones(B, device=device)
        )

        group_sums_e = group_sums_en.sum(dim=1)
        group_counts_e = group_counts_en.sum(dim=1)

        denom = (group_counts_e.unsqueeze(1) - group_counts_en).clamp(min=1)
        node_wise_mfs = (
            (group_sums_e.unsqueeze(1) - group_sums_en)
            / denom.unsqueeze(2)
        )
        node_wise_mfs = torch.where(
            (group_counts_e.unsqueeze(1) > group_counts_en).unsqueeze(2),
            node_wise_mfs,
            torch.zeros_like(node_wise_mfs)
        )
        return node_wise_mfs

    # ══════════════════════════════════════════════════
    # [OPT-3] TRÍCH XUẤT HÀM CHUNG
    # Loại bỏ 100% code trùng giữa train & eval
    # ══════════════════════════════════════════════════
    def _process_lower_slot(self, trainer, obs_lower, t_idx, s_idx,
                            batch_sizes, task_deadlines,
                            tasks_min_accuracy, store_buffer=False):
        """
        Xử lý MỘT time slot cho Lower Agent.
        Phân nhóm THEO EDGE (Source Node).
        """
        device = trainer.device
        agent = trainer.shared_lower_agent
        max_models = trainer.max_models

        if len(t_idx) == 0:
            trainer.env.time_manager.tick()
            return None, None, None

        # ── Xác định instance ID cho từng task dựa trên source node ──
        src_node_indices_all = torch.argmax(
            trainer.env.engine.terminal_to_node_map[t_idx], dim=1
        )
        # Chuyển node_id -> instance_id (0..num_edge_agents-1)
        inst_idx_all = torch.tensor(
            [trainer.node_to_instance[nid.item()] for nid in src_node_indices_all],
            device=device
        )

        # ── Sort tasks theo (edge_instance, deadline) ──
        sort_key = (
            inst_idx_all.long() * 1000000
            + torch.argsort(torch.argsort(task_deadlines))
        )
        sorted_indices = torch.argsort(sort_key)

        t_idx_sorted = t_idx[sorted_indices]
        s_idx_sorted = s_idx[sorted_indices]
        inst_idx_sorted = inst_idx_all[sorted_indices]
        batch_sizes_sorted = batch_sizes[sorted_indices]
        task_reqs_sorted = obs_lower["obs"]['task_reqs'][t_idx_sorted].clone()
        # Normalize task_reqs tại đây (1 lần duy nhất, KHÔNG normalize lại trong _build_simulated_state_batch)
        task_reqs_sorted[:, 0] /= trainer.config.norm_data_size  # data_size
        task_reqs_sorted[:, 2] /= 100.0                          # accuracy (index 1, giống ppo_stategy_v2)
        src_node_sorted = src_node_indices_all[sorted_indices]

        # ── Group by Edge Instance ──
        unique_inst_sorted, counts = torch.unique(
            inst_idx_sorted, return_counts=True
        )
        num_active_inst = len(unique_inst_sorted)
        max_seq_len = counts.max().item()
        offsets = torch.zeros(
            num_active_inst + 1, dtype=torch.long, device=device
        )
        offsets[1:] = torch.cumsum(counts, dim=0)

        # ── Init hidden states, workload, masks ──
        h_dim = agent.actor.gru_cell.hidden_size
        h_gru_batch = torch.zeros(num_active_inst, h_dim, device=device)
        workload_sent_batch = torch.zeros(
            num_active_inst, trainer.num_nodes, device=device
        )

        # Load balancing mask: Agent tại Edge chỉ quan tâm tới nodes nó có thể đẩy tới
        # Tuy nhiên ở đây mask dựa trên placement của service. 
        # Vì 1 instance xử lý nhiều service, chúng ta sẽ xây dựng mask động trong loop nếu cần,
        # hoặc tạm thời dùng mask cho phép tất cả các node được đặt service đó.
        
        # ── Init output arrays ──
        num_tasks = len(t_idx_sorted)
        final_n_idx = torch.zeros(num_tasks, dtype=torch.long, device=device)
        final_m_idx = torch.zeros(num_tasks, dtype=torch.long, device=device)

        sim_backlog = obs_lower["obs"]['backlog'].clone()
        sim_capacity = (
            obs_lower["obs"]['cpu_alloc']
            * trainer.env.engine.placement_matrix
        ).clone()

        if store_buffer:
            agent.memory.start_episode()
            _buf_states = {int(i): [] for i in unique_inst_sorted}
            _buf_hiddens = {int(i): [] for i in unique_inst_sorted}
            _buf_actions = {int(i): [] for i in unique_inst_sorted}
            _buf_logprobs = {int(i): [] for i in unique_inst_sorted}
            _buf_mfs = {int(i): [] for i in unique_inst_sorted}

        # ════════════════════════════════════════════
        # SEQUENTIAL LOOP
        # ════════════════════════════════════════════
        for k in range(max_seq_len):
            active = (counts > k)
            if not active.any():
                break

            batch_idx = offsets[:-1][active] + k
            curr_inst = unique_inst_sorted[active]
            curr_s = s_idx_sorted[batch_idx]
            curr_src = src_node_sorted[batch_idx]

            # Placement mask cho service hiện tại
            mask_node = torch.stack([
                self._get_service_mask(trainer, int(s))
                for s in curr_s
            ])
            mask_action = mask_node.repeat_interleave(max_models, dim=1)

            # Build state
            # q_s/f_s nay lay theo service cua task dang xet
            q_s = torch.stack([sim_backlog[:, s.long()] for s in curr_s])
            f_s = torch.stack([sim_capacity[:, s.long()] * mask_node[i] 
                               for i, s in enumerate(curr_s)])

            state_batch = self._build_simulated_state_batch(
                trainer, task_reqs_sorted[batch_idx],
                q_s, f_s, workload_sent_batch[active]
            )
            # MF lay theo instance (edge) va node nguon
            mf_input = self.lower_mf_prev[curr_inst.long(), curr_src.long()]

            # GRU forward
            a_ids, lp, new_h = agent.choose_action_batch(
                state_batch, mf_input,
                h_gru_batch[active],
                mask_action,
                curr_inst.long()
            )
            h_gru_batch[active] = new_h

            n_k = a_ids // max_models
            m_k = a_ids % max_models

            # Update simulation
            load_q = (task_reqs_sorted[batch_idx, 0]
                      / trainer.config.norm_gflop)
            wl = self._wl_tensor[
                curr_s.long(), m_k
            ] * batch_sizes_sorted[batch_idx] / trainer.config.norm_gflop

            # Update backlog cho dung node n_k va dung service curr_s
            for i, idx in enumerate(batch_idx):
                sim_backlog[n_k[i], curr_s[i].long()] += load_q[i]
            
            workload_sent_batch[active, n_k] += wl

            if store_buffer:
                for i, inst_id in enumerate(curr_inst):
                    ival = int(inst_id.item())
                    _buf_states[ival].append(state_batch[i])
                    _buf_hiddens[ival].append(new_h[i])
                    _buf_actions[ival].append(a_ids[i].item())
                    _buf_logprobs[ival].append(lp[i].item())
                    _buf_mfs[ival].append(mf_input[i])

            final_n_idx[batch_idx] = n_k
            final_m_idx[batch_idx] = m_k

        if store_buffer:
            for ival in _buf_states:
                if len(_buf_states[ival]) == 0: continue
                tr = agent.memory.current_trajectory[ival]
                tr['states'] = _buf_states[ival]
                tr['hiddens'] = _buf_hiddens[ival]
                tr['actions'] = _buf_actions[ival]
                tr['log_probs'] = _buf_logprobs[ival]
                tr['mfs'] = _buf_mfs[ival]

        final_n_idx_env = torch.zeros(len(t_idx), dtype=torch.long, device=device)
        final_m_idx_env = torch.zeros(len(t_idx), dtype=torch.long, device=device)
        final_n_idx_env[sorted_indices] = final_n_idx
        final_m_idx_env[sorted_indices] = final_m_idx

        curr_slot_mfs = self._compute_node_wise_mfs(
            trainer, t_idx_sorted, inst_idx_sorted,
            final_n_idx, final_m_idx, src_node_sorted
        )

        results = trainer.env.step_lower(
            t_idx, s_idx, batch_sizes,
            final_n_idx_env, final_m_idx_env,
            task_deadlines, tasks_min_accuracy
        )

        self.lower_mf_prev = curr_slot_mfs.clone()

        if store_buffer:
            true_reward = results['reward'] - results['obs']['virtual_drift']
            norm_rew = log_transform(
                true_reward / (trainer.config.norm_lower_rw if trainer.config.norm_lower_rw != 0 else 1.0)
            )
            mf_dim = trainer.num_nodes + trainer.max_models
            for inst_id in unique_inst_sorted:
                ival = int(inst_id.item())
                if len(_buf_states.get(ival, [])) == 0: continue
                
                # Global state cho critic (lay dai dien cho edge nay)
                # Vi truoc day gs phu thuoc vao service, nay chung ta dung mean metrics
                # q_final, f_final nay se la mean tren cac service tai edge nay?
                # De don gian, lay mean cua sim_backlog tai cac service ma edge nay tung xu ly
                q_final = sim_backlog.mean(dim=1) 
                f_final = sim_capacity.mean(dim=1) 
                wl_final = workload_sent_batch[list(unique_inst_sorted).index(inst_id)]
                mean_mf = curr_slot_mfs[ival].mean(dim=0)
                
                # Mock mask cho critic (cho phep tat ca vi critic dung global state)
                dummy_mask = torch.ones(trainer.num_nodes, device=device)
                gs = self._build_critic_global_state(
                    trainer, dummy_mask, q_final, f_final, wl_final, mean_mf
                )
                dummy_mf = torch.zeros(mf_dim, device=device)
                agent.memory.end_episode(ival, dummy_mf, 
                                        torch.ones(trainer.num_nodes * max_models, device=device), 
                                        gs, reward=norm_rew)
        return results, final_n_idx_env, final_m_idx_env

    # ══════════════════════════════════════════════════
    # [OPT-3] Hàm xử lý time slot chuẩn bị
    # ══════════════════════════════════════════════════
    def _prepare_and_process_slot(self, trainer, obs_lower, t_idx, s_idx,
                                  batch_sizes, tasks_min_accuracy,
                                  task_deadlines, store_buffer):
        """
        Wrapper gọi _process_lower_slot, xử lý slot rỗng,
        và finalize buffer nếu cần.
        """
        results, _, _ = self._process_lower_slot(
            trainer, obs_lower, t_idx, s_idx, batch_sizes,
            task_deadlines, tasks_min_accuracy,
            store_buffer=store_buffer
        )

        if results is None:
            return None

        if store_buffer:
            trainer.shared_lower_agent.memory.finalize_episode()

        return results

    # ══════════════════════════════════════════════════
    # MAIN TRAINING LOOP
    # ══════════════════════════════════════════════════
    def run_training(self, trainer):
        """
        Collect-then-Train loop (không chia phase):
          - Thu thập song song cho cả upper (512) và lower (5120).
          - Trong suốt quá trình collect KHÔNG train.
          - Khi CẢ HAI đủ ngưỡng: train upper trước → train lower → clear cả hai buffer.
          - Mỗi lần train xong tính là 1 cycle.
        """
        max_slots = trainer.env.time_manager.max_steps
        ep = 0
        pbar = tqdm(total=self.max_cycles, desc="Sequential GRU Progress")

        upper_agent = trainer.shared_upper_agent
        lower_agent = trainer.shared_lower_agent

        while self.cycle_num <= self.max_cycles:
            obs = trainer.env.reset()
            obs_upper = obs['upper']
            obs_lower = obs['lower']
            current_upper_state = self.build_upper_state(trainer, obs_upper)
            self._invalidate_placement_cache()

            for slot in range(max_slots):
                # ── Upper action / placement (frame boundary đầu slot) ──
                if trainer.env.time_manager.is_new_frame():
                    u_acts, u_lp, u_val = self.get_upper_actions(
                        trainer, current_upper_state, obs_upper
                    )
                    trainer.env.step_upper(u_acts)
                    self._invalidate_placement_cache()

                # ── Sinh workload và thu thập lower transitions ──
                t_idx, s_idx, bs, min_acc, deadlines = \
                    trainer.workload_gen.generate_step()

                if len(t_idx) > 0:
                    # Luôn store_buffer (collect lower) — KHÔNG train ngay
                    results = self._prepare_and_process_slot(
                        trainer, obs_lower, t_idx, s_idx, bs,
                        min_acc, deadlines, store_buffer=True
                    )

                    if results is None:
                        continue

                    obs_lower = results
                    trainer.aggregator.add_lower(results, mf_loss=0.0,
                                                 state=None)

                    if slot % 1000 == 0 and slot > 0:
                        ok = sum(trainer.aggregator.episode_success_qos)
                        fail = sum(trainer.aggregator.episode_violate_qos)
                        trainer.aggregator.log(
                            f"  > Slot {slot:4d} | OK: {ok:5.0f} "
                            f"| FAIL: {fail:5.0f}"
                        )
                else:
                    trainer.env.time_manager.tick()

                # ── Upper frame boundary: thu thập upper transition ──
                if trainer.env.time_manager.is_new_frame():
                    res_upper = trainer.env.collect_upper_metrics()
                    next_upper_state = self.build_upper_state(
                        trainer, res_upper
                    )
                    is_ep_done = (slot == max_slots - 1)
                    self.store_upper_transitions(
                        trainer, current_upper_state, next_upper_state,
                        obs_upper, res_upper, u_acts, u_lp, u_val,
                        is_ep_done
                    )
                    current_upper_state = next_upper_state
                    obs_upper = res_upper

                # ── Kiểm tra ngưỡng: CẢ HAI đủ → train rồi clear ──
                upper_ready = len(upper_agent.buffer) >= self.upper_collect_size
                lower_ready = len(lower_agent.memory) >= self.lower_collect_size

                if upper_ready and lower_ready:
                    # 1) Train upper
                    upper_loss = upper_agent.learn(
                        torch.arange(trainer.num_edge_agents,
                                     device=trainer.device)
                    )
                    if upper_loss is not None:
                        self.upper_train_num += 1
                        trainer.aggregator.record_td_losses(
                            upper_losses=upper_loss
                        )

                    # 2) Train lower
                    lower_loss = lower_agent.learn(
                        batch_size=self.lower_cfg['batch'],
                        k_epochs=self.lower_cfg['epochs']
                    )
                    if lower_loss is not None:
                        self.lower_train_num += 1
                        trainer.aggregator.record_td_losses(
                            lower_losses=lower_loss
                        )

                    # 3) Clear cả hai buffer
                    upper_agent.buffer.clear()
                    lower_agent.memory.clear()

                    self.cycle_num += 1
                    pbar.update(1)
                    print(f"\n[Cycle {self.cycle_num - 1}] "
                          f"Train done — upper×{self.upper_train_num} "
                          f"lower×{self.lower_train_num}")

                    if self.cycle_num > self.max_cycles:
                        break

            trainer.aggregator.store_history()
            trainer.aggregator.report_episode(ep)
            trainer.aggregator.reset_episode()
            ep += 1

        pbar.close()
        self.run_evaluation(trainer, num_episodes=5)

    # ══════════════════════════════════════════════════
    # EVALUATION — [OPT-3] Dùng chung _process_lower_slot
    # ══════════════════════════════════════════════════
    def run_evaluation(self, trainer, num_episodes=5):
        print(f"\n--- Evaluation ({num_episodes} Episodes) ---")
        self.is_evaluating = True
        max_slots = trainer.env.time_manager.max_steps

        for ep in range(num_episodes):
            res = trainer.env.reset()
            obs_upper = res['upper']
            obs_lower = res['lower']
            current_upper_state = self.build_upper_state(trainer, obs_upper)
            self._invalidate_placement_cache()
            ep_reward = 0

            for slot in range(max_slots):
                if trainer.env.time_manager.is_new_frame():
                    u_acts, _, _ = self.get_upper_actions(
                        trainer, current_upper_state, obs_upper
                    )
                    trainer.env.step_upper(u_acts)
                    self._invalidate_placement_cache()

                t_idx, s_idx, bs, min_acc, deadlines = \
                    trainer.workload_gen.generate_step()

                if len(t_idx) > 0:
                    # [OPT-3] Dùng đúng hàm chung — 0 code trùng
                    results, _, _ = self._process_lower_slot(
                        trainer, obs_lower, t_idx, s_idx, bs,
                        deadlines, min_acc, store_buffer=False
                    )
                    if results is not None:
                        obs_lower = results
                        ep_reward += results['reward']
                else:
                    trainer.env.time_manager.tick()

                if trainer.env.time_manager.is_new_frame():
                    res_upper = trainer.env.collect_upper_metrics()
                    obs_upper = res_upper
                    current_upper_state = self.build_upper_state(
                        trainer, res_upper
                    )

            print(f"Eval Episode {ep + 1}: Total Reward={ep_reward:.2f}")

        self.is_evaluating = False
