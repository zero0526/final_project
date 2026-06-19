import torch
from matrix_source.agents.ppo import PPOAgent
from matrix_source.agents.refinement import ResidualRoutingAgent

from matrix_source.trainers.strategies import AlgorithmStrategy
from matrix_source.trainers.train import log_transform
from matrix_source.visualize.kstep_monitor import KStepMonitor
from matrix_source.utils.math_utils import to_binary
from tqdm import tqdm


def compute_gae(rewards, next_values, values, dones, agent_ids, gamma, lmbda):
    """ (GIỮ NGUYÊN) Generalized Advantage Estimation (GAE) """
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


class RefinementStrategy(AlgorithmStrategy):
    def __init__(self):
        super().__init__()
        self.lower_train_num = 0
        self.upper_train_num = 0
        self.alt_train_num = 0
        self.alt_next = 'UPPER'
        self.upper_mf_ema = None
        self.lower_mf_prev = None
        self.mf_ema_alpha = 0.7

        self.lower_cfg = {'min_size': 4100, 'batch': 128, 'epochs': 7}
        self.upper_cfg = {'min_size': 410, 'batch': 64, 'epochs': 4}

        self.upper_warmup_steps = 5  
        self.lower_warmup_steps = 15  
        self.max_cycles = 1100
        self.proposal_only_cycles= 350
        self.phase = 'LOWER_ONLY'
        self.cycle_num = 1
        self.current_phase_updates = 0
        self.entropy_decay_rate = 0.99
        self.is_evaluating = False
        self.lower_collect_size= 4100
        self.lower_batch_size= 128
        self.lower_train_epochs= 4
        self.model_workloads= None

    def initialize_agents(self, trainer):
        # ==========================================
        # 1. UPPER AGENT (GIỮ NGUYÊN 100%)
        # ==========================================
        self.model_workloads = trainer.env.metadata["model_workloads"]
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
            num_instances=trainer.num_edge_agents, device=trainer.device
        )

        # ==========================================
        # 2. LOWER AGENT
        # ==========================================
        lower_mf_dim = trainer.num_nodes + trainer.max_models

        trainer.shared_lower_agent = ResidualRoutingAgent(
            model_workload=self.model_workloads.clone(),
            agent_id=-1, node_type="Terminal_Group",
            service_state_dim=trainer.num_nodes * 2,
            proposal_dim=trainer.num_nodes * trainer.max_models,
            mf_dim=trainer.num_nodes + trainer.max_models,
            action_dim=trainer.lower_action_dim,
            u_action_dim=trainer.lower_u_action_dim,
            mf_hidden_sizes=tuple(trainer.config.hyper_neural["MF_HIDDEN_LAYER"]),
            mf_lr=float(trainer.config.hyper_neural['MF_LR']),
            lr=float(trainer.config.hyper_neural['LOWER_LR']),
            buffer_min_size=self.lower_cfg['min_size'],
            batch_size= self.lower_cfg["batch"],
            buffer_size=100_000,
            clip_eps=trainer.config.hyper_neural.get('CLIP_EPS', 0.2),
            k_epochs=self.lower_cfg['epochs'],
            num_instances=trainer.num_edge_agents,
            device=trainer.device,
            min_steps=  self.proposal_only_cycles
        )

        if self.lower_mf_prev is None:
            # lower_mf_prev: (num_services, num_nodes, mf_dim)
            self.lower_mf_prev = torch.zeros(
                trainer.num_services, trainer.num_nodes, lower_mf_dim, device=trainer.device
            )

        if self.phase == 'LOWER_ONLY' and self.lower_warmup_steps == 0:
            self.phase = 'UPPER_ONLY'

        # standalone K-step diagnostics monitor
        self.kstep_monitor = KStepMonitor(
            save_dir   = trainer.config.plot_dir,
            name       = "residual_ppo",
            plot_every = 5,
            window     = 10
        )

    # ==========================================
    # UPPER LEVEL FUNCTIONS (GIỮ NGUYÊN TOÀN BỘ)
    # ==========================================
    def get_upper_actions(self, trainer, current_upper_state, obs_upper):
        act_matrix = torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device)
        mf_global = obs_upper.get('mean_fields', torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device))
        if self.upper_mf_ema is None:
            self.upper_mf_ema = mf_global.clone()
        else:
            self.upper_mf_ema = (1 - self.mf_ema_alpha) * self.upper_mf_ema + self.mf_ema_alpha * mf_global

        edge_states = current_upper_state[trainer.edge_node_ids]
        edge_mfs = self.upper_mf_ema[trainer.edge_node_ids]
        instance_indices = torch.tensor([trainer.node_to_instance[nid] for nid in trainer.edge_node_ids], device=trainer.device)
        is_det = self.is_evaluating

        batch_a_ids, log_probs, values = trainer.shared_upper_agent.choose_action_batch(
            edge_states, edge_mfs, agent_indices=instance_indices, deterministic=is_det
        )
        for i, nid in enumerate(trainer.edge_node_ids):
            act_matrix[nid] = torch.tensor(to_binary(batch_a_ids[i], trainer.num_services), device=trainer.device)
        for nid in trainer.env.static_matrices.get("cloud_ids", []):
            act_matrix[nid] = torch.ones(trainer.num_services, device=trainer.device)
        return act_matrix, log_probs, values

    def store_upper_transitions(self, trainer, s_all, ns_all, current_res, next_res, acts_matrix, log_probs, values, is_done):
        reward = next_res['reward_global']
        rew_divisor = trainer.config.norm_upper_rw
        norm_rew = log_transform(reward / (rew_divisor if rew_divisor != 0 else 1.0))
        avg_mf_loss = 0.0
        edge_states = s_all[trainer.edge_node_ids] if s_all is not None else None

        if not self.is_evaluating:
            edge_next_states = ns_all[trainer.edge_node_ids]
            dones = torch.full((trainer.num_edge_agents,), 1.0 if is_done else 0.0, dtype=torch.float32, device=trainer.device)
            next_raw_mf = next_res['mean_fields']
            edge_c_mfs = current_res['mean_fields'][trainer.edge_node_ids]
            edge_n_mfs = next_raw_mf[trainer.edge_node_ids]

            edge_acts = acts_matrix[trainer.edge_node_ids]
            pw2 = 2 ** torch.arange(trainer.num_services - 1, -1, -1, device=trainer.device).float()
            edge_a_ids = (edge_acts * pw2).sum(dim=1).long()
            rewards = torch.full((trainer.num_edge_agents,), norm_rew, dtype=torch.float32, device=trainer.device)
            instance_indices = torch.tensor([trainer.node_to_instance[nid] for nid in trainer.edge_node_ids], device=trainer.device)
            avg_mf_loss = trainer.shared_upper_agent.store_transition_train_mf_batch(
                edge_states, edge_c_mfs, edge_n_mfs, edge_a_ids, rewards, edge_next_states,
                dones, agent_ids=instance_indices, log_prob=log_probs, value=values
            )
        agg_state = edge_states[0] if (edge_states is not None and len(edge_states) > 0) else None
        trainer.aggregator.add_upper(next_res, mf_loss=avg_mf_loss, state=agg_state)
        if self.is_evaluating:
            return {'reward': next_res['reward_global'], 'backlog': next_res['obs']['backlog'].sum().item(), 'energy': next_res['info'].get('energy', 0.0)}
        return None

    # ==========================================
    # LOWER LEVEL HELPER FUNCTIONS
    # ==========================================
    def _get_service_mask(self, trainer, s_idx):
        """Lấy mask cho 1 service cụ thể across all nodes"""
        # Node mask (M,)
        mask_v = trainer.env.engine.placement_matrix[:, s_idx].float()
        # Action mask (M * K)
        mask_a = mask_v.repeat_interleave(trainer.max_models)
        return mask_v, mask_a

    def _build_service_observation(self, trainer, s_idx, obs_lower):
        """
        Build service_state (2M) for a specific service.
        f = CPU capacity normalized
        Q = task backlog normalized
        """
        # Capacity f (M,)
        f = (obs_lower["obs"]['cpu_alloc'][:, s_idx] * 
             trainer.env.engine.placement_matrix[:, s_idx]).float()
        f = f / trainer.config.norm_gflop

        # Backlog Q (M,)
        q = obs_lower["obs"]['backlog'][:, s_idx].float()
        q = q / trainer.config.norm_data_size

        return torch.cat([f, q])  # (2M)

    def _compute_node_wise_mfs(self, trainer, t_idx, s_idx, n_idx, m_idx, src_node_indices):
        """
        Tính toán Mean Field cho từng service, loại trừ nhóm task đến từ node đang xét.
        """
        B = len(t_idx)
        mf_dim = trainer.num_nodes + trainer.max_models
        device = trainer.device
        
        two_hot = torch.zeros(B, mf_dim, device=device)
        two_hot[torch.arange(B), n_idx.long()] = 1.0
        two_hot[torch.arange(B), trainer.num_nodes + m_idx.long()] = 1.0
        
        num_services = trainer.num_services
        num_nodes = trainer.num_nodes
        
        group_sums_sn = torch.zeros(num_services, num_nodes, mf_dim, device=device)
        group_counts_sn = torch.zeros(num_services, num_nodes, device=device)
        
        service_node_flat_idx = s_idx * num_nodes + src_node_indices
        group_sums_sn.view(-1, mf_dim).index_add_(0, service_node_flat_idx.long(), two_hot)
        group_counts_sn.view(-1).index_add_(0, service_node_flat_idx.long(), torch.ones(B, device=device))
        
        group_sums_s = group_sums_sn.sum(dim=1) 
        group_counts_s = group_counts_sn.sum(dim=1) 
        
        denom = (group_counts_s.unsqueeze(1) - group_counts_sn).clamp(min=1)
        node_wise_mfs = (group_sums_s.unsqueeze(1) - group_sums_sn) / denom.unsqueeze(2)
        
        node_wise_mfs = torch.where(
            (group_counts_s.unsqueeze(1) > group_counts_sn).unsqueeze(2),
            node_wise_mfs,
            torch.zeros_like(node_wise_mfs)
        )
        
        return node_wise_mfs 

    def get_lower_actions(self, trainer, obs_lower, t_idx, s_idx, 
                          tasks_min_accuracy, task_deadlines, batch_sizes):
        """
        Gộp nhóm tasks theo (Edge, Service) và thực hiện batch inference.
        Returns: 
            Dictionary chứa các thông tin cần thiết cho Enviroment Step và Storage.
        """
        device = trainer.device
        # 1. Grouping
        n_src = torch.argmax(trainer.env.engine.terminal_to_node_map[t_idx], dim=1)
        pairs = torch.stack([n_src, s_idx], dim=-1)
        unique_pairs, pair_idx = torch.unique(pairs, dim=0, return_inverse=True)
        B = unique_pairs.shape[0]

        # 2. Prepare Grouped Data
        b_agent_idx = unique_pairs[:, 0]
        b_svc_ids = unique_pairs[:, 1]
        b_svc_states, b_task_states, b_prev_mfs, b_masks, b_task_batch_sizes = [], [], [], [], []

        for i in range(B):
            v, s = int(b_agent_idx[i]), int(b_svc_ids[i])
            mask = (pair_idx == i)
            
            tasks_in_group = obs_lower["obs"]['task_reqs'][t_idx[mask]].clone()
            tasks_in_group[:, 0] /= trainer.config.norm_data_size
            tasks_in_group[:, 2] /= 100.0
            
            b_task_states.append(tasks_in_group)
            b_task_batch_sizes.append(batch_sizes[mask])
            b_svc_states.append(self._build_service_observation(trainer, s, obs_lower))
            b_prev_mfs.append(self.lower_mf_prev[s, v])
            _, mask_a = self._get_service_mask(trainer, s)
            b_masks.append(mask_a)

        b_svc_states = torch.stack(b_svc_states)
        b_prev_mfs = torch.stack(b_prev_mfs)

        # 3. Inference
        is_det = self.is_evaluating
        a_ids_list, lp_list, values, h_node = trainer.shared_lower_agent.choose_action_batch(
            service_states=b_svc_states,
            prev_mfs=b_prev_mfs,
            task_states=b_task_states,
            task_batch_sizes=b_task_batch_sizes,
            service_indices=b_svc_ids,
            masks_batch=b_masks,
            agent_indices=b_agent_idx,
            deterministic=is_det
        )

        return {
            'a_ids_list': a_ids_list,
            'lp_list': lp_list,
            'values': values,
            'h_node': h_node,
            'pair_idx': pair_idx,
            'unique_pairs': unique_pairs,
            'b_agent_idx': b_agent_idx,
            'b_svc_ids': b_svc_ids,
            'b_svc_states': b_svc_states,
            'b_task_states': b_task_states,
            'b_task_batch_sizes': b_task_batch_sizes,
            'b_prev_mfs': b_prev_mfs,
            'b_masks': b_masks,
            'n_src': n_src,
            'B': B
        }

    # ==========================================
    # MAIN TRAINING LOOP
    # ==========================================
    def run_training(self, trainer):
        max_slots = trainer.env.time_manager.max_steps
        ep = 0
        pbar = tqdm(total=self.max_cycles, desc="Residual Isolated Test (20P + 10R)")
        last_phase = trainer.shared_lower_agent.current_phase_name
        phase_transition_count = 0
        while self.cycle_num <= self.max_cycles:
            obs = trainer.env.reset()
            obs_upper = obs['upper']
            obs_lower = obs['lower']
            current_upper_state = self.build_upper_state(trainer, obs_upper)

            training_complete = False  # Cờ dừng khẩn cấp khi đủ 30 cycle

            for slot in range(max_slots):
                if training_complete:
                    break  # Thoát khỏi vòng lặp slot ngay lập tức nếu đủ 30 cycle

                # ── UPPER ACTION (Chạy nhưng KHÔNG HỌC) ──
                if trainer.env.time_manager.is_new_frame():
                    u_acts_matrix, u_log_probs, u_values = self.get_upper_actions(trainer, current_upper_state,
                                                                                  obs_upper)
                    trainer.env.step_upper(u_acts_matrix)

                # ── 1. GENERATE WORKLOAD ──
                t_idx, s_idx, batch_sizes, tasks_min_accuracy, task_deadlines = trainer.workload_gen.generate_step()

                if len(t_idx) > 0:
                    l_res = self.get_lower_actions(
                        trainer, obs_lower, t_idx, s_idx, 
                        tasks_min_accuracy, task_deadlines, batch_sizes
                    )
                    
                    # Record proposal logits magnitude
                    if hasattr(trainer.shared_lower_agent, 'last_prop_logits_mean'):
                        trainer.aggregator.record_prop_logits(trainer.shared_lower_agent.last_prop_logits_mean)

                    # ── 5. ENVIRONMENT STEP ──
                    final_n_idxSize = len(t_idx)
                    final_n_idx = torch.zeros(final_n_idxSize, dtype=torch.long, device=trainer.device)
                    final_m_idx = torch.zeros(final_n_idxSize, dtype=torch.long, device=trainer.device)

                    for i in range(l_res['B']):
                        a_ids = l_res['a_ids_list'][i]
                        final_n_idx[l_res['pair_idx'] == i] = a_ids // trainer.max_models
                        final_m_idx[l_res['pair_idx'] == i] = a_ids % trainer.max_models

                    results = trainer.env.step_lower(
                        t_idx, s_idx, batch_sizes, final_n_idx, final_m_idx,
                        task_deadlines, tasks_min_accuracy
                    )

                    # ── 6. REWARDS & STORAGE ──
                    dr_penalty = results['obs'].get('virtual_drift', 0.0)
                    true_reward = results['reward'] - dr_penalty
                    norm_rew = log_transform(true_reward / (trainer.config.norm_lower_rw or 1.0))

                    curr_slot_mfs = self._compute_node_wise_mfs(
                        trainer, t_idx, s_idx, final_n_idx, final_m_idx, l_res['n_src']
                    )

                    b_next_svc_states = torch.stack([
                        self._build_service_observation(trainer, int(l_res['b_svc_ids'][i]), results)
                        for i in range(l_res['B'])
                    ])
                    b_curr_mfs = torch.stack([
                        curr_slot_mfs[int(l_res['b_svc_ids'][i]), int(l_res['b_agent_idx'][i])]
                        for i in range(l_res['B'])
                    ])
                    b_rewards = torch.full((l_res['B'], 1), norm_rew, device=trainer.device)
                    b_dones = torch.zeros((l_res['B'], 1), device=trainer.device)

                    trainer.shared_lower_agent.store_transition_train_mf_batch(
                        service_states=l_res['b_svc_states'],
                        task_states=l_res['b_task_states'],
                        prev_mfs=l_res['b_prev_mfs'],
                        curr_mfs=b_curr_mfs,
                        actions=l_res['a_ids_list'],
                        rewards=b_rewards,
                        next_service_states=b_next_svc_states,
                        dones=b_dones,
                        agent_ids=l_res['b_agent_idx'],
                        log_probs=l_res['lp_list'],
                        values=l_res['values'],
                        service_indices=l_res['b_svc_ids'],
                        masks=l_res['b_masks'],
                        task_batch_sizes=l_res['b_task_batch_sizes']
                    )

                    # ══════════════════════════════════════════════════════
                    # ĐIỂM QUAN TRỌNG: KIỂM TRA ĐỦ DATA THÌ MỚI TÍNH LÀ 1 CYCLE
                    # ══════════════════════════════════════════════════════
                    # is_lower_enough= all((b.size >= self.lower_cfg["min_size"]) for  b in trainer.shared_lower_agent.memory.buffers)
                    # is_upper_enough= all(b.size >= self.upper_cfg["min_size"] for b in trainer.shared_upper_agent.memory.buffers )

                    m_len = trainer.shared_lower_agent.memory.total_size
                    if m_len >= self.lower_collect_size:
                        loss = trainer.shared_lower_agent.learn(step=self.cycle_num)
                        if loss is not None:
                            current_phase = trainer.shared_lower_agent.current_phase_name
                            if current_phase != last_phase:
                                phase_transition_count += 1
                                print(f"\n{'=' * 60}")
                                print(f"PHASE TRANSITION #{phase_transition_count}: "
                                      f"{last_phase} → {current_phase}")
                                print(f"{'=' * 60}\n")
                                last_phase = current_phase

                            print(f"[{current_phase}] Step {self.cycle_num} | Loss: {loss:.5f}")
                            trainer.aggregator.record_td_losses(lower_losses=loss)
                            self.kstep_monitor.record(trainer.shared_lower_agent)

                            # ══════════════════════════════════════════
                            # CHÍNH TẠI ĐÂY MỚI LÀ KẾT THÚC 1 CYCLE!
                            # ══════════════════════════════════════════
                            pbar.update(1)
                            self.cycle_num += 1

                            # Nếu đã đủ 30 cycle thì đánh cờ dừng
                            if self.cycle_num > self.max_cycles:
                                training_complete = True

                    self.lower_mf_prev = curr_slot_mfs.detach()
                    obs_lower = results
                    trainer.aggregator.add_lower(results, mf_loss=0.0, state=None)
                else:
                    trainer.env.time_manager.tick()

                # ── UPPER METRICS COLLECT (KHÔNG HỌC, KHÔNG ĐẾM CYCLE) ──
                if trainer.env.time_manager.is_new_frame():
                    res_upper = trainer.env.collect_upper_metrics()
                    next_upper_state = self.build_upper_state(trainer, res_upper)
                    is_ep_done = (slot == max_slots - 1)

                    self.store_upper_transitions(trainer, current_upper_state, next_upper_state, obs_upper, res_upper,
                                                 acts_matrix=u_acts_matrix, log_probs=u_log_probs, values= u_values, is_done= is_ep_done)
                    current_upper_state = next_upper_state
                    obs_upper = res_upper

            trainer.aggregator.store_history()
            trainer.aggregator.report_episode(ep)
            trainer.aggregator.reset_episode()
            ep += 1

        pbar.close()
        print("\n" + "=" * 60)
        print(f"ISOLATED TEST FINISHED! Total Lower Updates: {self.cycle_num}")
        print("=" * 60)
        self.run_evaluation(trainer, num_episodes=5)

    def run_evaluation(self, trainer, num_episodes=5):
        print(f"\n--- Starting Post-Training Evaluation ({num_episodes} Episodes) ---")
        self.is_evaluating = True
        max_slots = trainer.env.time_manager.max_steps
        
        for ep in range(num_episodes):
            res = trainer.env.reset()
            obs_upper, obs_lower = res['upper'], res['lower']
            current_upper_state = self.build_upper_state(trainer, obs_upper)
            ep_reward = 0

            for slot in range(max_slots):
                if trainer.env.time_manager.is_new_frame():
                    u_acts_matrix, _, _ = self.get_upper_actions(trainer, current_upper_state, obs_upper)
                    trainer.env.step_upper(u_acts_matrix)

                t_idx, s_idx, batch_sizes, tasks_min_accuracy, task_deadlines = trainer.workload_gen.generate_step()
                if len(t_idx) > 0:
                    l_res = self.get_lower_actions(
                        trainer, obs_lower, t_idx, s_idx, 
                        tasks_min_accuracy, task_deadlines, batch_sizes
                    )

                    final_n_idxSize = len(t_idx)
                    final_n_idx = torch.zeros(final_n_idxSize, dtype=torch.long, device=trainer.device)
                    final_m_idx = torch.zeros(final_n_idxSize, dtype=torch.long, device=trainer.device)
                    for i in range(l_res['B']):
                        a_ids = l_res['a_ids_list'][i]
                        final_n_idx[l_res['pair_idx'] == i] = a_ids // trainer.max_models
                        final_m_idx[l_res['pair_idx'] == i] = a_ids % trainer.max_models

                    results = trainer.env.step_lower(
                        t_idx, s_idx, batch_sizes, final_n_idx, final_m_idx, 
                        task_deadlines, tasks_min_accuracy
                    )
                    ep_reward += results['reward']
                    
                    curr_slot_mfs = self._compute_node_wise_mfs(
                        trainer, t_idx, s_idx, final_n_idx, final_m_idx, l_res['n_src']
                    )
                    self.lower_mf_prev = curr_slot_mfs.detach()
                    obs_lower = results
                else:
                    trainer.env.time_manager.tick()

                if trainer.env.time_manager.is_new_frame():
                    res_upper = trainer.env.collect_upper_metrics()
                    obs_upper = res_upper
                    next_upper_state = self.build_upper_state(trainer, res_upper)
                    current_upper_state = next_upper_state

            print(f"Eval Episode {ep + 1}: Total Reward={ep_reward:.2f}")

        self.is_evaluating = False