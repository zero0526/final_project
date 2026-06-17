import torch
from matrix_source.agents.ppo import PPOAgent
from matrix_source.agents.coma_residual import COMAResidualRoutingAgent

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


class COMAResidualStrategy(AlgorithmStrategy):
    def __init__(self):
        super().__init__()
        self.lower_train_num = 0
        self.upper_train_num = 0
        self.alt_train_num = 0
        self.alt_next = 'UPPER'
        self.upper_mf_ema = None
        self.lower_mf_prev = None
        self.mf_ema_alpha = 0.7

        self.lower_cfg = {'min_size': 4096, 'batch': 128, 'epochs': 7}
        self.upper_cfg = {'min_size': 512, 'batch': 64, 'epochs': 5}

        self.upper_warmup_steps = 3
        self.lower_warmup_steps = 8
        self.max_cycles = 1600
        self.proposal_only_cycles = 400
        self.phase = 'LOWER_ONLY'
        self.cycle_num = 1
        self.current_phase_updates = 0
        self.entropy_decay_rate = 0.99
        self.is_evaluating = False
        self.lower_collect_size = 4096
        self.lower_batch_size = 128
        self.model_workloads = None
        self.node_to_instance_tensor = None
        self.service_input_size = None

    def initialize_agents(self, trainer):
        # ==========================================
        # 1. UPPER AGENT (GIỮ NGUYÊN 100%)
        # ==========================================
        # num_service x num_model
        self.model_workloads = torch.tensor(trainer.env.metadata["model_workloads"], 
                                            device=trainer.device, dtype=torch.float32)
        # num_service x 1
        self.service_input_size = torch.tensor(trainer.env.metadata["service_input_size"], 
                                              device=trainer.device, dtype=torch.float32)

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
        # 2. LOWER AGENT
        # ==========================================
        lower_mf_dim = trainer.num_nodes + trainer.max_models

        trainer.shared_lower_agent = COMAResidualRoutingAgent(
            agent_id=-1, node_type="Terminal_Group",
            service_state_dim=trainer.num_nodes * 2,
            mf_dim=trainer.num_nodes + trainer.max_models,
            action_dim=trainer.lower_action_dim,
            u_action_dim=trainer.lower_u_action_dim,
            max_models= trainer.max_models,
            mf_hidden_sizes=tuple(trainer.config.hyper_neural["MF_HIDDEN_LAYER"]),
            mf_lr=float(trainer.config.hyper_neural['MF_LR']),
            lr=float(trainer.config.hyper_neural['LOWER_LR']),
            alpha=1.0,  # <--- Set target alpha là 1.0
            proposal_only_cycles=self.proposal_only_cycles, # <--- Truyền xuống để Agent tự tính warmup
            alpha_warmup_cycles=200, # <--- Mất 200 cycles để alpha tăng từ 0.1 -> 1.0
            buffer_min_size=self.lower_cfg['min_size'],
            buffer_size=100_000,
            clip_eps=trainer.config.hyper_neural.get('CLIP_EPS', 0.2),
            k_epochs=self.lower_cfg['epochs'],
            num_instances=trainer.num_edge_node ,
            device=trainer.device
        )

        if self.lower_mf_prev is None:
            # lower_mf_prev: (num_services, num_nodes, mf_dim)
            self.lower_mf_prev = torch.zeros(
                trainer.num_services, trainer.num_nodes, lower_mf_dim, device=trainer.device
            )
        
        # Initialize mapping tensors
        self._prepare_instance_mapping(trainer)

    def _prepare_instance_mapping(self, trainer):
        """Creates tensor mappings for fast physical-to-agent ID conversion."""
        # Lower Level: maps edge_ids (sources) to [0, num_edge_node - 1]
        self.low_instance_mapping = torch.full((trainer.num_nodes,), -1, dtype=torch.long, device=trainer.device)
        for nid, instance_idx in trainer.node_to_instance.items():
            self.low_instance_mapping[nid] = instance_idx

        # Upper Level: maps edge_node_ids to [0, num_edge_agents - 1]
        self.up_instance_mapping = torch.full((trainer.num_nodes,), -1, dtype=torch.long, device=trainer.device)
        for nid, instance_idx in trainer.comp_to_up_agent.items():
            self.up_instance_mapping[nid] = instance_idx

        if self.phase == 'LOWER_ONLY' and self.lower_warmup_steps == 0:
            self.phase = 'UPPER_ONLY'

        # standalone K-step diagnostics monitor
        self.kstep_monitor = KStepMonitor(
            save_dir=trainer.config.plot_dir,
            name="residual_ppo",
            plot_every=5,
            window=10
        )

    # ==========================================
    # UPPER LEVEL FUNCTIONS (GIỮ NGUYÊN TOÀN BỘ)
    # ==========================================
    def get_upper_actions(self, trainer, current_upper_state, obs_upper):
        act_matrix = torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device)
        mf_global = obs_upper.get('mean_fields',
                                  torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device))
        if self.upper_mf_ema is None:
            self.upper_mf_ema = mf_global.clone()
        else:
            self.upper_mf_ema = (1 - self.mf_ema_alpha) * self.upper_mf_ema + self.mf_ema_alpha * mf_global

        edge_states = current_upper_state[trainer.edge_node_ids]
        edge_mfs = self.upper_mf_ema[trainer.edge_node_ids]
        instance_indices = self.up_instance_mapping[trainer.edge_node_ids]
        is_det = self.is_evaluating

        batch_a_ids, log_probs, values = trainer.shared_upper_agent.choose_action_batch(
            edge_states, edge_mfs, agent_indices=instance_indices, deterministic=is_det
        )
        for i, nid in enumerate(trainer.edge_node_ids):
            act_matrix[nid] = torch.tensor(to_binary(batch_a_ids[i], trainer.num_services), device=trainer.device)
        for nid in trainer.env.static_matrices.get("cloud_ids", []):
            act_matrix[nid] = torch.ones(trainer.num_services, device=trainer.device)
        return act_matrix, log_probs, values

    def store_upper_transitions(self, trainer, s_all, ns_all, current_res, next_res, acts_matrix, log_probs, values,
                                is_done):
        """
        Lưu transition cho Upper Agent (PPO).
        Cập nhật: Đồng bộ interface và truyền thêm masks nếu cần.
        """
        reward = next_res['reward_global']
        rew_divisor = trainer.config.norm_upper_rw
        norm_rew = log_transform(reward / (rew_divisor if rew_divisor != 0 else 1.0))
        avg_mf_loss = 0.0
        is_frozen = (self.phase == 'LOWER_ONLY')
        edge_states = s_all[trainer.edge_node_ids] if s_all is not None else None

        if not is_frozen and not self.is_evaluating:
            edge_next_states = ns_all[trainer.edge_node_ids]
            dones = torch.full((trainer.num_edge_agents,), 1.0 if is_done else 0.0, dtype=torch.float32,
                               device=trainer.device)
            next_raw_mf = next_res['mean_fields']

            edge_c_mfs = current_res['mean_fields'][trainer.edge_node_ids]
            edge_n_mfs = next_raw_mf[trainer.edge_node_ids]
            edge_acts = acts_matrix[trainer.edge_node_ids]

            # Chuyển đổi action matrix thành action index (binary to decimal)
            pw2 = 2 ** torch.arange(trainer.num_services - 1, -1, -1, device=trainer.device).float()
            edge_a_ids = (edge_acts * pw2).sum(dim=1).long()

            rewards = torch.full((trainer.num_edge_agents,), norm_rew, dtype=torch.float32, device=trainer.device)
            instance_indices = self.up_instance_mapping[trainer.edge_node_ids]

            # Gọi store_transition của Upper Agent
            # Lưu ý: PPOAgent có thể không cần proposal_logits/h_nodes, nhưng ta truyền masks nếu có
            avg_mf_loss = trainer.shared_upper_agent.store_transition_train_mf_batch(
                service_states=edge_states,
                prev_mfs=edge_c_mfs,
                curr_mfs=edge_n_mfs,
                actions=edge_a_ids,
                rewards=rewards,
                next_service_states=edge_next_states,
                dones=dones,
                agent_ids=instance_indices,
                log_probs=log_probs,
                values=values,
                masks=None  # Upper Agent thường không có mask phức tạp như Lower Agent
            )

        agg_state = edge_states[0] if (edge_states is not None and len(edge_states) > 0) else None
        trainer.aggregator.add_upper(next_res, mf_loss=avg_mf_loss, state=agg_state)

        if self.is_evaluating:
            return {
                'reward': next_res['reward_global'],
                'backlog': next_res['obs']['backlog'].sum().item(),
                'energy': next_res['info'].get('energy', 0.0)
            }
        return None

    # ==========================================
    # LOWER LEVEL TRANSITION HELPERS
    # ==========================================
    def _get_service_mask(self, trainer, s_idx):
        """Lấy mask cho 1 service cụ thể across all nodes"""
        mask_v = trainer.env.engine.placement_matrix[:, s_idx].float()
        mask_a = mask_v.repeat_interleave(trainer.max_models)
        return mask_v, mask_a

    def _build_service_observation(self, trainer, s_idx, obs_lower):
        """Build service_state (2M) for a specific service."""
        f = (obs_lower["obs"]['cpu_alloc'][:, s_idx] * trainer.env.engine.placement_matrix[:, s_idx]).float()
        f = f / trainer.config.norm_gflop
        q = obs_lower["obs"]['backlog'][:, s_idx].float()
        q = q / trainer.config.norm_data_size
        return torch.cat([f, q])

    def _compute_node_wise_mfs(self, trainer, t_idx, s_idx, n_idx, m_idx, src_node_indices):
        """Computes Leave-One-Out Mean Field for each service."""
        B = len(t_idx)
        if B == 0: return torch.zeros(trainer.num_services, trainer.num_nodes, trainer.num_nodes + trainer.max_models, device=trainer.device)

        mf_dim, device = trainer.num_nodes + trainer.max_models, trainer.device
        two_hot = torch.zeros(B, mf_dim, device=device)
        two_hot[torch.arange(B), n_idx.long()] = 1.0
        two_hot[torch.arange(B), trainer.num_nodes + m_idx.long()] = 1.0

        group_sums_sn = torch.zeros(trainer.num_services, trainer.num_nodes, mf_dim, device=device)
        group_counts_sn = torch.zeros(trainer.num_services, trainer.num_nodes, device=device)
        service_node_flat_idx = (s_idx * trainer.num_nodes + src_node_indices).long()
        group_sums_sn.view(-1, mf_dim).index_add_(0, service_node_flat_idx, two_hot)
        group_counts_sn.view(-1).index_add_(0, service_node_flat_idx, torch.ones(B, device=device))

        group_sums_s, group_counts_s = group_sums_sn.sum(dim=1), group_counts_sn.sum(dim=1)
        denom = (group_counts_s.unsqueeze(1) - group_counts_sn).clamp(min=1)
        node_wise_mfs = (group_sums_s.unsqueeze(1) - group_sums_sn) / denom.unsqueeze(2)
        valid_mask = (group_counts_s.unsqueeze(1) > group_counts_sn).unsqueeze(2)
        # num_service x num_node x (num_node+num_model)
        return torch.where(valid_mask, node_wise_mfs, torch.zeros_like(node_wise_mfs))

    def _prepare_group_data(self, trainer, t_idx, s_idx, obs_lower):
        """Groups tasks by (Edge, Service) and prepares input tensors."""
        n_src = torch.argmax(trainer.env.engine.terminal_to_node_map[t_idx], dim=1)
        pairs = torch.stack([n_src, s_idx], dim=-1) # Group by physical node
        unique_pairs, pair_idx = torch.unique(pairs, dim=0, return_inverse=True)
        B = unique_pairs.shape[0]

        b_node_idx = unique_pairs[:, 0]
        # Map physical node IDs to Lower Agent instance indices [0, num_edge_node-1]
        b_agent_idx = self.low_instance_mapping[b_node_idx]
        
        b_svc_ids = unique_pairs[:, 1]
        b_svc_states, b_task_states, b_prev_mfs, b_masks = [], [], [], []

        for i in range(B):
            v_node, v_agent, s = int(b_node_idx[i]), int(b_agent_idx[i]), int(b_svc_ids[i])
            tasks_in_group = obs_lower["obs"]['task_reqs'][t_idx[pair_idx == i]].clone()
            tasks_in_group[:, 0] /= trainer.config.norm_data_size
            tasks_in_group[:, 2] /= 100.0

            b_task_states.append(tasks_in_group)
            b_svc_states.append(self._build_service_observation(trainer, s, obs_lower))
            b_prev_mfs.append(self.lower_mf_prev[s, v_node])
            _, mask_a = self._get_service_mask(trainer, s)
            b_masks.append(mask_a)

        return {
            'B': B, 'n_src': n_src, 'pair_idx': pair_idx,
            'agent_idx': b_agent_idx, 'svc_ids': b_svc_ids,
            'svc_states': torch.stack(b_svc_states),
            'task_states': b_task_states,
            'prev_mfs': torch.stack(b_prev_mfs),
            'masks': b_masks
        }

    def _calculate_group_metrics(self, trainer, task_states):
        """Computes statistical metrics for task groups."""
        b_omega, b_ds, b_dl, b_bs = [], [], [], []
        for ts in task_states:
            n = ts.shape[0]
            if n > 0:
                b_omega.append(ts[0, 0].view(1))
                m_ds, s_ds = ts[:, 1].mean(), ts[:, 1].std(unbiased=False) if n > 1 else torch.tensor(0.0, device=trainer.device)
                q_ds = torch.quantile(ts[:, 1], torch.tensor([0.25, 0.5, 0.75], device=trainer.device))
                b_ds.append(torch.cat([m_ds.view(1), s_ds.view(1), q_ds]))
                m_dl, s_dl = ts[:, 2].mean(), ts[:, 2].std(unbiased=False) if n > 1 else torch.tensor(0.0, device=trainer.device)
                q_dl = torch.quantile(ts[:, 2], torch.tensor([0.25, 0.5, 0.75], device=trainer.device))
                b_dl.append(torch.cat([m_dl.view(1), s_dl.view(1), q_dl]))
                b_bs.append(torch.tensor([float(n)], device=trainer.device))
            else:
                b_omega.append(torch.zeros(1, device=trainer.device))
                b_ds.append(torch.zeros(5, device=trainer.device))
                b_dl.append(torch.zeros(5, device=trainer.device))
                b_bs.append(torch.zeros(1, device=trainer.device))

        return {
            'omega': torch.stack(b_omega),
            'ds_metrics': torch.stack(b_ds),
            'deadline_metrics': torch.stack(b_dl),
            'batch_size': torch.stack(b_bs).view(-1, 1),
            'workload': torch.zeros(len(task_states), trainer.shared_lower_agent.M, device=trainer.device)
        }

    def _calculate_post_inference_workload(self, trainer, B, prop_logits, task_states, a_ids_list, svc_ids):
        """Calculates expected workload after inference, scaling by service intensity."""
        total_tasks = sum(len(a) for a in a_ids_list)
        if total_tasks == 0: 
            return torch.zeros(B, trainer.shared_lower_agent.M, device=trainer.device)
        
        # 1. Probabilities of mapping each task to each node (M)
        all_p_logits = torch.cat(prop_logits, dim=0)
        all_probs = torch.softmax(all_p_logits, dim=-1)
        # Sum probabilities over model instances for each node
        all_probs_M = all_probs.view(total_tasks, trainer.shared_lower_agent.M, trainer.shared_lower_agent.max_models).sum(dim=2)
        
        # 2. Respective intensities for each task's service
        # svc_ids are for groups, expand to all tasks
        t_lens = [len(a) for a in a_ids_list]
        t_lens_tensor = torch.tensor(t_lens, device=trainer.device)
        all_svc_ids = torch.repeat_interleave(svc_ids, t_lens_tensor)
        
        # Scale: Intensity[service] * DataSize[task]
        # We use service_input_size as a scaling factor for computing intensity
        intensities = self.service_input_size[all_svc_ids].to(trainer.device)
        all_tasks_cat = torch.cat(task_states, dim=0)
        all_ds = all_tasks_cat[:, 0].view(-1, 1) # Normalized data size
        
        expected_task_load = all_probs_M * (all_ds * intensities)
        
        # 3. Aggregate into groups (B agents)
        b_idx_exp = torch.repeat_interleave(torch.arange(B, device=trainer.device), t_lens_tensor)
        wl_group = torch.zeros(B, trainer.shared_lower_agent.M, device=trainer.device)
        wl_group.scatter_add_(0, b_idx_exp.view(-1, 1).expand(-1, trainer.shared_lower_agent.M), expected_task_load)
        
        return wl_group

    def _handle_lower_storage_and_learn(self, trainer, storage_data, phrase, pbar):
        """Encapsulates buffer storage and learning process."""
        trainer.shared_lower_agent.store_transition_train_mf_batch(**storage_data)
        
        if trainer.shared_lower_agent.memory.total_size >= self.lower_collect_size:
            loss_dict = trainer.shared_lower_agent.learn(phrase=phrase, step=self.cycle_num)
            if loss_dict is not None:
                self.lower_train_num += 1
                trainer.aggregator.record_td_losses(lower_losses=loss_dict)
                self.kstep_monitor.record(trainer.shared_lower_agent)
                self._print_diagnostics(phrase, loss_dict)
                
                pbar.update(1)
                self.cycle_num += 1
                return True # Cycle completed
        return False

    def _print_diagnostics(self, phrase, loss_dict):
        """Prints training status periodically."""
        if self.cycle_num % 20 == 0:
            print(f"\n{'=' * 20} DIAGNOSTICS [Cycle {self.cycle_num:4d} | Phase: {phrase}] {'=' * 20}")
            print(f"  Losses  -> P: {loss_dict['p_loss']:.4f} | R: {loss_dict['r_loss']:.4f} | V: {loss_dict['v_loss']:.4f}")
            print(f"  Refine  -> Delta Norm: {loss_dict.get('delta_norm', 0):.4f} | Flip Rate: {loss_dict.get('flip_rate', 0) * 100:5.2f}% | KL Div: {loss_dict.get('kl_div', 0):.4f}")
            print(f"  Credit  -> Hybrid Adv: {loss_dict.get('hybrid_adv', 0):.4f} | Refine Grad: {loss_dict.get('refine_grad', 0):.5f}")
            print('=' * 70)

    def run_training(self, trainer):
        max_slots = trainer.env.time_manager.max_steps
        pbar = tqdm(total=self.max_cycles, desc="Training")

        while self.cycle_num <= self.max_cycles:
            obs = trainer.env.reset()
            obs_upper, obs_lower = obs['upper'], obs['lower']
            current_upper_state = self.build_upper_state(trainer, obs_upper)
            phrase = "Proposal_Only" if self.cycle_num <= self.proposal_only_cycles else "Proposal_Free"
            stop_ep = False

            for slot in range(max_slots):
                if stop_ep: break
                
                # 1. Upper Decision
                if trainer.env.time_manager.is_new_frame():
                    u_acts, u_log_probs, u_values = self.get_upper_actions(trainer, current_upper_state, obs_upper)
                    trainer.env.step_upper(u_acts)

                # 2. Lower Rollout
                t_idx, s_idx, b_sz, t_acc, t_dl = trainer.workload_gen.generate_step()
                if len(t_idx) > 0:
                    # Preparation & Metrics
                    gd = self._prepare_group_data(trainer, t_idx, s_idx, obs_lower)
                    metrics = self._calculate_group_metrics(trainer, gd['task_states'])

                    # Inference
                    a_ids_list, lp_list, values, h_nodes, prop_logits = trainer.shared_lower_agent.choose_action_batch(
                        gd['svc_states'], gd['prev_mfs'], gd['task_states'], gd['masks'], gd['agent_idx'], phrase=phrase, metrics=metrics
                    )
                    metrics['workload'] = self._calculate_post_inference_workload(
                        trainer, gd['B'], prop_logits, gd['task_states'], a_ids_list, gd['svc_ids']
                    )

                    # Env Step
                    f_n, f_m = torch.zeros_like(t_idx), torch.zeros_like(t_idx)
                    for i in range(gd['B']):
                        f_n[gd['pair_idx'] == i] = a_ids_list[i] // trainer.max_models
                        f_m[gd['pair_idx'] == i] = a_ids_list[i] % trainer.max_models
                    
                    results = trainer.env.step_lower(t_idx, s_idx, b_sz, f_n, f_m, t_dl, t_acc)
                    rew = log_transform((results['reward'] - results['obs'].get('virtual_drift', 0)) / (trainer.config.norm_lower_rw or 1.0))
                    
                    # Storage & Storage Learn
                    curr_mfs = self._compute_node_wise_mfs(trainer, t_idx, s_idx, f_n, f_m, gd['n_src'])
                    next_svc = torch.stack([self._build_service_observation(trainer, int(gd['svc_ids'][i]), results) for i in range(gd['B'])])
                    curr_mfs_b = torch.stack([curr_mfs[int(gd['svc_ids'][i]), int(gd['agent_idx'][i])] for i in range(gd['B'])])
                    
                    storage_data = {
                        'service_states': gd['svc_states'], 'task_states': gd['task_states'], 'prev_mfs': gd['prev_mfs'], 
                        'curr_mfs': curr_mfs_b, 'actions': a_ids_list, 'rewards': torch.full((gd['B'], 1), rew, device=trainer.device),
                        'next_service_states': next_svc, 'dones': torch.zeros((gd['B'], 1), device=trainer.device),
                        'agent_ids': gd['agent_idx'], 'log_probs': lp_list, 'values': values, 'masks': gd['masks'],
                        'proposal_logits': prop_logits, 'h_nodes': h_nodes, 'workloads': metrics['workload'],
                        'ds_metrics': metrics['ds_metrics'], 'deadline_metrics': metrics['deadline_metrics'],
                        'omegas': metrics['omega'], 'batch_sizes': metrics['batch_size']
                    }
                    
                    if self._handle_lower_storage_and_learn(trainer, storage_data, phrase, pbar):
                        phrase = "Proposal_Only" if self.cycle_num <= self.proposal_only_cycles else "Proposal_Free"
                        if self.cycle_num > self.max_cycles: stop_ep = True
                    
                    self.lower_mf_prev, obs_lower = curr_mfs.detach(), results
                    trainer.aggregator.add_lower(results, mf_loss=0.0)
                else: trainer.env.time_manager.tick()

                # 3. Upper Transition
                if trainer.env.time_manager.is_new_frame():
                    res_u = trainer.env.collect_upper_metrics()
                    self.store_upper_transitions(trainer, current_upper_state, self.build_upper_state(trainer, res_u), obs_upper, res_u, u_acts, u_log_probs, u_values, slot == max_slots -1)
                    current_upper_state, obs_upper = self.build_upper_state(trainer, res_u), res_u

            trainer.aggregator.store_history()
            trainer.aggregator.report_episode(0) # Simplified ep count
            trainer.aggregator.reset_episode()

        pbar.close()
        self.run_evaluation(trainer, num_episodes=5)

    def run_evaluation(self, trainer, num_episodes=5):
        print(f"\n--- Starting Evaluation ({num_episodes} Episodes) ---")
        self.is_evaluating = True
        max_slots = trainer.env.time_manager.max_steps

        for ep in range(num_episodes):
            res = trainer.env.reset()
            obs_upper, obs_lower = res['upper'], res['lower']
            current_upper_state = self.build_upper_state(trainer, obs_upper)
            ep_reward = 0

            for slot in range(max_slots):
                # 1. Upper Decision
                if trainer.env.time_manager.is_new_frame():
                    u_acts, _, _ = self.get_upper_actions(trainer, current_upper_state, obs_upper)
                    trainer.env.step_upper(u_acts)

                # 2. Lower Decision
                t_idx, s_idx, b_sz, t_acc, t_dl = trainer.workload_gen.generate_step()
                if len(t_idx) > 0:
                    gd = self._prepare_group_data(trainer, t_idx, s_idx, obs_lower)
                    metrics = self._calculate_group_metrics(trainer, gd['task_states'])

                    a_ids_list, _, _, _, _ = trainer.shared_lower_agent.choose_action_batch(
                        gd['svc_states'], gd['prev_mfs'], gd['task_states'], gd['masks'], gd['agent_idx'], 
                        deterministic=True, phrase="Proposal_Free", metrics=metrics
                    )

                    f_n, f_m = torch.zeros_like(t_idx), torch.zeros_like(t_idx)
                    for i in range(gd['B']):
                        f_n[gd['pair_idx'] == i] = a_ids_list[i] // trainer.max_models
                        f_m[gd['pair_idx'] == i] = a_ids_list[i] % trainer.max_models

                    results = trainer.env.step_lower(t_idx, s_idx, b_sz, f_n, f_m, t_dl, t_acc)
                    ep_reward += results['reward']

                    curr_mfs = self._compute_node_wise_mfs(trainer, t_idx, s_idx, f_n, f_m, gd['n_src'])
                    self.lower_mf_prev, obs_lower = curr_mfs.detach(), results
                else: trainer.env.time_manager.tick()

                # 3. Upper Metrics
                if trainer.env.time_manager.is_new_frame():
                    res_u = trainer.env.collect_upper_metrics()
                    current_upper_state, obs_upper = self.build_upper_state(trainer, res_u), res_u

            print(f"Eval Episode {ep + 1}: Total Reward={ep_reward:.2f}")

        self.is_evaluating = False