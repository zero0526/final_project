import torch
from matrix_source.agents.d3qn import D3QNAgent
from matrix_source.agents.d3qn_scaffold_v2 import D3QNAgentV2
from matrix_source.trainers.strategies import AlgorithmStrategy
from matrix_source.utils.math_utils import to_binary
from tqdm import tqdm
from matrix_source.trainers.train import Trainer


class D3QNScaffoldStrategy(AlgorithmStrategy):
    def initialize_agents(self, trainer):
        # Upper Agent
        trainer.shared_upper_agent = D3QNAgent(
            node_id=-2, node_type="Edge_Group",
            state_dim=trainer.upper_state_dim,
            action_dim=trainer.upper_action_dim,
            u_action_dim=trainer.upper_u_action_dim,
            mf_hidden_sizes=tuple(trainer.config.hyper_neural["MF_HIDDEN_LAYER"]),
            mf_lr=float(trainer.config.hyper_neural['MF_LR']),
            buffer_min_size=float(trainer.config.hyper_neural["BUFFER_MIN_SIZE"][0]),
            hidden_sizes=trainer.config.hyper_neural['AGENT_HIDDEN_LAYER'],
            lr=float(trainer.config.hyper_neural['UPPER_LR']),
            gamma=trainer.config.hyper_neural['DISCOUNT_FACTOR'],
            alpha=float(trainer.config.hyper_neural['UPDATE_TARGET_COEF']),
            buffer_size=trainer.config.hyper_neural['MEMORY_SIZE'],
            batch_size=trainer.config.hyper_neural['BATCH_SIZE'],
            num_instances=trainer.num_edge_agents,
            device=trainer.device,
            logs_q=True
        )

        # Lower Agent (Uses SCAFFOLD)
        trainer.shared_lower_agent = D3QNAgentV2(
            node_id=-1, node_type="Terminal_Group",
            state_dim=trainer.lower_state_dim,
            action_dim=trainer.lower_action_dim,
            u_action_dim=trainer.lower_u_action_dim,
            mf_hidden_sizes=tuple(trainer.config.hyper_neural["MF_HIDDEN_LAYER"]),
            mf_lr=float(trainer.config.hyper_neural['MF_LR']),
            buffer_min_size=float(trainer.config.hyper_neural["BUFFER_MIN_SIZE"][1]),
            hidden_sizes=tuple(trainer.config.hyper_neural['AGENT_HIDDEN_LAYER']),
            lr=float(trainer.config.hyper_neural['LOWER_LR']),
            gamma=trainer.config.hyper_neural['DISCOUNT_FACTOR'],
            alpha=float(trainer.config.hyper_neural['UPDATE_TARGET_COEF']),
            buffer_size=trainer.config.hyper_neural['MEMORY_SIZE'],
            batch_size=trainer.config.hyper_neural['BATCH_SIZE'],
            num_instances=trainer.num_terminals,
            device=trainer.device,
            logs_q=False,
            use_scaffold=True, # Enabled for Terminals
            total_rounds= trainer.config.hyper_neural['NUMOF_TRAIN_EP']
        )
        # Initialize Cluster Mapping (Terminal -> Edge)
        # terminal_to_comp_node_map is 2D: (num_terminals, num_comp_nodes)
        connectivity = trainer.env.static_matrices['terminal_to_comp_node_map']
        self.terminal_to_node = connectivity.argmax(dim=1).tolist()
            
        self.node_to_terminals = {} # node_id -> list of terminal_instance_ids
        for t_idx, n_id in enumerate(self.terminal_to_node):
            n_id = int(n_id) # ensure it is an int for dict keys
            if n_id not in self.node_to_terminals:
                self.node_to_terminals[n_id] = []
            self.node_to_terminals[n_id].append(t_idx)
            
        print(f"[D3QNScaffoldStrategy] Initialized {len(self.node_to_terminals)} SCAFFOLD clusters.")

    def get_upper_actions(self, trainer, current_upper_state, obs_upper):
        act_matrix = torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device)
        mf_global = obs_upper.get('mean_fields', torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device))
        edge_states = current_upper_state[trainer.edge_node_ids]
        edge_mfs = mf_global[trainer.edge_node_ids]
        instance_indices = torch.tensor([trainer.node_to_instance[nid] for nid in trainer.edge_node_ids], device=trainer.device)
        
        batch_a_ids = trainer.shared_upper_agent.choose_action_batch(
            edge_states, edge_mfs, trainer.eps_upper, trainer.zeta_upper, agent_indices=instance_indices
        )
        
        for i, nid in enumerate(trainer.edge_node_ids):
            act_matrix[nid] = torch.tensor(to_binary(batch_a_ids[i], trainer.num_services), device=trainer.device)
        
        for nid in trainer.env.static_matrices.get("cloud_ids", []):
            act_matrix[nid] = torch.ones(trainer.num_services, device=trainer.device)
        return act_matrix

    def get_lower_actions(self, trainer:Trainer, res_lower, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes):
        obs_dict = res_lower['obs']
        mf_terminals = res_lower['mean_field']
        meta = trainer.env.metadata
        unit_sizes = meta['service_input_size']
        placement_matrix = trainer.env.engine.placement_matrix

        data_sizes = batch_sizes * unit_sizes[s_idx].squeeze(-1)
        s_tasks = torch.stack([data_sizes, tasks_min_accuracy, task_deadlines, meta['service_omega'][s_idx].squeeze(-1)], dim=1).float()
        
        # Advanced indexing using s_idx (which is already a vector per request)
        # placement[:, s_idx] -> (nodes, num_reqs) -> .T -> (num_reqs, nodes)
        service_placements = placement_matrix[:, s_idx].T
        s_backlogs = obs_dict['backlog'][:, s_idx].T * service_placements
        s_cpus = obs_dict['cpu_alloc'][:, s_idx].T * service_placements
        
        states = torch.cat([s_tasks, s_backlogs, s_cpus], dim=1)
        
        states[:, 0] /= trainer.config.norm_data_size
        states[:, 1] /= 100.0
        if states.shape[1] > 4:
            states[:, 4:4+2*trainer.num_nodes] /= trainer.config.norm_gflop
            
        masks = D3QNScaffoldStrategy.calculate_lower_masks(trainer, t_idx, s_idx, tasks_min_accuracy, placement_matrix)
        batch_actions = trainer.shared_lower_agent.choose_action_batch(
            states, mf_terminals[t_idx], trainer.eps_lower, trainer.zeta_lower, masks_batch=masks.to(trainer.device),
            agent_indices=torch.arange(trainer.num_terminals, device=trainer.device)
        )
        
        a_ids = torch.tensor(batch_actions, device=trainer.device)
        return a_ids // trainer.max_models, a_ids % trainer.max_models, masks

    @staticmethod
    def calculate_lower_masks(trainer, t_idx, s_idx, tasks_min_accuracy, placement_matrix):
        num_reqs = len(t_idx)
        # model_accs = trainer.env.metadata['model_accuracies']
        
        # Advanced indexing (placement[:, s_idx].T gives shape: num_reqs, num_nodes)
        current_placements = placement_matrix[:, s_idx].T
        node_masks = current_placements.unsqueeze(-1).expand(-1, -1, trainer.max_models).reshape(num_reqs, -1)
        # acc_mask = (model_accs[s_idx, :] >= tasks_min_accuracy.unsqueeze(1)).float()
        # acc_masks = acc_mask.unsqueeze(1).expand(-1, self.num_nodes, -1).reshape(num_reqs, -1)
        masks = node_masks
                 # * acc_masks)
        
        invalid_mask_rows = (masks.sum(dim=1) == 0)
        masks[invalid_mask_rows] = 1.0
        return masks

    def store_lower_transitions(self, trainer, current_res, next_res, t_idx, s_idx, n_idx, m_idx, masks, next_masks):
        from matrix_source.trainers.train import log_transform
        reward = next_res['reward']
        done = torch.tensor([next_res["new_frame"]]*len(t_idx), dtype=torch.float32, device=trainer.device)
        c_obs, n_obs = current_res['obs'], next_res['obs']
        c_mf, n_mf = current_res['mean_field'], next_res['mean_field']
        cur_placements = masks[:, ::trainer.max_models]
        next_placements = next_masks[:, ::trainer.max_models]
        def build_state(obs, tidx, sidx, place):
            # Advanced indexing on service dim if sidx is a vector
            back = obs['backlog'][:, sidx].T * place
            cpu_a = obs['cpu_alloc'][:, sidx].T * place
            
            st = torch.cat([obs['task_reqs'][tidx], back, cpu_a], dim=1)
            st[:, 0] /= trainer.config.norm_data_size
            st[:, 1] /= 100.0
            if st.shape[1] > 4: st[:, 4:4+2*trainer.num_nodes] /= trainer.config.norm_gflop
            return st

        # Use masks to extract placements
        cur_node_placements = masks[:, ::trainer.max_models]
        next_node_placements = next_masks[:, ::trainer.max_models]
        states = build_state(c_obs, t_idx, s_idx, cur_node_placements)
        next_states = build_state(n_obs, t_idx, s_idx, next_node_placements)
        
        rew_divisor = trainer.config.norm_lower_rw
        reward -= 1.5*next_res["obs"]["virtual_drift"]
        norm_rew = log_transform(reward / (rew_divisor if rew_divisor != 0 else 1.0))
        rewards = torch.full((len(t_idx),), norm_rew, dtype=torch.float32, device=trainer.device)
        rewards -= 0.5*next_res["violations"]
        a_ids = (n_idx * trainer.max_models + m_idx).long()
        
        avg_mf_loss = trainer.shared_lower_agent.store_transition_train_mf_batch(
            states, c_mf[t_idx], n_mf[t_idx], a_ids, rewards, next_states, done, agent_ids=t_idx, masks=masks, next_masks=next_masks
        )
        trainer.aggregator.add_lower(next_res, mf_loss=avg_mf_loss, state=states[0] if len(states) > 0 else None)

    def store_upper_transitions(self, trainer, s_all, ns_all, current_res, next_res, acts_matrix, is_done):
        from matrix_source.trainers.train import log_transform
        reward = next_res['reward_global']
        c_mf = current_res.get('mean_fields', torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device))
        n_mf = next_res['mean_fields']
        dones = torch.full((trainer.num_edge_agents,), 1.0 if is_done else 0.0, dtype=torch.float32, device=trainer.device)
        
        edge_states = s_all[trainer.edge_node_ids]
        edge_next_states = ns_all[trainer.edge_node_ids]
        edge_c_mfs = c_mf[trainer.edge_node_ids]
        edge_n_mfs = n_mf[trainer.edge_node_ids]
        edge_acts = acts_matrix[trainer.edge_node_ids]
        
        pw2 = 2 ** torch.arange(trainer.num_services - 1, -1, -1, device=trainer.device).float()
        edge_a_ids = (edge_acts * pw2).sum(dim=1).long()
        
        rew_divisor = trainer.config.norm_upper_rw
        norm_rew = log_transform(reward / (rew_divisor if rew_divisor != 0 else 1.0))
        rewards = torch.full((trainer.num_edge_agents,), norm_rew, dtype=torch.float32, device=trainer.device)
        instance_indices = torch.tensor([trainer.node_to_instance[nid] for nid in trainer.edge_node_ids], device=trainer.device)

        avg_mf_loss = trainer.shared_upper_agent.store_transition_train_mf_batch(
            edge_states, edge_c_mfs, edge_n_mfs, edge_a_ids, rewards, edge_next_states, dones, agent_ids=instance_indices
        )
        trainer.aggregator.add_upper(next_res, mf_loss=avg_mf_loss, state=edge_states[0] if len(edge_states) > 0 else None)

    def perform_scaffold_aggregation(self, agent: D3QNAgent):
        """
        Cluster-Based Federated aggregation for Multi-Instance D3QNAgent using SCAFFOLD.
        Aggregates terminals locally within each Edge Node cluster.
        """
        if not agent.use_scaffold:
            return

        with torch.no_grad():
            # Parameters in agent.eval_net.get_base_params() are shape (num_instances, ...)
            # We iterate through each cluster of terminals connected to a specific edge or cloud node
            for node_id, terminal_ids in self.node_to_terminals.items():
                if not terminal_ids: continue
                
                t_ids = torch.tensor(terminal_ids, device=agent.device)
                
                # 1. Aggregate local base weights (theta_i) for THIS cluster only
                for i, p in enumerate(agent.eval_net.get_base_params()):
                    # p.data[t_ids] has shape (len(terminal_ids), ...)
                    cluster_p = p.data[t_ids].mean(dim=0, keepdim=True)
                    
                    # SCAFFOLD Control Variate (c_i) update logic:
                    # c_i = c_i - c_cluster + (grad_sum / K)
                    K_dims = [len(terminal_ids)] + [1] * (p.data.dim() - 1)
                    K_expanded = agent.steps_in_round[t_ids].float().view(*K_dims)
                    K_expanded = torch.clamp(K_expanded, min=1.0)
                    
                    # c_cluster for this specific parameter set and this specific cluster
                    # self.c_global[i] in the agent stores the "global" for each instance.
                    # We treat all instances in t_ids as sharing the same global value.
                    
                    # delta_c: (len(terminal_ids), ...)
                    delta_c = agent.grad_sum[i][t_ids] / K_expanded
                    
                    # Update local c_i for cluster members
                    agent.c_i[i][t_ids] = agent.c_i[i][t_ids] - agent.c_global[i][t_ids] + delta_c
                    
                    # Broadcast average cluster_p back to all terminals in this cluster
                    p.data[t_ids] = cluster_p.expand(len(terminal_ids), *cluster_p.shape[1:])
                
                # 2. Update cluster-specific global control variate (c_global)
                # All terminals in t_ids should now share the same new c_global value
                for i in range(len(agent.c_i)):
                    new_cluster_c_global = agent.c_i[i][t_ids].mean(dim=0, keepdim=True)
                    agent.c_global[i][t_ids] = new_cluster_c_global.expand(len(terminal_ids), *new_cluster_c_global.shape[1:])
            
            # 3. Synchronize Target Network (Base part) for all instances
            # (Targets are synchronized to their local eval weights which were just averaged per-cluster)
            for target_p, eval_p in zip(agent.target_net.get_base_params(), agent.eval_net.get_base_params()):
                target_p.data.copy_(eval_p.data)
                
            # 4. Checkpoint for next round
            agent.save_base_initial()

    def perform_scaffold_aggregation_v2(self, agent, round_idx: int = 0):
        """
        Cluster-Based Federated Aggregation for D3QNAgentV2 (split backbone/head).
        Called once per federated round, after all local learning steps are done.

        Phase 1 & 2: averages backbone weights per cluster.
        Phase 3:     backbone frozen, only control variates are synced.
        Always:      c_b and c_h aggregated per cluster.
        """
        if not agent.use_scaffold:
            return

        phase = agent._get_phase(round_idx)

        with torch.no_grad():
            for node_id, terminal_ids in self.node_to_terminals.items():
                if not terminal_ids:
                    continue

                t_ids = torch.tensor(terminal_ids, device=agent.device)

                # Step 1: Update local control variates before we read them
                agent.update_local_cvariates(t_ids)

                # Step 2: Aggregate backbone weights (Phase 1 & 2 only)
                # Fix 4: Do NOT hard-sync target net here — that breaks Polyak averaging.
                # _soft_update() (alpha=0.005) is the only target-net update path.
                if phase < 3:
                    # ✅ FIX 2: Aggregate Backbone weights
                    bone_params = list(agent.eval_net.backbone.parameters())
                    for p in bone_params:
                        cluster_mean = p.data[t_ids].mean(dim=0, keepdim=True)
                        p.data[t_ids] = cluster_mean.expand(len(terminal_ids), *cluster_mean.shape[1:])
                    
                    # ✅ FIX 3: Aggregate MF network weights per cluster
                    # Only aggregate MF when backbone is still training
                    mf_params = list(agent.mf_net.parameters())
                    for p in mf_params:
                        cluster_mean = p.data[t_ids].mean(dim=0, keepdim=True)
                        p.data[t_ids] = cluster_mean.expand(
                            len(terminal_ids), 
                            *cluster_mean.shape[1:]
                        )

                # Step 3: Aggregate backbone control variates -> new c_b_global
                c_b_local_slices = agent.get_c_b_local(t_ids)
                c_b_new_global = [c.mean(dim=0, keepdim=True).expand(len(terminal_ids), *c.shape[1:])
                                for c in c_b_local_slices]
                agent.set_c_b_global(t_ids, c_b_new_global)

        # Reset grad accumulators and step counters for next round
        agent.save_base_initial()
        agent._soft_update()

    def run_training(self, trainer: Trainer):
        num_eps = trainer.config.hyper_neural['NUMOF_TRAIN_EP']
        max_slots = trainer.env.time_manager.max_steps

        for ep in tqdm(range(num_eps), desc="Training"):
            obs = trainer.env.reset()
            obs_upper = obs['upper']
            prev_lower_res = obs['lower']
            
            current_upper_state = self.build_upper_state(trainer, obs_upper) 
            u_acts_matrix = torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device)
            pending_lower_data = None

            for slot in range(max_slots):
                # 1. Update Upper (Decide New placement at start of frame)
                if trainer.env.time_manager.is_new_frame():
                    if slot > 0:
                        res_upper_final = trainer.env.collect_upper_metrics()
                        next_upper_state = self.build_upper_state(trainer, res_upper_final)
                        trainer.aggregator.add_upper(res_upper_final)
                        
                        if trainer.total_lower_steps >= trainer.lower_stable_threshold / 10:
                            self.store_upper_transitions(trainer, current_upper_state, next_upper_state, obs_upper, res_upper_final, u_acts_matrix, False)
                        
                        u_loss = trainer.shared_upper_agent.learn(torch.arange(trainer.num_edge_agents, device=trainer.device))
                        if u_loss is not None:
                            trainer.total_upper_steps += 1
                            trainer.aggregator.record_td_losses(upper_losses=u_loss)

                        current_upper_state = next_upper_state
                        obs_upper = res_upper_final

                    u_acts_matrix = self.get_upper_actions(trainer, current_upper_state, obs_upper)
                    trainer.env.step_upper(u_acts_matrix)
                
                # 2. STORE PREVIOUS TRANSITION (Every Slot)
                if pending_lower_data is not None:
                    p_res, n_res, t_i, s_i, n_i, m_i, cur_m, min_acc = pending_lower_data
                    next_masks = D3QNScaffoldStrategy.calculate_lower_masks(trainer, t_i, s_i, min_acc, trainer.env.engine.placement_matrix)
                    self.store_lower_transitions(trainer, p_res, n_res, t_i, s_i, n_i, m_i, cur_m, next_masks)
                    pending_lower_data = None

                t_idx, s_idx, batch_sizes, tasks_min_accuracy, task_deadlines = trainer.workload_gen.generate_step()
                if len(t_idx) > 0:
                    n_idx, m_idx, masks = self.get_lower_actions(trainer, prev_lower_res, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes)
                    results = trainer.env.step_lower(t_idx, s_idx, batch_sizes, n_idx, m_idx, task_deadlines, tasks_min_accuracy)
                    pending_lower_data = (prev_lower_res, results, t_idx, s_idx, n_idx, m_idx, masks, tasks_min_accuracy)
                    
                    # Accumulate Metrics
                    trainer.aggregator.add_step_matrices(
                        f_alloc=trainer.env.engine.cpu_alloc_matrix,
                        arrivals=results['info']['arrival_matrix'],
                        backlog=trainer.env.engine.backlog_queue.sum(dim=-1)
                    )
                    
                    prev_lower_res = results
                    
                    # train lower
                    res = trainer.shared_lower_agent.learn(torch.arange(trainer.num_terminals, device=trainer.device))
                    if res is not None:
                        trainer.total_lower_steps += 1
                        if isinstance(res, dict):
                            loss = res["loss"]
                            trainer.aggregator.record_q_stats("Terminal_Group", res["q_min"], res["q_max"], res["q_mean"])
                        else:
                            loss = res
                        trainer.aggregator.record_td_losses(lower_losses=loss)
                else:
                    trainer.env.time_manager.tick()

            if pending_lower_data is not None:
                p_res, n_res, t_i, s_i, n_i, m_i, cur_m, min_acc = pending_lower_data
                self.store_lower_transitions(trainer, p_res, n_res, t_i, s_i, n_i, m_i, cur_m, cur_m)

            # update ep and history
            # Federated Aggregation (SCAFFOLD) for lower agents
            self.perform_scaffold_aggregation_v2(trainer.shared_lower_agent, round_idx=ep)

            trainer.update_rates(ep)
            trainer.aggregator.store_history()
            trainer.aggregator.report_episode(ep)
            print(f"--- Global Metrics ---")
            print(f"Lower Samples: {trainer.total_lower_steps} | Upper Samples: {trainer.total_upper_steps}")
            print(f"Zeta Lower: {trainer.zeta_lower:.4f} | Zeta Upper: {trainer.zeta_upper:.4f}")
            print(f"Current Epsilon upper: {trainer.eps_upper:.4f} lower: {trainer.eps_lower:.4f}")

