import torch
import random
import time
from collections import defaultdict
import matrix_source.utils.tensor_ops as ops
from matrix_source.models.resource_solver import KKTSolverADMM

class MatrixPhysicalEngine:
    def __init__(self, config, static_matrices, metadata, device):
        self.config = config
        self.device = device
        
        # Static Parameters
        self.resource_specs = static_matrices['resource_matrix']
        self.delay_matrix = static_matrices['transmission_delay_matrix']
        self.terminal_to_node_map = static_matrices['terminal_to_comp_node_map']
        self.adj_matrix = static_matrices['adj_matrix'].to(device)
        self.terminal_adj_matrix = static_matrices['terminal_adj_matrix'].to(device)
        
        self.service_omega = metadata['service_omega'].to(device)
        self.service_deadlines = metadata['service_deadlines'].to(device)
        self.service_input_size = metadata['service_input_size'].to(device)
        self.model_workloads = metadata['model_workloads'].to(device)
        self.model_accuracies = metadata['model_accuracies'].to(device)
        self.max_queue_delay = static_matrices['max_queue_delay'].to(device)
        self.service_size = (metadata['service_size'] / 1024.0).to(device) # MB -> GB
        
        # Dynamics Configuration
        self.slot_duration = config.hyper_neural["SLOT_DURATION"]
        self.lypa_coef = config.lypa_coef
        self.energy_coef = config.energy_coef
        self.cold_start_delay_min = config.cold_start_time.get("min")
        self.cold_start_delay_max = config.cold_start_time.get("max")
        self.energy_cold_start = config.cold_start_energy_coef
        
        # Reward Weights
        self.omega_1 = config.hyper_neural["OMEGA_Q1"]
        self.omega_2 = config.hyper_neural["OMEGA_Q2"]
        
        # State Tensors
        self.num_nodes = self.resource_specs.shape[0]
        self.num_services = self.service_omega.shape[0]
        self.num_terminals = self.terminal_to_node_map.shape[0]
        self.max_K = config.max_queue_size
        
        self.backlog_queue = torch.zeros((self.num_nodes, self.num_services, self.max_K), device=self.device)
        self.backlog_counts = torch.zeros((self.num_nodes, self.num_services), dtype=torch.long, device=self.device)
        self.deadline_queue = torch.zeros((self.num_nodes, self.num_services, self.max_K), device=self.device)
        self.f_min_queue = torch.zeros((self.num_nodes, self.num_services, self.max_K), device=self.device)
        
        self.cpu_alloc_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.placement_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.prev_placement_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.newly_placed_mask = torch.zeros((self.num_nodes, self.num_services), dtype=torch.bool, device=self.device)
        self.used_resources = torch.zeros((self.num_nodes, 4), device=self.device)
        
        # New Q: Terminal ID tracking
        self.terminal_queue = torch.zeros((self.num_nodes, self.num_services, self.max_K), dtype=torch.long, device=self.device) - 1
        self.terminal_fail_counts = torch.zeros((self.num_terminals, self.num_services), device=self.device)
        
        # Lower Level Action Tracking (for MARL)
        self.prev_node_indices = torch.zeros(self.num_terminals, dtype=torch.long, device=self.device)
        self.prev_model_indices = torch.zeros(self.num_terminals, dtype=torch.long, device=self.device)
        self.current_task_reqs = torch.zeros((self.num_terminals, 4), device=self.device)
        
        # Accumulators
        self.phi_accumulator = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.reward_global_accumulator = 0.0
        
        # Solver
        self.f_max_all = self.resource_specs[:, 0].to(self.device).unsqueeze(1)
        self.solver = KKTSolverADMM(
            f_max_node=self.f_max_all,
            rho=config.admm_rho,
            max_iter=config.admm_max_iter,
            tol=config.admm_tol
        )
        self.immediate_fails = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.fail_placement = torch.zeros((self.num_nodes, self.num_services), device=self.device) # New: Penalty for wrong node
        self.fail_deadline = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.fail_hw = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.fail_queue = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.service_hw_deficit = torch.zeros(self.num_services, device=self.device)
        self.service_hw_fail_count = torch.zeros(self.num_services, device=self.device)
        
        self.arrival_counts_step = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.placement_violations = 0
        self.current_num_tasks = 0

        # --- Virtual Queue (Augmented Lyapunov) ---
        # Z_queue accumulates REJECTED workload (hw-limit, placement, deadline)
        # to penalize agents that exploit the "kill task early" trick.
        # Z decays each slot by the node's max capacity (virtual service rate).
        self.Z_queue = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.beta_virtual = config.hyper_neural.get('BETA_VIRTUAL_DRIFT', 1.0)
        # rejected_workload_step: filled each slot in process_arrivals
        self.rejected_workload_step = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        
        # Profiling
        self.prof = defaultdict(float)
        self.prof_counts = defaultdict(int)
        self.profiling_step = 0

    def reset(self):
        self.backlog_queue.zero_()
        self.backlog_counts.zero_()
        self.deadline_queue.zero_()
        self.f_min_queue.zero_()
        self.immediate_fails.zero_()
        self.fail_placement.zero_()
        self.fail_deadline.zero_()
        self.fail_hw.zero_()
        self.fail_queue.zero_()
        self.terminal_fail_counts.zero_()
        self.terminal_queue.fill_(-1)
        self.service_hw_deficit.zero_()
        self.service_hw_fail_count.zero_()
        self.arrival_counts_step.zero_()
        self.prev_placement_matrix.zero_()
        self.newly_placed_mask.zero_()
        self.reward_global_accumulator = 0.0
        self.phi_accumulator.zero_()
        self.prev_node_indices.zero_()
        self.prev_model_indices.zero_()
        self.current_task_reqs.zero_()
        self.current_num_tasks = 0
        self.Z_queue.zero_()
        self.rejected_workload_step.zero_()
        
        obs_upper = {
            "actions": self.placement_matrix.clone(),
            "phi_prob": torch.zeros_like(self.phi_accumulator),
            'mean_fields': torch.zeros((self.num_nodes, self.num_services), device=self.device),
            "resources": torch.zeros((self.num_nodes, 3), device=self.device)
        }
        obs_lower = self.get_lower_obs()
        return {"upper": obs_upper, "lower": obs_lower}

    def collect_upper_metrics(self):
        neighbor_count = self.adj_matrix.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_fields = (self.adj_matrix @ self.placement_matrix) / neighbor_count
        phi_prob = ops.transform2prob(self.phi_accumulator)
        
        # Resource utilization: CPU, RAM, HDD (3 dimensions)
        # Assuming resource_specs columns: 0=CPU, 1=RAM, 2=HDD, 3=Price?
        # we skip price and take 3. Normalized by total capacity.
        # used_resources: (N, 4)
        cpu_util = self.used_resources[:, 0] / self.resource_specs[:, 0].clamp(min=1.0)
        ram_util = self.used_resources[:, 1] / self.resource_specs[:, 1].clamp(min=1.0)
        hdd_util = self.used_resources[:, 2] / self.resource_specs[:, 2].clamp(min=1.0)
        resources = torch.stack([cpu_util, ram_util, hdd_util], dim=1) # (N, 3)

        res = {
            "actions": self.placement_matrix.clone(),
            "phi_prob": phi_prob,
            "mean_fields": mean_fields,
            "resources": resources,
            "reward_global": -float(self.reward_global_accumulator)
        }
        self.phi_accumulator.zero_()
        self.reward_global_accumulator = 0.0
        self.current_num_tasks = 0
        return res

    def get_lower_obs(self):
        mean_field_terminals = self._calc_terminal_mean_field()
        obs = {
            "external_snack": torch.zeros((self.num_nodes, self.num_services), device=self.device),
            "task_reqs": self.current_task_reqs.clone(),
            "backlog": self.backlog_queue.sum(dim=-1).clone(),
            "cpu_alloc": self.cpu_alloc_matrix.clone()
        }
        return {
            "pre_reward": 0,
            "obs": obs,
            "mean_field": mean_field_terminals,
            "prev_actions": {
                "node_selection": self.prev_node_indices.clone(),
                "model_selection": self.prev_model_indices.clone()
            }
        }

    def _calc_terminal_mean_field(self):
        neighbor_count = self.terminal_adj_matrix.sum(dim=1, keepdim=True).clamp(min=1.0)
        node_one_hot = torch.nn.functional.one_hot(self.prev_node_indices, num_classes=self.num_nodes).float()
        num_models = self.model_workloads.shape[1]
        model_one_hot = torch.nn.functional.one_hot(self.prev_model_indices, num_classes=num_models).float()
        
        mf_node = (self.terminal_adj_matrix @ node_one_hot) / neighbor_count
        mf_model = (self.terminal_adj_matrix @ model_one_hot) / neighbor_count
        return torch.cat([mf_node, mf_model], dim=-1)

    def set_upper_action(self, new_placement):
        self.prev_placement_matrix = self.placement_matrix.clone()
        valid_placement = new_placement.clone()
        omega_1_mask = (self.service_omega.squeeze() == 1)
        omega_0_mask = (self.service_omega.squeeze() == 0)
        
        ram_reqs = (valid_placement * omega_1_mask) @ self.service_size
        ram_over = ram_reqs > self.resource_specs[:, 1].unsqueeze(1).to(self.device)
        hdd_reqs = (valid_placement * omega_0_mask) @ self.service_size
        hdd_over = hdd_reqs > self.resource_specs[:, 2].unsqueeze(1).to(self.device)
        
        over_mask = (ram_over | hdd_over).expand_as(valid_placement)
        valid_placement[over_mask] = self.placement_matrix[over_mask]
        self.placement_violations = over_mask.float().sum().item()
        
        self.placement_matrix = valid_placement
        self.newly_placed_mask = (self.placement_matrix > 0) & (self.prev_placement_matrix == 0)
        self.cpu_alloc_matrix *= self.placement_matrix

    def process_arrivals(self, terminal_indices, svc_indices, node_indices, model_indices, task_batch_sizes, task_deadlines, task_accuracies):
        t0 = time.perf_counter()
        
        # Reset slot-wise counters to prevent accumulation across slots
        self.immediate_fails.zero_()
        self.fail_placement.zero_()
        self.fail_deadline.zero_()
        self.fail_hw.zero_()
        self.fail_queue.zero_()
        self.arrival_counts_step.zero_()
        self.service_hw_deficit.zero_()
        self.service_hw_fail_count.zero_()
        self.rejected_workload_step.zero_()
        
        # Update action history - Flattening strictly for CUDA
        t_idx_flat = terminal_indices.reshape(-1).long()
        n_idx_flat = node_indices.reshape(-1).long()
        m_idx_flat = model_indices.reshape(-1).long()
        
        self.prev_node_indices[t_idx_flat] = n_idx_flat
        self.prev_model_indices[t_idx_flat] = m_idx_flat
        
        num_tasks = len(svc_indices)
        node_arrival_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        f_min_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.immediate_fails.zero_()
        self.fail_deadline.zero_()
        self.fail_hw.zero_()
        self.fail_queue.zero_()
        self.terminal_fail_counts.zero_()
        self.service_hw_deficit.zero_()
        self.service_hw_fail_count.zero_()
        self.arrival_counts_step.zero_()
        
        # Enforce Long 1D for index_put_
        n_at = node_indices.reshape(-1).long()
        s_at = svc_indices.reshape(-1).long()
        ones = torch.ones(n_at.shape[0], dtype=torch.float, device=self.device)
        self.arrival_counts_step.index_put_((n_at, s_at), ones, accumulate=True)
        self.current_task_reqs.zero_()
        

        if num_tasks == 0:
            self.current_num_tasks = 0
            return node_arrival_matrix, 0.0, torch.tensor([], device=self.device), f_min_matrix

        self.current_num_tasks = num_tasks
        
        # Transmission metrics

        # Transmission metrics
        src_node_indices = torch.argmax(self.terminal_to_node_map[terminal_indices], dim=1)
        base_input_sizes = self.service_input_size[svc_indices].squeeze()
        task_data_sizes = base_input_sizes * task_batch_sizes
        
        trans_delays, trans_energy_tasks = ops.compute_transmission_metrics(
            src_node_indices, node_indices, self.delay_matrix, task_data_sizes, 
            beta=self.config.transmission_coef
        )
        trans_energy_total = trans_energy_tasks.sum().item()

        # Task details
        base_workloads = self.model_workloads[svc_indices, model_indices]
        task_workloads = base_workloads * task_batch_sizes
        task_max_queue = self.max_queue_delay[node_indices, svc_indices]
        task_types = self.service_omega[svc_indices].squeeze()
        
        # Log requirements
        req_features = torch.stack([task_data_sizes, task_deadlines, task_accuracies, task_types], dim=-1)
        self.current_task_reqs.index_copy_(0, terminal_indices, req_features)

        # Cold start
        task_cold_start = self.newly_placed_mask[node_indices, svc_indices] & (task_types == 0)
        rand_vals = torch.rand(task_cold_start.shape, device=self.device)
        cold_delays = task_cold_start.float() * (rand_vals * (self.cold_start_delay_max - self.cold_start_delay_min) + self.cold_start_delay_min)
        
        # Placement check: Are tasks being sent to valid nodes?
        placement_mask = self.placement_matrix[node_indices, svc_indices] > 0
        if (~placement_mask).any():
            fn_p = node_indices[~placement_mask]
            fs_p = svc_indices[~placement_mask]
            ft_p = terminal_indices[~placement_mask]
            self.immediate_fails.index_put_((fn_p, fs_p), torch.ones_like(fs_p, dtype=torch.float), accumulate=True)
            self.fail_placement.index_put_((fn_p, fs_p), torch.ones_like(fs_p, dtype=torch.float), accumulate=True)
            self.terminal_fail_counts.index_put_((ft_p, fs_p), torch.ones_like(ft_p, dtype=torch.float), accumulate=True)

        # Deadline calculation
        t_rem_raw = task_deadlines - trans_delays
        t_q_rem = t_rem_raw - task_max_queue
        
        valid_mask = t_rem_raw >= 1e-4
        # Initialize immediate fails with tasks failing initial checks
        fails_idx = (~valid_mask)
        if fails_idx.any():
            fn = node_indices[fails_idx]
            fs = svc_indices[fails_idx]
            ft = terminal_indices[fails_idx]
            self.immediate_fails.index_put_((fn, fs), torch.ones_like(fs, dtype=torch.float), accumulate=True)
            self.fail_deadline.index_put_((fn, fs), torch.ones_like(fs, dtype=torch.float), accumulate=True)
            self.terminal_fail_counts.index_put_((ft, fs), torch.ones_like(ft, dtype=torch.float), accumulate=True)
        
        if valid_mask.any():
            vn = node_indices[valid_mask]
            vs = svc_indices[valid_mask]
            vw = task_workloads[valid_mask]
            vt = t_rem_raw[valid_mask]
            vq = t_q_rem[valid_mask]
            vb = task_batch_sizes[valid_mask]
            
            self.phi_accumulator.index_put_((vn, vs), vb, accumulate=True)
            node_max_f = self.resource_specs[:, 0]
            
            p_f_min_all = vw / (vq + 1e-9)
            hw_mask = p_f_min_all <= node_max_f[vn]
            
            # 1. Các task không đủ phần cứng
            fail_hw_mask = ~hw_mask
            if fail_hw_mask.any():
                fhn = vn[fail_hw_mask]
                fhs = vs[fail_hw_mask]
                fht = vn[fail_hw_mask]
                self.immediate_fails.index_put_((fhn, fhs), torch.ones_like(fhs, dtype=torch.float), accumulate=True)
                self.fail_hw.index_put_((fhn, fhs), torch.ones_like(fhs, dtype=torch.float), accumulate=True)
                self.terminal_fail_counts.index_put_((fht, fhs), torch.ones_like(fht, dtype=torch.float), accumulate=True)

                # --- Virtual Queue: accumulate REJECTED workload (hw-fail) ---
                # We count the full task workload as "rejected work"
                self.rejected_workload_step.index_put_((fhn, fhs), vw[fail_hw_mask], accumulate=True)
                # -------------------------------------------------------------

                # Deficit Analysis
                deficit = (vw[fail_hw_mask] / node_max_f[vn][fail_hw_mask]) - vq[fail_hw_mask]
                self.service_hw_deficit.index_put_((fhs,), deficit, accumulate=True)
                self.service_hw_fail_count.index_put_((fhs,), torch.ones_like(fhs, dtype=torch.float), accumulate=True)

            # Vectorized Queue Pointer Calculation
            if hw_mask.any():
                v_hw_n, v_hw_s = vn[hw_mask], vs[hw_mask]
                v_hw_w, v_hw_t = vw[hw_mask], vt[hw_mask]
                v_hw_fmin = p_f_min_all[hw_mask]

                # We need to compute indices for (node, service) pairs as if they were flattened
                # But since multiple tasks for the same (n, s) might arrive in the same slot,
                # we use cumsum to find their positions relative to the current counts.
                flat_idx = v_hw_n * self.num_services + v_hw_s
                
                # Sort arrivals by (node, service) to group them
                sort_idx = torch.argsort(flat_idx)
                v_hw_n, v_hw_s, v_hw_w, v_hw_t, v_hw_fmin, flat_idx = \
                    v_hw_n[sort_idx], v_hw_s[sort_idx], v_hw_w[sort_idx], v_hw_t[sort_idx], v_hw_fmin[sort_idx], flat_idx[sort_idx]

                # Current counts for these (node, service) pairs
                base_counts = self.backlog_counts[v_hw_n, v_hw_s]
                
                # Rank tasks within the same (node, service) pair in this batch
                # Using a trick with diff and cumsum to find ranks
                diffs = torch.cat([torch.tensor([1], device=self.device), (flat_idx[1:] != flat_idx[:-1]).long()])
                ranks_in_batch = torch.cumsum(torch.ones_like(flat_idx), dim=0) - 1
                group_start_rank = torch.masked_select(ranks_in_batch, diffs.bool())
                # Expanded starts to match every element in flat_idx
                expanded_starts = torch.repeat_interleave(group_start_rank, torch.diff(torch.cat([torch.where(diffs)[0], torch.tensor([len(flat_idx)], device=self.device)])))
                relative_ranks = ranks_in_batch - expanded_starts
                
                absolute_ks = base_counts + relative_ranks
                
                # Mask out tasks that still exceed capacity
                valid_queue_mask = absolute_ks < self.max_K
                
                if valid_queue_mask.any():
                    vq_n, vq_s, vq_k = v_hw_n[valid_queue_mask], v_hw_s[valid_queue_mask], absolute_ks[valid_queue_mask]
                    self.backlog_queue[vq_n, vq_s, vq_k] = v_hw_w[valid_queue_mask]
                    self.deadline_queue[vq_n, vq_s, vq_k] = v_hw_t[valid_queue_mask]
                    self.f_min_queue[vq_n, vq_s, vq_k] = v_hw_fmin[valid_queue_mask]
                    self.terminal_queue[vq_n, vq_s, vq_k] = terminal_indices[valid_mask][hw_mask][sort_idx][valid_queue_mask]
                    
                    # Update counts: Need to find max relative rank per node-service + 1
                    # A safe way is to just use index_put with atomic add (not supported for max, but we can do it with a counts update)
                    added_counts = torch.zeros_like(self.backlog_counts)
                    added_counts.index_put_((vq_n, vq_s), torch.ones_like(vq_n, dtype=torch.long), accumulate=True)
                    self.backlog_counts += added_counts

                # Queue Full Failures
                if (~valid_queue_mask).any():
                    fq_n, fq_s = v_hw_n[~valid_queue_mask], v_hw_s[~valid_queue_mask]
                    fq_t = terminal_indices[valid_mask][hw_mask][sort_idx][~valid_queue_mask]
                    self.immediate_fails.index_put_((fq_n, fq_s), torch.ones_like(fq_s, dtype=torch.float), accumulate=True)
                    self.fail_queue.index_put_((fq_n, fq_s), torch.ones_like(fq_s, dtype=torch.float), accumulate=True)
                    self.terminal_fail_counts.index_put_((fq_t, fq_s), torch.ones_like(fq_t, dtype=torch.float), accumulate=True)
            
            node_arrival_matrix.index_put_((vn, vs), vw, accumulate=True)
            
        f_min_matrix = self.f_min_queue.max(dim=-1)[0]
        self.prof['1_process_arrivals'] += time.perf_counter() - t0
        return node_arrival_matrix, trans_energy_total, cold_delays, f_min_matrix

    def optimize_allocation(self, node_arrival_matrix, f_min_matrix):
        t0 = time.perf_counter()
        current_backlog_total = self.backlog_queue.sum(dim=-1)
        G = current_backlog_total * self.placement_matrix
        G[G < 1e-3] = 0
        Z = self.lypa_coef * self.energy_coef * self.placement_matrix * node_arrival_matrix
        f_max = (self.resource_specs[:, 0:1] * self.placement_matrix).to(self.device)
        f_min = f_min_matrix.clamp(max=f_max)
        self.cpu_alloc_matrix = self.solver.solve(G, Z, f_min, f_max, debug=False)
        self.prof['2_optimize'] += time.perf_counter() - t0

    def execute_and_collect_metrics(self, node_arrival_matrix, trans_energy_total, cold_delays, is_discrete):
        # --- Update Virtual Queue Z BEFORE depletion ---
        # Z decays by max_node_capacity * slot_duration (virtual max throughput per slot)
        f_max = self.resource_specs[:, 0].unsqueeze(1) * self.placement_matrix  # (N, S)
        virtual_service = f_max * self.slot_duration   # max work node can do for each service
        self.Z_queue = (self.Z_queue - virtual_service + self.rejected_workload_step).clamp(min=0)
        t0 = time.perf_counter()
        current_backlog_total = self.backlog_queue.sum(dim=-1)
        count_before = self.backlog_counts.clone()
        prev_cpu_alloc = self.cpu_alloc_matrix.clone()
        
        # Terminal -> source node mapping (num_terminals,)
        src_node_mapping = torch.argmax(self.terminal_to_node_map, dim=1).long()
        
        self.backlog_queue, actual_processed, local_processed = ops.deplete_float_queue(
            self.backlog_queue, self.deadline_queue, self.terminal_queue, src_node_mapping, self.cpu_alloc_matrix, self.slot_duration
        )
        
        self.backlog_queue, self.deadline_queue, processed_aux, expired_counts_tensor, failed_terminal_ids, failed_svc_ids = ops.age_and_clean_dual_queue(
            self.backlog_queue, self.deadline_queue, self.slot_duration, self.f_min_queue, self.terminal_queue
        )

        self.f_min_queue, self.terminal_queue = processed_aux[0], processed_aux[1]
        
        if failed_terminal_ids is not None and len(failed_terminal_ids) > 0:
            self.terminal_fail_counts.index_put_((failed_terminal_ids.long(), failed_svc_ids.long()), torch.ones_like(failed_terminal_ids, dtype=torch.float), accumulate=True)

        # Update counts
        self.backlog_counts = (self.backlog_queue > 1e-6).sum(dim=-1)
        
        violate_step_tensor = expired_counts_tensor + self.immediate_fails
        num_violations = violate_step_tensor.sum()
        
        # Success count calculation
        # Cast to float to avoid precision loss and correct formula (remove arrival_counts_step)
        count_before_f = count_before.float()
        backlog_counts_f = self.backlog_counts.float()
        success_qos_tensor = (count_before_f - backlog_counts_f - violate_step_tensor).clamp(min=0)
        # N x S: resources spent on tasks offloaded FROM other nodes
        total_capacity_used = self.cpu_alloc_matrix * self.slot_duration
        external_snack = (total_capacity_used - local_processed).clamp(min=0)
        
        total_drift = ops.calculate_lyapunov_drift(current_backlog_total, node_arrival_matrix, self.cpu_alloc_matrix * self.slot_duration)

        # --- Virtual Drift (Augmented Lyapunov) ---
        # Penalizes accumulated rejected workload at each (node, service).
        # virtual_drift = beta * sum(Z * rejected_workload_step)
        # If an agent keeps sending tasks that die at hw limit, Z grows and
        # this term keeps growing each slot -> agent cannot escape punishment.
        virtual_drift = self.beta_virtual * (self.Z_queue * self.rejected_workload_step).sum()
        # -----------------------------------------

        comp_energy = ops.compute_batch_energy(
            self.cpu_alloc_matrix, actual_processed, self.energy_coef, 
            cold_delays, epsilon_cold=self.energy_cold_start
        )
        total_energy = comp_energy + trans_energy_total
        
        f1 = total_drift + self.lypa_coef * total_energy
        self.reward_global_accumulator += f1
        
        # Refined QoS penalty
        qos_penalty = self.omega_1 * num_violations.float()
        
        reward = -(f1 +qos_penalty)
        obs = {
            "virtual_drift": virtual_drift,
            "total_drift": total_drift,
            "virtual_drift": virtual_drift,
            # N x S: CPU capacity spent on externally-offloaded tasks
            "external_snack": external_snack.clone(),
            "task_reqs": self.current_task_reqs.clone(),
            "backlog": current_backlog_total.clone(),
            "cpu_alloc": self.cpu_alloc_matrix.clone()
        }
        info = {
            "num_tasks": self.current_num_tasks,
            "external_snack": external_snack.clone(),
            "immediate_fails": self.immediate_fails.sum(),
            "expired_count": violate_step_tensor.sum() - self.immediate_fails.sum(),
            "remaining": self.backlog_counts.sum(),
            "success_qos": success_qos_tensor, 
            "violate_qos": violate_step_tensor,
            "terminal_fail_counts": self.terminal_fail_counts.clone(),
            "arrival_matrix": self.arrival_counts_step.clone(),
            "fail_reasons": {
                "deadline": self.fail_deadline.sum(),
                "hardware": self.fail_hw.sum(),
                "queue_full": self.fail_queue.sum(),
                "invalid_placement": self.fail_placement.sum(),
                "expired": expired_counts_tensor.sum(),
                "hw_deficit_per_svc": self.service_hw_deficit,
                "hw_fail_count_per_svc": self.service_hw_fail_count
            }
        }
        res = {
            "pre_reward": -total_energy,
            "reward": reward,
            "energy": total_energy,
            "violations": num_violations,
            "obs": obs,
            "info": info,
            "mean_field": self._calc_terminal_mean_field(),
            "prev_actions": {
                "node_selection": self.prev_node_indices.clone(),
                "model_selection": self.prev_model_indices.clone()
            }
        }
        self.prof['3_execute'] += time.perf_counter() - t0
        self.profiling_step += 1
        
        # Reset newly_placed_mask after the first slot of the timeframe
        self.newly_placed_mask.zero_()
        
        # if self.profiling_step % 500 == 0:
        #     print(f"\n--- Engine Profiling (Step {self.profiling_step}) ---")
        #     for k, v in sorted(self.prof.items()):
        #         print(f"  {k:20s}: {v*1000/500:8.3f} ms/step")
        #     self.prof.clear()
            
        return res
