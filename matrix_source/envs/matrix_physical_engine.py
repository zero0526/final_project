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
        
        # Lower Level Action Tracking (for MARL)
        self.prev_node_indices = torch.zeros(self.num_terminals, dtype=torch.long, device=self.device)
        self.prev_model_indices = torch.zeros(self.num_terminals, dtype=torch.long, device=self.device)
        self.current_task_reqs = torch.zeros((self.num_terminals, 4), device=self.device)
        
        # Accumulators
        self.phi_accumulator = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.reward_global_accumulator = 0.0
        
        # Solver
        f_max_all = self.resource_specs[:, 0].to(self.device).unsqueeze(1)
        self.solver = KKTSolverADMM(
            f_max_node=f_max_all,
            rho=config.admm_rho,
            max_iter=config.admm_max_iter,
            tol=config.admm_tol
        )
        self.immediate_fails = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.arrival_counts_step = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.placement_violations = 0
        self.current_num_tasks = 0
        
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
        self.arrival_counts_step.zero_()
        self.prev_placement_matrix.zero_()
        self.newly_placed_mask.zero_()
        self.reward_global_accumulator = 0.0
        self.phi_accumulator.zero_()
        self.prev_node_indices.zero_()
        self.prev_model_indices.zero_()
        self.current_task_reqs.zero_()
        self.current_num_tasks = 0
        
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
        self.immediate_fails.zero_()
        self.arrival_counts_step.zero_()
        return res

    def get_lower_obs(self):
        mean_field_terminals = self._calc_terminal_mean_field()
        obs = {
            "task_reqs": self.current_task_reqs.clone(),
            "backlog": self.backlog_queue.sum(dim=-1).clone(),
            "cpu_alloc": self.cpu_alloc_matrix.clone()
        }
        return {
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
        # Update action history
        self.prev_node_indices.index_copy_(0, terminal_indices, node_indices)
        self.prev_model_indices.index_copy_(0, terminal_indices, model_indices)

        num_tasks = len(svc_indices)
        node_arrival_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        f_min_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.immediate_fails.zero_()
        self.arrival_counts_step.zero_()
        self.arrival_counts_step.index_put_((node_indices, svc_indices), torch.ones_like(svc_indices, dtype=torch.float), accumulate=True)
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
        
        # Deadline calculation
        t_rem_raw = task_deadlines - trans_delays - cold_delays
        t_q_rem = t_rem_raw - task_max_queue
        
        valid_mask = t_rem_raw >= 1e-4
        # Initialize immediate fails with tasks failing initial checks
        fails_idx = (~valid_mask)
        if fails_idx.any():
            self.immediate_fails.index_put_((node_indices[fails_idx], svc_indices[fails_idx]), torch.ones_like(svc_indices[fails_idx], dtype=torch.float), accumulate=True)
        
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
                self.immediate_fails.index_put_(
                    (vn[fail_hw_mask], vs[fail_hw_mask]), 
                    torch.ones_like(vs[fail_hw_mask], dtype=torch.float), 
                    accumulate=True
                )

            # 2. Xử lý các task hợp lệ bằng Batching trên CPU (Bỏ qua sync GPU chậm)
            valid_hw_mask = hw_mask
            if valid_hw_mask.any():
                # Chuyển dữ liệu sang List của Python (chạy trên RAM)
                sv_n = vn[valid_hw_mask].tolist()
                sv_s = vs[valid_hw_mask].tolist()
                sv_w = vw[valid_hw_mask].tolist()
                sv_t = vt[valid_hw_mask].tolist()
                sv_fmin = p_f_min_all[valid_hw_mask].tolist()
                
                # Fetch pointers về RAM 1 lần duy nhất thay vì item() mỗi vòng
                local_counts = self.backlog_counts.cpu().numpy()
                
                batch_n, batch_s, batch_k = [], [], []
                batch_w, batch_t, batch_fmin = [], [], []
                fail_n, fail_s = [], []
                
                for n, s, w, t, fmin in zip(sv_n, sv_s, sv_w, sv_t, sv_fmin):
                    ptr = local_counts[n, s]
                    if ptr < self.max_K:
                        batch_n.append(n)
                        batch_s.append(s)
                        batch_k.append(ptr)
                        batch_w.append(w)
                        batch_t.append(t)
                        batch_fmin.append(fmin)
                        local_counts[n, s] += 1
                    else:
                        fail_n.append(n)
                        fail_s.append(s)
                
                # Cập nhật hàng loạt (Bulk update) vào GPU
                if batch_n:
                    b_n = torch.tensor(batch_n, device=self.device)
                    b_s = torch.tensor(batch_s, device=self.device)
                    b_k = torch.tensor(batch_k, device=self.device)
                    
                    self.backlog_queue[b_n, b_s, b_k] = torch.tensor(batch_w, dtype=self.backlog_queue.dtype, device=self.device)
                    self.deadline_queue[b_n, b_s, b_k] = torch.tensor(batch_t, dtype=self.deadline_queue.dtype, device=self.device)
                    self.f_min_queue[b_n, b_s, b_k] = torch.tensor(batch_fmin, dtype=self.f_min_queue.dtype, device=self.device)
                    self.backlog_counts.copy_(torch.from_numpy(local_counts).to(self.device))
                
                if fail_n:
                    f_n = torch.tensor(fail_n, device=self.device)
                    f_s = torch.tensor(fail_s, device=self.device)
                    self.immediate_fails.index_put_((f_n, f_s), torch.ones_like(f_s, dtype=torch.float), accumulate=True)
            
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

    def execute_and_collect_metrics(self, node_arrival_matrix, trans_energy_total, cold_delays):
        t0 = time.perf_counter()
        current_backlog_total = self.backlog_queue.sum(dim=-1)
        count_before = self.backlog_counts.clone()
        prev_cpu_alloc = self.cpu_alloc_matrix.clone()
        
        self.backlog_queue, actual_processed, in_slot_violation_mask = ops.deplete_float_queue(
            self.backlog_queue, self.deadline_queue, self.cpu_alloc_matrix, self.slot_duration
        )
        
        self.backlog_queue, self.deadline_queue, self.f_min_queue, expired_counts_tensor = ops.age_and_clean_dual_queue(
            self.backlog_queue, self.deadline_queue, self.f_min_queue, in_slot_violation_mask, self.slot_duration
        )
        
        # Update counts
        self.backlog_counts = (self.backlog_queue > 1e-6).sum(dim=-1)
        
        violate_step_tensor = expired_counts_tensor + self.immediate_fails
        num_violations = int(violate_step_tensor.sum().item())
        
        # Success count calculation: before + arrivals - current - failed
        # Careful: arrivals_counts_step only includes those that ENTERED or were IMMEDIATELY FAILED.
        # So count_before + arrivals_counts_step is the total tasks we dealt with.
        success_qos_tensor = (count_before + self.arrival_counts_step - self.backlog_counts - violate_step_tensor).clamp(min=0)
        
        total_drift = ops.calculate_lyapunov_drift(current_backlog_total, node_arrival_matrix, self.cpu_alloc_matrix * self.slot_duration)
        comp_energy = ops.compute_batch_energy(
            self.cpu_alloc_matrix, actual_processed, self.energy_coef, 
            cold_delays, epsilon_cold=self.energy_cold_start
        )
        total_energy = comp_energy + trans_energy_total
        
        f1 = total_drift + self.lypa_coef * total_energy
        self.reward_global_accumulator += f1.item()
        
        qos_penalty = self.omega_1 * torch.exp(torch.tensor(self.omega_2 * num_violations, device=self.device))
        reward = -(f1 + qos_penalty)
        obs = {
            "total_drift": total_drift,
            "task_reqs": self.current_task_reqs.clone(),
            "backlog": self.backlog_queue.sum(dim=-1).clone(),
            "cpu_alloc": self.cpu_alloc_matrix.clone()
        }
        info = {
            "num_tasks": self.current_num_tasks,
            "immediate_fails": int(self.immediate_fails.sum().item()),
            "expired_count": int(violate_step_tensor.sum().item() - self.immediate_fails.sum().item()),
            "remaining": int(self.backlog_counts.sum().item()),
            "success_qos": {i: success_qos_tensor[i].cpu().numpy() for i in range(self.num_nodes)},
            "violate_qos": {i: violate_step_tensor[i].cpu().numpy() for i in range(self.num_nodes)},
            "arrival_matrix": self.arrival_counts_step.clone() # Return snapshot
        }
        res = {
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
        
        if self.profiling_step % 500 == 0:
            print(f"\n--- Engine Profiling (Step {self.profiling_step}) ---")
            for k, v in sorted(self.prof.items()):
                print(f"  {k:20s}: {v*1000/500:8.3f} ms/step")
            self.prof.clear()
            
        return res
