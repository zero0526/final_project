import random
import torch
import matrix_source.utils.tensor_ops as ops
from matrix_source.models.resource_solver import KKTSolverADMM

class MatrixPhysicalEngine:
    def __init__(self, config, static_matrices, metadata, device):
        self.config = config
        self.device = device
        
        # Static Parameters
        self.resource_specs = static_matrices['resource_specs']
        self.delay_matrix = static_matrices['delay_matrix']
        self.terminal_to_node_map = static_matrices['terminal_to_node_map']
        
        self.service_omega = metadata['service_omega']
        self.service_deadlines = metadata['service_deadlines']
        self.service_input_size = metadata['service_input_size']
        self.model_workloads = metadata['model_workloads']
        self.max_queue_delay = metadata['max_queue_delay']
        self.service_size = metadata['service_size'] / 1024.0 # Convert MB to GB
        
        # Dynamics Configuration
        self.slot_duration = config.get('slot_duration', 0.1)
        self.lypa_coef = config.get('lypa_coef', 10.0)
        self.energy_coef = config.get('energy_coef', 1.0)
        self.cold_start_delay_min = config.get('cold_start_delay_min', 0.5)
        self.cold_start_delay_max = config.get('cold_start_delay_max', 0.85)
        self.energy_cold_start = config.get('energy_cold_start', 10.0)
        
        # State Tensors (3D Float Queue)
        self.num_nodes = self.resource_specs.shape[0]
        self.num_services = self.service_omega.shape[0]
        self.max_K = config.get('max_queue_size', 100) # Thiết lập K=100 theo yêu cầu
        
        # (Nodes x Services x Max_K)
        self.backlog_queue = torch.zeros((self.num_nodes, self.num_services, self.max_K), device=self.device)
        self.deadline_queue = torch.zeros((self.num_nodes, self.num_services, self.max_K), device=self.device)
        
        self.cpu_alloc_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.placement_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.prev_placement_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.newly_placed_mask = torch.zeros((self.num_nodes, self.num_services), dtype=torch.bool, device=self.device)
        self.used_resources = torch.zeros((self.num_nodes, 4), device=self.device)
        
        # Solver
        self.solver = KKTSolverADMM(config, self.resource_specs, self.service_omega)
        self.immediate_fails = 0
        self.placement_violations = 0

    def reset(self):
        self.backlog_queue.zero_()
        self.deadline_queue.zero_()
        self.cpu_alloc_matrix.zero_()
        self.placement_matrix.zero_()
        self.prev_placement_matrix.zero_()
        self.newly_placed_mask.zero_()
        self.used_resources.zero_()
        self.immediate_fails = 0
        self.placement_violations = 0

    def update_placement(self, new_placement):
        self.prev_placement_matrix = self.placement_matrix.clone()
        valid_placement = new_placement.clone()
        omega_1 = (self.service_omega.squeeze() == 1)
        omega_0 = (self.service_omega.squeeze() == 0)
        
        ram_reqs = (valid_placement * omega_1) @ self.service_size
        ram_over = ram_reqs > self.resource_specs[:, 1]
        hdd_reqs = (valid_placement * omega_0) @ self.service_size
        hdd_over = hdd_reqs > self.resource_specs[:, 2]
        
        over_mask = ram_over | hdd_over
        valid_placement[over_mask] = self.placement_matrix[over_mask]
        self.placement_violations = over_mask.float().sum().item()
        
        self.placement_matrix = valid_placement
        self.newly_placed_mask = (self.placement_matrix > 0) & (self.prev_placement_matrix == 0)
        self.cpu_alloc_matrix *= self.placement_matrix
        return self.placement_violations

    def process_arrivals(self, terminal_indices, svc_indices, node_indices, model_indices):
        num_tasks = len(svc_indices)
        trans_energy_total = 0.0
        node_arrival_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.immediate_fails = 0
        
        if num_tasks == 0:
            return node_arrival_matrix, trans_energy_total

        src_node_indices = torch.argmax(self.terminal_to_node_map[terminal_indices], dim=1)
        task_data_sizes = self.service_input_size[svc_indices].squeeze()
        
        trans_delays, trans_energy_tasks = ops.compute_transmission_metrics(
            src_node_indices, node_indices, self.delay_matrix, task_data_sizes, 
            beta=self.config.get('trans_energy_beta', 1e-6)
        )
        trans_energy_total = trans_energy_tasks.sum()

        task_workloads = self.model_workloads[svc_indices, model_indices]
        task_mean_deadlines = self.service_deadlines[svc_indices, 2] 
        task_max_queue = self.max_queue_delay[node_indices, svc_indices]
        
        task_cold_start = self.newly_placed_mask[node_indices, svc_indices] & (self.service_omega[svc_indices].squeeze() == 0)
        cold_delays = task_cold_start * random.uniform(self.cold_start_delay_min, self.cold_start_delay_max)
        t_rem_raw = task_mean_deadlines - trans_delays - task_max_queue - cold_delays
        
        valid_mask = t_rem_raw >= 1e-4
        self.immediate_fails = (~valid_mask).sum().item()
        
        if valid_mask.any():
            vn, vs, vw, vt = node_indices[valid_mask], svc_indices[valid_mask], task_workloads[valid_mask], t_rem_raw[valid_mask]
            
            # CHUẨN FIFO: Luôn điền vào sau Task cuối cùng (vì queue đã được dồn hàng)
            for n, s, w, t in zip(vn, vs, vw, vt):
                # Đếm số task đang có để tìm vị trí tiếp theo
                num_active = (self.backlog_queue[n, s, :] > 0).sum().item()
                if num_active < self.max_K:
                    self.backlog_queue[n, s, int(num_active)] = w
                    self.deadline_queue[n, s, int(num_active)] = t
                else:
                    # Queue của node này đã đạt 100 -> Đánh fail luôn
                    self.immediate_fails += 1
            
            node_arrival_matrix.index_put_((vn, vs), vw, accumulate=True)
            
        return node_arrival_matrix, trans_energy_total

    def get_f_min_matrix(self):
        f_min_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        valid_tasks = self.backlog_queue > 0
        if not valid_tasks.any():
            return f_min_matrix
            
        cum_backlog = torch.cumsum(self.backlog_queue, dim=-1)
        req_matrix = cum_backlog / (self.deadline_queue + 1e-9)
        req_matrix = torch.where(valid_tasks, req_matrix, torch.zeros_like(req_matrix))
        f_min_matrix, _ = torch.max(req_matrix, dim=-1)
        return f_min_matrix

    def optimize_allocation(self, node_arrival_matrix, f_min_matrix):
        current_backlog_total = self.backlog_queue.sum(dim=-1)
        G = current_backlog_total * self.placement_matrix
        Z = self.lypa_coef * self.energy_coef * self.placement_matrix * node_arrival_matrix

        f_max = (self.resource_specs[:, 0:1] * self.placement_matrix)
        f_min = f_min_matrix.clamp(max=f_max)
        self.cpu_alloc_matrix = self.solver.solve(G, Z, f_min, f_max)

    def execute_and_collect_metrics(self, node_arrival_matrix, trans_energy_total):
        current_backlog_total = self.backlog_queue.sum(dim=-1)
        available_f_slot = self.cpu_alloc_matrix * self.slot_duration
        
        # 1. Deplete (FIFO)
        self.backlog_queue, actual_processed = ops.deplete_float_queue(self.backlog_queue, available_f_slot)
        
        # 2. Aging + Cleaning + DỒN HÀNG (Compaction)
        self.backlog_queue, self.deadline_queue, expired_count = ops.age_and_clean_float_queue(
            self.backlog_queue, self.deadline_queue, self.slot_duration
        )
        
        num_violations = expired_count + self.immediate_fails
        total_drift = ops.calculate_lyapunov_drift(current_backlog_total, node_arrival_matrix, actual_processed)
        comp_energy = ops.compute_batch_energy(
            self.cpu_alloc_matrix, actual_processed, self.energy_coef, 
            self.newly_placed_mask, self.service_omega, epsilon_cold=self.energy_cold_start
        )
        total_energy = comp_energy + trans_energy_total
        
        reward = -(total_drift + self.lypa_coef * total_energy + num_violations * 10.0 + self.placement_violations * 50.0)
        
        self.used_resources[:, 0] = self.cpu_alloc_matrix.sum(dim=1) 
        self.used_resources[:, 1] = (self.placement_matrix * (self.service_omega.squeeze() == 1)) @ self.service_size
        self.used_resources[:, 2] = (self.placement_matrix * (self.service_omega.squeeze() == 0)) @ self.service_size
        self.used_resources[:, 3] = comp_energy / (self.resource_specs[:, 3] + 1e-9)
        
        self.placement_violations = 0
        
        return {
            "reward": reward,
            "backlog": current_backlog_total,
            "energy": total_energy,
            "violations": num_violations
        }
