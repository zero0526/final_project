import torch
import torch.nn as nn
from .workload_generator import MatrixWorkloadGenerator
from .time_manager import MatrixTimeManager
from ..utils import tensor_ops as ops

from ..models.resource_solver import BatchResourceSolver

class MatrixSixGEnvironment:
    # ... (init remains same)
    def __init__(self, config, static_matrices, metadata):
        """
        config: Dictionary containing hyperparameters (lypa_coef, energy_coef, etc.)
        static_matrices: Output from init_static_matrices (delay_matrix, resource_matrix, mapping)
        metadata: Output from init_metadata_tensors (workloads, deadlines, omega, etc.)
        """
        self.config = config
        self.device = config.get('device', 'cpu')
        
        # Static Matrices (Pre-calculated)
        self.resource_specs = static_matrices['resource_matrix'].to(self.device).float() # (M, 3)
        self.delay_matrix = static_matrices['delay_matrix'].to(self.device).float() # (M, M)
        self.terminal_to_node_map = static_matrices['terminal_to_node_map'].to(self.device).float() # (T, M)
        self.max_queue_delay = static_matrices['max_queue_delay'].to(self.device).float() # (M, S)
        
        # Service Metadata
        self.model_workloads = metadata['model_workloads'].to(self.device).float() # (S, Max_Models)
        self.model_accuracies = metadata['model_accuracies'].to(self.device).float() # (S, Max_Models)
        self.service_deadlines = metadata['service_deadlines'].to(self.device).float() # (S, 5)
        self.service_omega = metadata['service_omega'].to(self.device).float() # (S, 1)
        self.service_input_size = metadata['service_input_size'].to(self.device).float() # (S, 1)
        self.service_resource_req = metadata['service_resource_specs'].to(self.device).float() # (S, 2)
        
        self.num_nodes = self.resource_specs.shape[0]
        self.num_services = self.service_omega.shape[0]
        
        # Dynamic State Matrices
        self.backlog_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.cpu_alloc_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.placement_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.prev_placement_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.newly_placed_mask = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.mean_field_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.used_resources = torch.zeros((self.num_nodes, 2), device=self.device) # [RAM, HDD]
        
        # Components
        self.workload_gen = MatrixWorkloadGenerator(config, metadata)
        self.time_manager = MatrixTimeManager(
            config.get('slot_duration', 0.1),
            config.get('timeframe_size', 10),
            config.get('max_steps', 1000)
        )
        
        # Solver (Fixed CPU capacity per node)
        f_max_node = self.resource_specs[0, 0].item()
        self.solver = BatchResourceSolver(f_max_node=f_max_node)
        
        # Constants
        self.lypa_coef = config.get('lypa_coef', 1e-7)
        self.energy_coef = config.get('energy_coef', 5e-10)
        self.cold_start_delay = config.get('cold_start_delay', 0.5)
        self.energy_cold_start = config.get('energy_cold_start', 0.5)

    def reset(self):
        self.backlog_matrix.zero_()
        self.cpu_alloc_matrix.zero_()
        self.placement_matrix.zero_()
        self.prev_placement_matrix.zero_()
        self.newly_placed_mask.zero_()
        self.mean_field_matrix.zero_()
        self.used_resources.zero_()
        self.time_manager.reset()

    def step_lower(self, terminal_indices, svc_indices, node_indices, model_indices):
        """
        terminal_indices, svc_indices: Task info from WorkloadGenerator
        node_indices, model_indices: Decisions from Lower-level Agent
        """
        # newly_placed_mask is maintained from step_upper for the entire timeframe

        # 1. Determine Source Nodes and Transmission Metrics
        num_tasks = len(svc_indices)
        trans_energy_total = 0.0
        node_arrival_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        
        if num_tasks > 0:
            src_node_indices = torch.argmax(self.terminal_to_node_map[terminal_indices], dim=1)
            task_data_sizes = self.service_input_size[svc_indices].squeeze()
            
            # Use utility for transmission
            trans_delays, trans_energy_tasks = ops.compute_transmission_metrics(
                src_node_indices, node_indices, self.delay_matrix, task_data_sizes, 
                beta=self.config.get('trans_energy_beta', 1e-6)
            )
            trans_energy_total = trans_energy_tasks.sum()

            # 2. Add workload (Assignment phase)
            task_workloads = self.model_workloads[svc_indices, model_indices]
            node_arrival_matrix.index_put_((node_indices, svc_indices), task_workloads, accumulate=True)
            self.backlog_matrix.index_put_((node_indices, svc_indices), task_workloads, accumulate=True)

            # 3. Refined f_min Calculation (With Cold Start Delay)
            task_mean_deadlines = self.service_deadlines[svc_indices, 2] 
            task_max_queue = self.max_queue_delay[node_indices, svc_indices]
            
            # Task-specific cold start delay if service is newly placed and omega=0
            # newly_placed_mask is persistent for the timeframe
            task_cold_start = self.newly_placed_mask[node_indices, svc_indices] & (self.service_omega[svc_indices].squeeze() == 0)
            
            t_rem = (task_mean_deadlines - trans_delays - task_max_queue - (task_cold_start * self.cold_start_delay)).clamp(min=0.01)
            f_req_per_task = task_workloads / t_rem
            
            f_min_matrix = torch.zeros_like(self.backlog_matrix)
            for i in range(num_tasks):
                f_min_matrix[node_indices[i], svc_indices[i]] = torch.max(f_min_matrix[node_indices[i], svc_indices[i]], f_req_per_task[i])
        else:
            f_min_matrix = torch.zeros_like(self.backlog_matrix)

        # 4. OPTIMIZE RESOURCE ALLOCATION (ADMM)
        G = self.backlog_matrix * self.placement_matrix
        Z = self.lypa_coef * self.energy_coef * self.placement_matrix * node_arrival_matrix

        f_max = (self.resource_specs[:, 0:1] * self.placement_matrix)
        f_min = f_min_matrix.clamp(max=f_max)
        self.cpu_alloc_matrix = self.solver.solve(G, Z, f_min, f_max)

        # 5. Execution & Metrics
        slot_duration = self.time_manager.slot_duration
        processed_workload = torch.min(self.backlog_matrix, self.cpu_alloc_matrix * slot_duration)
        
        # QoS & Drift
        mean_deadlines = self.service_deadlines[:, 2].unsqueeze(0)
        num_violations = ops.calculate_qos_violations(
            self.backlog_matrix, self.cpu_alloc_matrix, mean_deadlines, self.max_queue_delay,
            cold_start_mask=self.newly_placed_mask, cold_start_delay=self.cold_start_delay
        )
        total_drift = ops.calculate_lyapunov_drift(self.backlog_matrix, node_arrival_matrix, processed_workload)

        self.backlog_matrix = ops.update_backlog(self.backlog_matrix, torch.zeros_like(node_arrival_matrix), processed_workload)
        
        comp_energy = ops.compute_batch_energy(
            self.cpu_alloc_matrix, processed_workload, self.energy_coef, 
            self.newly_placed_mask, self.service_omega, epsilon_cold=self.energy_cold_start
        )
        total_energy = comp_energy + trans_energy_total
        
        f1 = total_drift + self.lypa_coef * total_energy
        reward = -(f1 + num_violations * 10.0)
        
        self.time_manager.tick()
        
        return {
            "reward": reward.item(),
            "backlog": self.backlog_matrix.clone(),
            "energy": total_energy.item(),
            "violations": num_violations.item(),
            "new_frame": self.time_manager.is_new_frame()
        }

    def step_upper(self, placement_actions):
        """
        placement_actions: (M, S) - Binary matrix of desired placement.
        """
        # 1. Store current as previous
        self.prev_placement_matrix = self.placement_matrix.clone()
        
        # 2. Reset resource usage
        self.used_resources.zero_()
        
        # 3. Check resource constraints (Vectorized)
        total_req = torch.matmul(placement_actions.to(self.device), self.service_resource_req)
        capacities = self.resource_specs[:, 1:3]
        can_host = (total_req <= capacities).all(dim=1)
        
        # 4. Update placement matrix
        self.placement_matrix = placement_actions * can_host.unsqueeze(1)
        self.used_resources = total_req * can_host.unsqueeze(1)
        
        # 5. Maintain newly_placed_mask for the duration of this timeframe
        self.newly_placed_mask = (self.placement_matrix == 1) & (self.prev_placement_matrix == 0)
        
        return {
            "placement": self.placement_matrix.clone(),
            "used_resources": self.used_resources.clone()
        }

    def get_observation(self, node_indices=None):
        """
        Returns the current state in a format suitable for the RL agents.
        """
        # Common observation: (Backlog, CPU_Alloc, Placement, Mean_Field)
        obs = torch.stack([
            self.backlog_matrix,
            self.cpu_alloc_matrix,
            self.placement_matrix,
            self.mean_field_matrix
        ], dim=-1) # (M, S, 4)
        
        if node_indices is not None:
            return obs[node_indices]
        return obs
