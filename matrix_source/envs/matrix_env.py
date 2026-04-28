import torch
from matrix_source.envs.time_manager import TimeManager
from matrix_source.envs.matrix_physical_engine import MatrixPhysicalEngine

class MatrixSixGEnvironment:
    def __init__(self, config, static_matrices, metadata, device="cpu"):
        self.config = config
        self.device = device
        
        # Engine handles all physical states and optimizations
        self.engine = MatrixPhysicalEngine(config, static_matrices, metadata, device)
        
        # Time Management
        self.time_manager = TimeManager(slot_duration=self.engine.slot_duration)

    def reset(self):
        self.engine.reset()
        self.time_manager.reset()

    def step_upper(self, placement_matrix):
        """
        Upper-level decision: Update service placement.
        """
        self.engine.update_placement(placement_matrix)

    def step_lower(self, terminal_indices, svc_indices, node_indices, model_indices):
        """
        Execute the lower-level step via the physical engine.
        """
        # 1. Process Arrivals
        node_arrival_matrix, trans_energy_total = self.engine.process_arrivals(
            terminal_indices, svc_indices, node_indices, model_indices
        )
        
        # 2. Solver Optimization
        f_min_matrix = self.engine.get_f_min_matrix()
        self.engine.optimize_allocation(node_arrival_matrix, f_min_matrix)
        
        # 3. Execution & Metrics
        results = self.engine.execute_and_collect_metrics(node_arrival_matrix, trans_energy_total)
        
        # 4. Finalize Slot
        self.time_manager.tick()
        
        return {
            "reward": results['reward'].item(),
            "backlog": results['backlog'].clone(),
            "energy": results['energy'].item(),
            "violations": results['violations'].item(),
            "new_frame": self.time_manager.is_new_frame()
        }

    def get_observation(self, node_indices=None):
        """
        Returns condensed observations from the Engine's state.
        Now compatible with 3D Float Queue.
        """
        # (Nodes x Services)
        backlog_2d = self.engine.backlog_queue.sum(dim=-1)
        
        # Observable channels: (Backlog, CPU_Alloc, Placement, CPU_Util, RAM_Util, HDD_Util, Power_Util)
        obs = torch.stack([
            backlog_2d,
            self.engine.cpu_alloc_matrix,
            self.engine.placement_matrix,
            self.engine.used_resources[:, 0].unsqueeze(1).expand(-1, self.engine.num_services),
            self.engine.used_resources[:, 1].unsqueeze(1).expand(-1, self.engine.num_services),
            self.engine.used_resources[:, 2].unsqueeze(1).expand(-1, self.engine.num_services),
            self.engine.used_resources[:, 3].unsqueeze(1).expand(-1, self.engine.num_services)
        ], dim=-1) # (M, S, 7)
        
        if node_indices is not None:
            return obs[node_indices]
        return obs
