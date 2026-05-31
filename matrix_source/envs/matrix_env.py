import torch
import time
from collections import defaultdict
from matrix_source.envs.time_manager import TimeManager
from matrix_source.envs.matrix_physical_engine import MatrixPhysicalEngine
from matrix_source.envs.init_matrices import init_static_matrices, init_metadata_tensors

class MatrixSixGEnvironment:
    def __init__(self, config, device="cpu"):
        self.config = config
        self.device = device
        
        # 1. Initialize Matrices and Metadata (Automated)
        init_data = init_static_matrices(config, device=device)
        self.static_matrices = init_data
        self.terminals = init_data['terminals']
        
        metadata = init_metadata_tensors(config, device=device)
        self.metadata = metadata
        
        # 2. Initialize Engine
        self.engine = MatrixPhysicalEngine(config, self.static_matrices, metadata, device)
        # 3. Time Management
        self.time_manager = TimeManager(
            slot_duration=self.engine.slot_duration,
            timeframe_size=config.hyper_neural["TIME_SLOT_PER_TIMEFRAME"] ,
            max_steps=config.hyper_neural["NUMOF_TF_EP"]*config.hyper_neural["TIME_SLOT_PER_TIMEFRAME"]
        )
        self.prof = defaultdict(float)
        self.prof_counts = defaultdict(int)
        self.step_count = 0

    def reset(self):
        t0 = time.perf_counter()
        self.time_manager.reset()
        res = self.engine.reset()
        self.prof['reset'] += time.perf_counter() - t0
        return res

    def step_upper(self, placement_matrix):
        t0 = time.perf_counter()
        self.engine.set_upper_action(placement_matrix)
        self.prof['step_upper'] += time.perf_counter() - t0
        
    def collect_upper_metrics(self):
        t0 = time.perf_counter()
        metrics = self.engine.collect_upper_metrics()
        metrics["is_done"] = self.time_manager.current_step >= self.time_manager.max_steps
        self.prof['collect_upper'] += time.perf_counter() - t0
        return metrics

    def step_lower(self, terminal_indices, svc_indices, task_batch_sizes, node_indices, model_indices, task_deadlines, tasks_min_accuracy, is_discrete= False):
        t0 = time.perf_counter()
        # 1. Process Arrivals
        node_arrival_matrix, trans_energy_total, cold_delays, f_min_matrix = self.engine.process_arrivals(
            terminal_indices, svc_indices, node_indices, model_indices, task_batch_sizes, task_deadlines, tasks_min_accuracy
        )
        
        # 2. Solver Optimization
        self.engine.optimize_allocation(node_arrival_matrix, f_min_matrix)
        
        # 3. Execution & Metrics
        results = self.engine.execute_and_collect_metrics(node_arrival_matrix, trans_energy_total, cold_delays, is_discrete)
        
        # 4. Finalize Slot
        self.time_manager.tick()
        self.prof['step_lower'] += time.perf_counter() - t0
        self.step_count += 1
        
        # if self.step_count % 1000 == 0:
        #     print(f"\n<<< Env Profiling (Step {self.step_count}) >>>")
        #     for k, v in sorted(self.prof.items()):
        #         print(f"  {k:20s}: {v*1000/1000:8.3f} ms/call")
        #     self.prof.clear()

        return {
            "pre_reward":results["pre_reward"],
            "reward": results['reward'],
            "energy": results['energy'],
            "violations": results['violations'],
            "obs": results['obs'],
            "info": results["info"],
            "mean_field": results['mean_field'],
            "prev_actions": results['prev_actions'],
            "new_frame": self.time_manager.is_new_frame()
        }
