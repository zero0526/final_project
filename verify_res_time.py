import torch
import sys
import os

# Add current directory to path to find matrix_source
sys.path.append(os.getcwd())

from matrix_source.envs.matrix_physical_engine import MatrixPhysicalEngine

class MockConfig:
    def __init__(self):
        self.hyper_neural = {
            "SLOT_DURATION": 0.1,
            "MAX_CYCLES": 10,
            "OMEGA_Q1": 1.0,
            "OMEGA_Q2": 1.0,
            "BETA_VIRTUAL_DRIFT": 1.0
        }
        self.max_queue_size = 10
        self.lypa_coef = 1.0
        self.energy_coef = 0.1
        self.cold_start_time = {"min": 0.1, "max": 0.5}
        self.cold_start_energy_coef = 10.0
        self.admm_rho = 1.0
        self.admm_max_iter = 10
        self.admm_tol = 1e-3
        self.transmission_coef = 1e-6

def test_resolution_time():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    num_nodes = 4
    num_services = 3
    num_terminals = 5
    max_K = 10
    
    config = MockConfig()
    
    # Transmission delay of 0.2s for node 0
    trans_matrix = torch.ones((num_nodes, num_nodes), device=device) * 0.2
    
    static_matrices = {
        'resource_matrix': torch.ones((num_nodes, 4), device=device),
        'transmission_delay_matrix': trans_matrix,
        'terminal_to_comp_node_map': torch.eye(num_terminals, num_nodes, device=device),
        'adj_matrix': torch.eye(num_nodes, device=device),
        'terminal_adj_matrix': torch.eye(num_terminals, device=device),
        'max_queue_delay': torch.ones((num_nodes, num_services), device=device) * 10.0,
        'terminal_to_node_map': torch.eye(num_terminals, num_nodes, device=device)
    }
    
    metadata = {
        'service_omega': torch.zeros((num_services, 1), device=device),
        'service_deadlines': torch.ones((num_services, 1), device=device) * 5.0,
        'service_input_size': torch.ones((num_services, 1), device=device),
        'model_workloads': torch.ones((num_services, 2), device=device),
        'model_accuracies': torch.ones((num_services, 2), device=device),
        'service_size': torch.ones(num_services, device=device) * 1024.0,
        'service_id_map': {i: i for i in range(num_services)},
        'service_size_val': 1.0
    }
    
    engine = MatrixPhysicalEngine(config, static_matrices, metadata, device)
    engine.placement_matrix.fill_(1.0)
    
    # Simulate arrivals
    terminal_indices = torch.tensor([0, 1], device=device)
    svc_indices = torch.tensor([0, 0], device=device)
    node_indices = torch.tensor([0, 0], device=device)
    model_indices = torch.tensor([0, 0], device=device)
    task_batch_sizes = torch.tensor([1.0, 1.0], device=device) # data_size = 1.0 * input_size(1.0) = 1.0
    task_deadlines = torch.tensor([5.0, 5.0], device=device)
    task_accuracies = torch.tensor([0.9, 0.9], device=device)
    
    # Transmission delay = delay_matrix[src, dst] * data_size = 0.2 * 1.0 = 0.2s
    
    print("Processing arrivals with Transmission Delay = 0.2s...")
    engine.process_arrivals(terminal_indices, svc_indices, node_indices, model_indices, task_batch_sizes, task_deadlines, task_accuracies)
    
    # Manually set CPU allocation
    engine.cpu_alloc_matrix.fill_(20.0) 
    
    print("Executing slot 1...")
    res = engine.execute_and_collect_metrics(torch.zeros((num_nodes, num_services), device=device), 0.0, torch.zeros(2, device=device), False)
    print(f"Slot 1 - Success QoS: {res['info']['success_qos'].sum().item()}")
    print(f"Slot 1 - Avg Resolution Time Success: {res['info']['avg_resolution_time_success']:.4f}")
    
    # Expected Slot 1: Trans(0.2) + Queue(0) + Proc(0.075) = 0.275s
    
    # Simulate more arrivals that stay in queue
    print("\nAdding tasks that stay in queue...")
    engine.cpu_alloc_matrix.zero_()
    engine.process_arrivals(terminal_indices, svc_indices, node_indices, model_indices, task_batch_sizes, task_deadlines, task_accuracies)
    
    print("Executing slot 2 (no processing)...")
    res = engine.execute_and_collect_metrics(torch.zeros((num_nodes, num_services), device=device), 0.0, torch.zeros(2, device=device), False)
    
    print("Executing slot 3 (processing tasks from slot 2)...")
    engine.cpu_alloc_matrix.fill_(20.0)
    res = engine.execute_and_collect_metrics(torch.zeros((num_nodes, num_services), device=device), 0.0, torch.zeros(2, device=device), False)
    print(f"Slot 3 - Success QoS: {res['info']['success_qos'].sum().item()}")
    print(f"Slot 3 - Avg Resolution Time Success: {res['info']['avg_resolution_time_success']:.4f}")
    
    # Expected Slot 3: Trans(0.2) + WaitSlot2(0.1) + Proc(0.075) = 0.375s
    
if __name__ == "__main__":
    test_resolution_time()
