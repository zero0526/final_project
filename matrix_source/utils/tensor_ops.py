import torch

import torch

def compute_batch_energy(f_alloc, processed_workload, energy_coef, newly_placed_mask, omega, epsilon_cold=0.5):
    """
    Tính toán năng lượng tiêu thụ (Computation Energy).
    newly_placed_mask: (M, S) - 1 nếu service mới được đặt/kích hoạt tại node
    """
    # 1. Dynamic Energy: E = epsilon * workload * f^2
    dynamic_energy = energy_coef * processed_workload * (f_alloc ** 2)
    
    # 2. Cold Start Energy (Chỉ cho Occasional services omega == 0 và mới được kích hoạt)
    cold_start_event = newly_placed_mask * (omega.T == 0)
    cold_start_energy = cold_start_event.float() * epsilon_cold
    
    return dynamic_energy.sum() + cold_start_energy.sum()

def compute_transmission_metrics(src_nodes, dst_nodes, delay_matrix, data_sizes, beta=1e-6):
    """
    src_nodes, dst_nodes: (Task_Count,)
    delay_matrix: (M, M)
    data_sizes: (Task_Count,)
    """
    # 1. Transmission Delay
    trans_delays = delay_matrix[src_nodes, dst_nodes]
    
    # 2. Transmission Energy: E = beta * delay * data_size
    trans_energy = beta * trans_delays * data_sizes
    
    return trans_delays, trans_energy

def calculate_qos_violations(backlog, cpu_alloc, deadlines, max_queue_delay, cold_start_mask, cold_start_delay=0.5):
    """
    Kiểm tra vi phạm deadline theo lô.
    cold_start_mask: (M, S) - 1 nếu task đang chịu cold start
    """
    # est_delay = queue_delay + computation_delay + cold_start_delay
    computation_delay = backlog / (cpu_alloc + 1e-6)
    est_delay = computation_delay + (cold_start_mask * cold_start_delay)
    
    violation_mask = est_delay > deadlines
    return violation_mask.float().sum()

def update_backlog(backlog, arrival, processed):
    """
    Q(t+1) = max(0, Q(t) - processed) + arrival
    """
    return torch.clamp(backlog - processed, min=0.0) + arrival

def calculate_lyapunov_drift(backlog, arrival, processed):
    """
    Tính Drift: sum( Q * (A - W) )
    """
    return (backlog * (arrival - processed)).sum()
