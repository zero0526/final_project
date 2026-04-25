import torch

import torch

def compute_batch_energy(f_alloc, processed_workload, energy_coef, placement_matrix, omega, epsilon_cold=0.5):
    """
    Tính toán năng lượng tiêu thụ (Computation Energy).
    f_alloc: (M, S)
    processed_workload: (M, S)
    energy_coef: scalar or (M, 1)
    placement_matrix: (M, S) - Trạng thái đặt chỗ hiện tại
    omega: (S, 1) - Loại service (1: Continuous, 0: Occasional)
    """
    # 1. Dynamic Energy: E = epsilon * workload * f^2
    dynamic_energy = energy_coef * processed_workload * (f_alloc ** 2)
    
    # 2. Cold Start Energy (Chỉ cho Occasional services omega == 0 và mới được đặt f > 0)
    # Giả định placement_matrix đã được cập nhật. Nếu placement == 1 nhưng omega == 0, 
    # ta có thể coi là tốn thêm năng lượng khởi chạy (đơn giản hóa).
    cold_start_mask = (placement_matrix == 1) * (omega.T == 0)
    cold_start_energy = cold_start_mask.float() * epsilon_cold
    
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

def calculate_qos_violations(backlog, cpu_alloc, deadlines, max_queue_delay):
    """
    Kiểm tra vi phạm deadline theo lô.
    backlog: (M, S)
    cpu_alloc: (M, S)
    deadlines: (1, S) or (M, S)
    max_queue_delay: (M, S)
    """
    # Ước lượng delay hiện tại: queue_delay + comp_delay
    # queue_delay = backlog / (cpu_alloc + 1e-6)
    est_delay = backlog / (cpu_alloc + 1e-6)
    
    # Vi phạm nếu est_delay > (deadline - transmission - queue_max) 
    # Hoặc đơn giản là vượt quá ngưỡng deadline
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
