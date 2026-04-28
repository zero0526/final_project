import torch

# ==========================================
# 1. TRANSMISSION METRICS
# ==========================================

def compute_transmission_metrics(src_nodes, dst_nodes, delay_matrix, data_sizes, beta=1e-6):
    """
    src_nodes, dst_nodes: (Task_Count,) indices
    delay_matrix: (M, M)
    data_sizes: (Task_Count,)
    """
    # 1. Transmission Delay
    trans_delays = delay_matrix[src_nodes, dst_nodes] * data_sizes
    
    # 2. Transmission Energy
    trans_energy = beta * trans_delays
    
    return trans_delays, trans_energy

# ==========================================
# 2. FLOAT DEADLINE QUEUE OPERATIONS (3D)
# ==========================================

def deplete_float_queue(backlog, available_f_slot):
    """
    Xử lý hàng đợi FIFO dựa trên năng lực tính toán (vectorized).
    backlog: (M, S, K) - Khối lượng GFLOPS của từng Task
    available_f_slot: (M, S) - Tổng GFLOPS có thể xử lý cho cặp (Node, Service)
    """
    # 1. Tính khối lượng công việc tích lũy (Cumulative Backlog)
    cum_backlog = torch.cumsum(backlog, dim=-1)
    
    # Khối lượng đứng trước mỗi task k
    before_processed = cum_backlog - backlog
    
    # 2. Xác định phần đã xử lý của mỗi task trong slot này
    start_processing_f = (available_f_slot.unsqueeze(-1) - before_processed).clamp(min=0)
    
    # Task chỉ được xử lý tối đa bằng khối lượng còn lại của nó
    actual_task_processed = torch.min(start_processing_f, backlog)
    
    # 3. Cập nhật backlog và tính tổng thực tế đã xử lý
    new_backlog = backlog - actual_task_processed
    actual_processed_total = actual_task_processed.sum(dim=-1)
    
    return new_backlog, actual_processed_total

def age_and_clean_float_queue(backlog, deadline, slot_duration):
    """
    Trừ deadline, phát hiện vi phạm QoS và DỒN HÀNG (Vectorized).
    backlog: (M, S, K)
    deadline: (M, S, K)
    """
    # 1. Giảm deadline
    deadline = deadline - slot_duration
    
    # 2. Phát hiện vi phạm
    violations_mask = (deadline <= 0) & (backlog > 0)
    violation_count = violations_mask.sum().item()
    
    # 3. Xóa dữ liệu task vi phạm hoặc đã xong
    backlog[violations_mask] = 0
    deadline[violations_mask] = 0
    
    # 4. CHUYÊN NGHIỆP: Dồn hàng (Compaction) về phía trước để giữ FIFO
    # Tạo mặt nạ task còn sống (1: sống, 0: trống)
    mask = (backlog > 0).float()
    
    # Sắp xếp mặt nạ Giảm dần để đẩy các số 1 lên đầu. 
    # Dùng stable=True để giữ nguyên thứ tự thời gian của các task sống.
    _, indices = torch.sort(mask, dim=-1, descending=True, stable=True)
    
    # Gom hàng lại theo indices mới
    backlog = torch.gather(backlog, dim=-1, index=indices)
    deadline = torch.gather(deadline, dim=-1, index=indices)
    
    return backlog, deadline, violation_count

# ==========================================
# 3. LYAPUNOV & ENERGY
# ==========================================

def calculate_lyapunov_drift(current_backlog, arrivals, processed):
    drift = current_backlog * (arrivals - processed)
    return drift.sum()

def compute_batch_energy(f_alloc, processed, epsilon_comp, newly_placed_mask, omega, epsilon_cold=10.0):
    comp_energy = epsilon_comp * (f_alloc ** 2) * processed
    cold_start_mask = newly_placed_mask & (omega.squeeze() == 0)
    cold_energy = cold_start_mask.float() * epsilon_cold
    return comp_energy.sum() + cold_energy.sum()
