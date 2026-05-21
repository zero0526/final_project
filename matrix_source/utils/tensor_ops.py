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
    trans_delays = delay_matrix[src_nodes, dst_nodes] * data_sizes
    trans_energy = beta * trans_delays
    return trans_delays, trans_energy

# ==========================================
# 2. HIGH-PRECISION FLOAT QUEUE OPERATIONS (3D)
# ==========================================

def deplete_float_queue(backlog, deadline, cpu_alloc, slot_duration):
    """
    Xử lý hàng đợi các tasks thành công trong timeslot và thât bại trong timeslot.
    trả về backlog mới đã clean success và fail, khối lượng công việc thực sự đã triển khai, số tasks fail
    backlog: (M, S, K)
    deadline: (M, S, K)
    cpu_alloc: (M, S) - Tần số CPU cấp cho cặp (Node, Service)
    """
    f = cpu_alloc.unsqueeze(-1) + 1e-9 # (M, S, 1)
    
    # 1. Tính thời điểm hoàn thành dự kiến của từng task (Time to Finish)
    cum_backlog = torch.cumsum(backlog, dim=-1)
    time_to_finish = cum_backlog / f # (M, S, K)
    
    # 2. Phân loại Task
    # Thành công trong slot: Xong trước deadline và trong tầm 0.1s
    success_mask = (backlog > 0) & (time_to_finish <= slot_duration) & (time_to_finish <= deadline)
    
    # Thất bại trong slot: Deadline hết trước khi kịp xong và deadline nằm trong 0.1s hiện tại
    violation_mask = (backlog > 0) & (deadline <= slot_duration) & (deadline < time_to_finish)
    
    # 3. Cập nhật khối lượng (Chỉ những task chưa Done và chưa Fail mới giữ lại backlog)
    processed_mask = success_mask | violation_mask
    
    # Tính toán lượng thực sự xử lý để tính năng lượng
    # Nếu task thành công, xử lý hết backlog. Nếu thất bại, xử lý một phần hoặc 0? 
    # Để đơn giản: ta tính năng lượng dựa trên cpu_alloc và slot_duration ở lớp trên.
    # Ở đây ta chỉ cập nhật backlog.
    new_backlog = backlog.clone()
    new_backlog[processed_mask] = 0
    
    # Những task chưa xong hẳn nhưng cũng chưa fail:
    pending_mask = (backlog > 0) & (~processed_mask)
    # Giảm bớt khối lượng cho task đang được xử lý dở dang (nếu có)
    # Lượng CPU còn dư sau khi xử lý các task trước đó
    before_backlog = cum_backlog - backlog
    available_for_pending = (f * slot_duration - before_backlog).clamp(min=0)
    actual_processed_pending = torch.min(available_for_pending, new_backlog)
    new_backlog = new_backlog - actual_processed_pending
    
    actual_processed_total = (backlog - new_backlog).sum(dim=-1)
    
    return new_backlog, actual_processed_total, violation_mask

def age_and_clean_dual_queue(backlog, deadline, aux_queue, in_slot_violation_mask, slot_duration):
    """
    Trừ deadline và dọn dẹp hàng đợi.
    in_slot_violation_mask: Mask các task đã fail ngay trong bước deplete
    """
    # 1. Giảm deadline tuyệt đối
    deadline = deadline - slot_duration
    
    # 2. Xác định tổng số vi phạm
    # Vi phạm cũ (từ bước deplete) + Vi phạm mới (do vừa trừ slot_duration xong bị âm)
    # Lưu ý: backlog > 0 đảm bảo ta không đếm lại các task đã xử lý xong
    total_violation_mask = in_slot_violation_mask | ((deadline <= 0) & (backlog > 0))
    violation_counts = total_violation_mask.sum(dim=-1) # (M, S)
    
    # 3. Xóa Task vi phạm và làm sạch dữ liệu cũ (Xử lý cả task đã xong từ bước deplete)
    backlog[total_violation_mask] = 0
    
    empty_mask = (backlog <= 1e-7)
    backlog[empty_mask] = 0
    deadline[empty_mask] = 0
    if aux_queue is not None:
        aux_queue[empty_mask] = 0

    # 4. DỒN HÀNG (Compaction)
    mask = (backlog > 0).float()
    _, indices = torch.sort(mask, dim=-1, descending=True, stable=True)
    
    backlog = torch.gather(backlog, dim=-1, index=indices)
    deadline = torch.gather(deadline, dim=-1, index=indices)
    if aux_queue is not None:
        aux_queue = torch.gather(aux_queue, dim=-1, index=indices)
    
    return backlog, deadline, aux_queue, violation_counts


# ==========================================
# 3. LYAPUNOV & ENERGY
# ==========================================

def calculate_lyapunov_drift(current_backlog, arrivals, processed):
    drift = current_backlog * (arrivals - processed)
    return drift.sum()

def transform2prob(phi: torch.Tensor)-> torch.Tensor:
    # phi: num_node x num_service 
    sum_workload_per_node= phi.sum(dim=-1, keepdim=True) + 1e-6
    return phi/sum_workload_per_node


def compute_batch_energy(f_alloc, processed, epsilon_comp, cold_delays, epsilon_cold=10.0):
    comp_energy = epsilon_comp * (f_alloc ** 2) * processed
    cold_energy = cold_delays*epsilon_cold
    return comp_energy.sum() + cold_energy.sum()

def get_running_counts(labels):
    """
    Computes a running count (offset) for each label in the input tensor.
    Example: [0, 1, 0, 0, 1, 2] -> [0, 0, 1, 2, 1, 0]
    """
    if labels.numel() == 0:
        return torch.empty_like(labels)
        
    sort_idx = torch.argsort(labels)
    sorted_labels = labels[sort_idx]
    
    # Identify where labels change using a shift-comparison
    transitions = torch.zeros_like(sorted_labels, dtype=torch.long)
    transitions[0] = 1
    transitions[1:] = (sorted_labels[1:] != sorted_labels[:-1]).long()
    
    # first_occurrences contains the first index of each new label in the sorted array
    first_occurrences_idx = torch.where(transitions == 1)[0]
    
    # counts per label
    counts = torch.diff(torch.cat([first_occurrences_idx, torch.tensor([len(labels)], device=labels.device)]))
    
    # repeat the first occurrence index to match the scale of sorted_labels
    repeated_first = first_occurrences_idx.repeat_interleave(counts)
    
    # calculate offsets in the sorted array
    sorted_offsets = torch.arange(len(labels), device=labels.device) - repeated_first
    
    # Map back to original order
    offsets = torch.zeros_like(sorted_offsets)
    offsets[sort_idx] = sorted_offsets
    return offsets
