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

def deplete_float_queue(backlog, deadline, terminal_queue, age_queue, src_node_mapping, cpu_alloc, slot_duration):
    f = cpu_alloc.unsqueeze(-1) + 1e-9  # (M, S, 1)
    num_nodes = backlog.shape[0]

    # 1. Tính thời điểm hoàn thành dự kiến của từng task
    cum_backlog = torch.cumsum(backlog, dim=-1)
    time_to_finish = cum_backlog / f

    # 2. Xác định Task thành công (Hoàn thành trong slot này VÀ chưa hết hạn)
    success_mask = (backlog > 0) & (time_to_finish <= slot_duration) & (time_to_finish <= deadline)

    # 3. Tính Realized Delay (Thời gian từ lúc gửi -> lúc xong)
    # Delay = Tuổi lúc bắt đầu slot + Thời gian xử lý trong slot
    realized_delays = None
    if success_mask.any():
        realized_delays = age_queue[success_mask] + time_to_finish[success_mask]

    # 4. Cập nhật backlog: Xóa các task thành công
    new_backlog = backlog.clone()
    new_backlog[success_mask] = 0

    # 5. Xử lý task đang dở dang (pending) - Gọt bớt workload theo CPU còn dư
    pending_mask = (backlog > 0) & (~success_mask)
    before_backlog = cum_backlog - backlog
    available_for_pending = (f * slot_duration - before_backlog).clamp(min=0)
    actual_processed_pending = torch.min(available_for_pending, new_backlog)
    new_backlog = new_backlog - actual_processed_pending

    # 6. Tính toán lượng xử lý (Workload Delta)
    workload_delta = backlog - new_backlog

    # 7. Phân tách Local vs External
    node_ids = torch.arange(num_nodes, device=backlog.device).view(num_nodes, 1, 1)
    valid_terminal_mask = (terminal_queue >= 0)
    src_nodes = torch.zeros_like(terminal_queue)
    src_nodes[valid_terminal_mask] = src_node_mapping[terminal_queue[valid_terminal_mask]]

    is_local_mask = (src_nodes == node_ids) & valid_terminal_mask

    local_processed_total = (workload_delta * is_local_mask.float()).sum(dim=-1)
    actual_processed_total = workload_delta.sum(dim=-1)

    return new_backlog, actual_processed_total, local_processed_total, realized_delays


def age_and_clean_dual_queue(backlog, deadline, slot_duration, *aux_queues):
    """
    Trừ deadline, tăng tuổi (age) và dọn dẹp hàng đợi.
    """
    # 1. Giảm deadline tuyệt đối
    deadline = deadline - slot_duration

    # 2. Xác định các task THẬT SỰ vi phạm (Deadline <= 0 VÀ vẫn còn tồn tại trong queue)
    true_violation_mask = (deadline <= 0) & (backlog > 0)
    violation_counts = true_violation_mask.sum(dim=-1)  # (M, S)

    # Tính khối lượng công việc bị thất thoát do Expired
    failed_workload_total = torch.zeros_like(violation_counts, dtype=torch.float)
    if true_violation_mask.any():
        failed_workload_total = (backlog * true_violation_mask.float()).sum(dim=-1)

    # 3. Lấy thông tin các task bị fail TRƯỚC KHI XÓA
    failed_terminal_ids = None
    failed_svc_ids = None
    if len(aux_queues) > 0:
        term_queue = aux_queues[-1]
        failed_terminal_ids = term_queue[true_violation_mask].clone()

        violation_indices = torch.nonzero(true_violation_mask)
        failed_svc_ids = violation_indices[:, 1]

        valid_mask = (failed_terminal_ids >= 0)
        failed_terminal_ids = failed_terminal_ids[valid_mask]
        failed_svc_ids = failed_svc_ids[valid_mask]

    # 4. Xử lý Age Queue (Giả định age_queue là aux đầu tiên)
    # Tăng tuổi cho tất cả các task còn lại trong hàng đợi
    if len(aux_queues) > 0:
        age_q = aux_queues[0]
        age_q[backlog > 0] += slot_duration

    # 5. Xóa Task vi phạm và làm sạch dữ liệu cũ
    backlog[true_violation_mask] = 0
    empty_mask = (backlog <= 1e-7)
    backlog[empty_mask] = 0
    deadline[empty_mask] = 0

    # 6. DỒN HÀNG (Compaction)
    processed_aux = []
    mask = (backlog > 0).float()
    _, indices = torch.sort(mask, dim=-1, descending=True, stable=True)

    backlog = torch.gather(backlog, dim=-1, index=indices)
    deadline = torch.gather(deadline, dim=-1, index=indices)

    for q in aux_queues:
        if q is not None:
            q_gathered = torch.gather(q, dim=-1, index=indices)
            processed_aux.append(q_gathered)

    return backlog, deadline, processed_aux, violation_counts, failed_terminal_ids, failed_svc_ids, failed_workload_total


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
