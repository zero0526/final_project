def calc_computation_energy(epsilon_c, f_allocated, workload_total, omega, epsilon_cold, t_cold):
    """
    Args:
        epsilon_c: (epsilon_v^c) - 5e-10
        f_allocated:  (f_v,s) - GFLOPS
        workload_total: (batches * GFLOPS/batch) -  (d/D)*F
        omega: (1: continuous, 0: occasional)
        epsilon_cold: (epsilon_v_cold)
        t_cold: cold start (t_s,v_cold)
    """
    # 1.(Dynamic Energy)
    # E_dynamic = epsilon_v^c * Workload * f^2
    e_dynamic = epsilon_c * workload_total * (f_allocated ** 2)

    # 2.(Cold Start Energy)
    # omega = 0 (Occasional Service)
    e_cold = (1 - omega) * epsilon_cold * t_cold

    return e_dynamic + e_cold

def update_backlog(current_backlog, processed_workload, arrival_workload):
    """
    Tính Q(t+1) dựa trên Q(t), W(t) và A(t).

    Args:
        current_backlog (float): Q(t) - Backlog hiện tại (GFLOPS).
        processed_workload (float): W(t) - Khối lượng đã xử lý (f * duration).
        arrival_workload (float): A(t) - Khối lượng task mới nhận vào.

    Returns:
        float: Q(t+1) - Backlog tiếp theo.
    """
    # Công thức: Q_next = max(Q_curr - W, 0) + A
    remaining = max(0.0, current_backlog - processed_workload)
    next_backlog = remaining + arrival_workload

    return next_backlog