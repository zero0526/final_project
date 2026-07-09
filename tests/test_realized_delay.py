import torch
import numpy as np
from matrix_source.utils import tensor_ops as ops
from matrix_source.visualize.aggregator import MetricsAggregator

def test_sensor_ops_with_age():
    # Shape parameters
    num_nodes = 3
    num_services = 2
    max_K = 5
    slot_duration = 1.0

    # Backlog queue with 1 task
    backlog = torch.zeros((num_nodes, num_services, max_K))
    backlog[0, 0, 0] = 10.0  # workload 10
    
    # Deadline queue
    deadline = torch.zeros((num_nodes, num_services, max_K))
    deadline[0, 0, 0] = 5.0

    # Terminal queue
    terminal_queue = torch.zeros((num_nodes, num_services, max_K), dtype=torch.long) - 1
    terminal_queue[0, 0, 0] = 0

    # Age queue (initial transmission delay / age)
    age_queue = torch.zeros((num_nodes, num_services, max_K))
    age_queue[0, 0, 0] = 1.5

    src_node_mapping = torch.tensor([0, 1, 2], dtype=torch.long)

    # CPU alloc (capacity 20.0 per service per node)
    cpu_alloc = torch.zeros((num_nodes, num_services))
    cpu_alloc[0, 0] = 20.0

    # 1. Test deplete_float_queue
    new_backlog, actual_processed, local_processed, realized_delays = ops.deplete_float_queue(
        backlog, deadline, terminal_queue, age_queue, src_node_mapping, cpu_alloc, slot_duration
    )

    print("--- Test deplete_float_queue ---")
    print(f"new_backlog sum: {new_backlog.sum().item()}")
    print(f"actual_processed val: {actual_processed[0, 0].item()}")
    print(f"local_processed val: {local_processed[0, 0].item()}")
    print(f"realized_delays: {realized_delays}")

    assert new_backlog[0, 0, 0].item() == 0, "Task should be completed"
    assert actual_processed[0, 0].item() == 10.0
    assert local_processed[0, 0].item() == 10.0
    assert realized_delays is not None
    # time_to_finish = cum_backlog / f = 10.0 / 20.0 = 0.5
    # realized_delay = age + time_to_finish = 1.5 + 0.5 = 2.0
    assert abs(realized_delays[0].item() - 2.0) < 1e-4, "Realized delay should be 2.0"

    print("Success: deplete_float_queue works properly with realized delays!")

    # 2. Test age_and_clean_dual_queue
    backlog_age = torch.zeros((num_nodes, num_services, max_K))
    backlog_age[1, 1, 0] = 5.0
    
    deadline_age = torch.zeros((num_nodes, num_services, max_K))
    deadline_age[1, 1, 0] = 0.5  # Will expire afterslot_duration (1.0) because deadline - slot_duration <= 0

    age_q = torch.zeros((num_nodes, num_services, max_K))
    age_q[1, 1, 0] = 2.0

    f_min = torch.zeros((num_nodes, num_services, max_K))
    term_q = torch.zeros((num_nodes, num_services, max_K), dtype=torch.long) - 1
    term_q[1, 1, 0] = 1

    new_backlog, new_deadline, processed_aux, violation_counts, failed_terminal_ids, failed_svc_ids, failed_workload_total = ops.age_and_clean_dual_queue(
        backlog_age, deadline_age, slot_duration, age_q, f_min, term_q
    )

    print("--- Test age_and_clean_dual_queue ---")
    print(f"violation_counts: {violation_counts[1, 1].item()}")
    print(f"failed_workload_total: {failed_workload_total[1, 1].item()}")
    
    assert violation_counts[1, 1].item() == 1, "There should be 1 violation"
    assert failed_workload_total[1, 1].item() == 5.0, "Failed workload should capture backlog"
    assert failed_svc_ids[0].item() == 1
    
    print("Success: age_and_clean_dual_queue works properly and returns failed_workload_total!")

def test_metrics_aggregator():
    agg = MetricsAggregator(name="test_agg")
    
    # Pack output
    step_output = {
        "reward": -5.0,
        "energy": 2.0,
        "violations": 1.0,
        "obs": {
            "total_drift": 0.5,
            "virtual_drift": 0.2
        },
        "info": {
            "success_qos": torch.tensor([5.0, 1.0]),
            "violate_qos": torch.tensor([1.0, 0.0]),
            "remaining": 2,
            "num_tasks": 10,
            "immediate_fails": 1,
            "expired_count": 0,
            "realized_delay": torch.tensor([1.5, 2.5]),
            "ttl_penalty_val": 4.0
        }
    }
    
    agg.add_lower(step_output)
    
    # Store history
    agg.store_history()
    
    # Verify the accumulated metrics in history
    assert "realized_delay" in agg.history
    assert "avg_ttl_penalty" in agg.history
    assert "avg_virtual_drift" in agg.history
    
    print(f"Aggregator history realized_delay: {agg.history['realized_delay']}")
    print(f"Aggregator history avg_ttl_penalty: {agg.history['avg_ttl_penalty']}")
    print(f"Aggregator history avg_virtual_drift: {agg.history['avg_virtual_drift']}")
    
    print("Success: MetricsAggregator integration is verified!")

if __name__ == "__main__":
    test_sensor_ops_with_age()
    test_metrics_aggregator()
