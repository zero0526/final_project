from typing import List, Dict, Any

class Task:
    def __init__(
            self,
            task_id,
            terminal_id,
            source_node_id,  # attack to edge
            batch_size: int,
            deadline: float,
            min_accuracy: float,
            created_at: float,
            service_info: dict,
    ):
        self.id = task_id
        self.terminal_id = terminal_id
        self.source_node_id = source_node_id
        self.service_id = service_info.get("id")
        self.models:List[Dict[str, Any]] = service_info.get("models")
        self.batch_size = batch_size

        # input_data_size: MB per item (Table II)
        unit_size_mb = service_info.get('input_data_size', 0.0)

        self.total_data_size_mb = unit_size_mb * self.batch_size

        # --- QoS Requirements (SLA) ---
        self.deadline = deadline  # t^{th}_{i,s} (giây)
        self.min_accuracy = min_accuracy  # Acc^{th}_{i,s} (0.0 - 1.0)

        # Service Type: 1 (Continuous - Always On), 0 (Occasional - Cold Start)
        self.omega = service_info.get('omega', 1)

        # --- Lower assignment ---
        self.assigned_node_id = None  # Nút đích (v) - Offloading
        self.selected_model_idx = None  # Model (b) - Model Selection
        self.model: Dict[str, Any]= None
        self.required_workload_gflops = 0.0  # (F: gflop)
        self.remaining_workload_gflops = -1.0

        # --- Reward/Log) ---
        self.created_at = created_at #second
        self.finished_at = None

        # trace task
        self.transmission_delay = 0.0
        self.queue_delay = 0.0
        self.computation_delay = 0.0
        self.cold_start_delay = 0.0

        self.transmission_energy = 0.0
        self.computation_energy = 0.0
        self.cold_start_energy = 0.0

    def assign_schedule(self, node_id, model_idx):
        """
        Gán quyết định từ Agent (Lower-level) cho Task.

        Args:
            node_id: computing node is assigned task (v).
            model_idx: Index của model được chọn trong danh sách model của service này giữ nguyên order trong file config service
            unit_workload: Khối lượng tính toán (GFLOPS) để xử lý 1 đơn vị dữ liệu (1 item).
        """
        self.assigned_node_id = node_id
        self.selected_model_idx = model_idx
        self.model = self.models[model_idx]
        # eq12: Workload = unit_workload * Batch_size
        total_workload= self.model.get("workload") * self.batch_size
        self.required_workload_gflops = total_workload
        self.remaining_workload_gflops = total_workload

    @property
    def total_delay(self):
        """total time to complete task."""
        return (self.transmission_delay +
                self.queue_delay +
                self.computation_delay +
                self.cold_start_delay)

    @property
    def is_successful(self):
        """Check SLA"""
        if self.finished_at is None:
            return False
        if not self.model:
            return False
        acc:float = self.model.get("accuracy")
        return self.total_delay <= self.deadline and acc >= self.min_accuracy

    @property
    def is_assigned(self):
        return self.assigned_node_id is not None

    def trace_task(self, attr: Dict[str, Any]):
        self.finished_at = attr.get("finished_at", None)

        self.transmission_delay = attr.get("transmission_delay", 0.0)
        self.queue_delay = attr.get("queue_delay", 0.0)
        self.computation_delay = attr.get("computation_delay", 0.0)
        self.cold_start_delay = attr.get("cold_start_delay", 0.0)


        self.transmission_energy = attr.get("transmission_energy", 0.0)
        self.computation_energy = attr.get("computation_energy", 0.0)
        self.cold_start_energy = attr.get("cold_start_energy", 0.0)

    @property
    def time_comsume(self):
        return self.transmission_delay + self.queue_delay + self.computation_delay +self.cold_start_delay

    @property
    def energy_comsume(self):
        return self.transmission_energy + self.computation_energy + self.cold_start_energy

    def __repr__(self):
        status = "DONE" if self.finished_at else "PENDING"
        return (f"<Task {self.id} | Src:{self.source_node_id} | Svc:{self.service_id} | deadline: {self.deadline} "
                f"Data:{self.total_data_size_mb:.2f}MB | Workload:{self.required_workload_gflops:.2f}G | "
                f"Status:{status}>")