import numpy as np
from typing import List, Dict
from src.envs.entities.task_node import Task
from src.envs.entities.terminal_node import Terminal
from src.configs.configs import cfg, BaseConfig

class WorkloadGenerator:
    def __init__(
            self,
            terminals: List[Terminal],
            config: BaseConfig = cfg
    ):
        """
        Quản lý việc sinh tải cho toàn mạng.
        Nó chứa các tham số workload toàn cục và danh sách các Terminal.

        Args:
            terminals: Danh sách các đối tượng Terminal đã được khởi tạo
        """
        self.terminals = terminals

        self.arrival_rate = config.task_arrival_rate
        self.zipf_param = config.zipf_param
        self.fixed_batch_size = config.default_batch_size

        # Zipf (Pre-calculation)
        self.zipf_probs = self._calculate_zipf_probs(len(config.services), self.zipf_param)

        print(f"[WorkloadGen] Initialized with {len(terminals)} terminals.")
        print(f"[WorkloadGen] Fixed Batch Size per Task: {self.fixed_batch_size}")
        print(f"[WorkloadGen] Service Probabilities (Zipf s={self.zipf_param}): {self.zipf_probs}")

    def _calculate_zipf_probs(self, n: int, s: float) -> np.ndarray:
        """Tính vector xác suất Zipf cho n dịch vụ."""
        if n == 0: return np.array([])
        ranks = np.arange(1, n + 1)
        weights = 1.0 / np.power(ranks, s)
        probs = weights / np.sum(weights)
        return probs

    def step(self, current_time_slot: int) -> List[Task]:
        """
        Hàm chính được gọi tại mỗi bước mô phỏng (Simulation Step).
        Yêu cầu tất cả các Terminal sinh task.

        Returns:
            List các Task được sinh ra trong slot này.
        """
        generated_tasks = []

        for terminal in self.terminals:
            # Ủy quyền việc sinh task cho từng Terminal
            # Truyền các tham số workload toàn cục vào cho Terminal xử lý
            task = terminal.step_generate_task(
                current_time_slot=current_time_slot,
                batch_size=self.fixed_batch_size,
                zipf_probs=self.zipf_probs
            )

            if task:
                generated_tasks.append(task)

        return generated_tasks