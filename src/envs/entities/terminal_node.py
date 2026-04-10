import numpy as np
import random
from typing import Optional, Dict, List
from src.envs.entities.task_node import Task
from src.configs.configs import cfg, BaseConfig

class Terminal:
    def __init__(
            self,
            terminal_id: str,
            edge_id: str,
            arrival_rate: float = 1.0,
            default_batch_size: int = 10,
            config:BaseConfig=cfg
    ):
        """
        present Terminal attack to a nearest edge(Lower level agent).
        Args:
            terminal_id: (VD: "UE_0")
            edge_id:  Edge Node id is assigned terminal (src_i)
            arrival_rate: probability generate task in 1 time slot (0 -> 1.0)
            default_batch_size: num of batch for request (Paper: 20)
        """
        self.id = terminal_id
        self.edge_id = edge_id
        self.arrival_rate = arrival_rate
        self.default_batch_size = default_batch_size

        self.services= {s.get("id"):s for s in config.services.values()}
        self.current_task: Optional[Task] = None

        self.generated_tasks_count = 0

    def step_generate_task(
            self,
            abs_current_time: float,
            batch_size: int,
            zipf_probs: np.ndarray,
    ) -> Optional[Task]:
        """
        Args:
            current_time_slot: Thời điểm hiện tại (tau)
            arrival_rate: Xác suất sinh task cho terminal này trong slot này.
            batch_size: Số lượng batch cố định cho mỗi task.
            zipf_probs: Mảng xác suất Zipf đã tính sẵn (từ Generator).
            service_config_list: Danh sách cấu hình toàn bộ dịch vụ (từ Generator).
        """
        if np.random.random() > self.arrival_rate:
            self.current_task = None
            return None

        # 2. Chọn 1 Service duy nhất dựa trên phân phối Zipf
        num_services = len(self.services)
        selected_service_id_index = np.random.choice(
            np.arange(num_services),
            p=zipf_probs
        )
        svc_info = self.services[selected_service_id_index]
        task_acc = [m.get('accuracy', 0.0) for m in svc_info['models']]
        mu = np.mean(task_acc)
        sigma = 0.1
        min_acc_required = float(min(np.random.normal(mu, sigma), max(task_acc)))
        # 4. create Task
        task_id = f"T_{self.id}_{abs_current_time}"
        deadline = random.gauss(svc_info.get('mean_deadline'), svc_info.get('std_deadline'))
        new_task = Task(
            task_id=task_id,
            terminal_id=self.id,
            source_node_id=self.edge_id,
            batch_size=batch_size,
            deadline=abs_current_time + max(deadline, 1.0),
            min_accuracy=min_acc_required,
            created_at=abs_current_time,
            service_info=svc_info
        )

        return new_task

    def __repr__(self):
        return f"<Terminal {self.id} @ {self.edge_id}>"