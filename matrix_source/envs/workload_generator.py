import torch

class MatrixWorkloadGenerator:
    """
    NHIỆM VỤ CỦA THÀNH PHẦN (WORKLOAD GENERATOR):
    1. Sinh ra các Task mới dưới dạng Ma Trận (Arrival Matrix).
       - Kích thước: (Num_Terminals x Num_Services).
    """
    def __init__(self, config, metadata):
        self.num_terminals = config.get('num_terminals', 1)
        self.num_services = len(metadata['service_ids']) if 'service_ids' in metadata else 0
        self.zipf_probs = metadata['zipf_probs']
        self.device = config.get('device', 'cpu')

    def generate_step(self):
        """
        Mỗi terminal sinh đúng 1 task tại mỗi step.
        Trả về (terminal_indices, svc_indices) của các task mới.
        """
        # 1. Toàn bộ terminals đều có task (chỉ số 0 đến num_terminals - 1)
        terminal_indices = torch.arange(self.num_terminals, device=self.device)
        
        # 2. Sample service cho từng terminal theo xác suất Zipf
        svc_indices = torch.multinomial(self.zipf_probs.to(self.device), self.num_terminals, replacement=True)
        
        return terminal_indices, svc_indices
