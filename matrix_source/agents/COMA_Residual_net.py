import torch
import torch.nn as nn
import torch.nn.functional as F
from matrix_source.agents.base import MultiInstanceLinear, MultiInstanceRMSNorm


class MFNetwork(nn.Module):
    """
    Mean Field Network: Xử lý/Dự đoán trường trung bình (Mean Field).
    mf đại diện cho hành vi trung bình của các Edge khác (VD: tải kỳ vọng họ gửi đến các node).
    """
    def __init__(self, input_dim, output_dim, hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        self.fc1 = MultiInstanceLinear(num_instances, input_dim, h1)
        self.norm = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2 = MultiInstanceLinear(num_instances, h1, h2)
        self.out = MultiInstanceLinear(num_instances, h2, output_dim)

    def forward(self, x, indices=None):
        x = F.silu(self.norm(self.fc1(x, indices), indices))
        x = F.silu(self.fc2(x, indices))
        return torch.sigmoid(self.out(x, indices))

class ProposalActor(nn.Module):
    """
    Phase 1: Baseline Prior (Per-task).
    Chỉ nhìn vào state của 1 task cụ thể, service state và mean field.
    Hoàn toàn không quan tâm đến các task khác trong nhóm.
    """

    def __init__(self, task_state, service_state, mf_dim, action_dim,
                 hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        in_dim = task_state + service_state + mf_dim

        self.fc1 = MultiInstanceLinear(num_instances, in_dim, h1)
        self.norm1 = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2 = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2 = MultiInstanceRMSNorm(num_instances, h2)
        self.logits = MultiInstanceLinear(num_instances, h2, action_dim)

    def forward(self, task, svc, mf, indices=None):
        x = torch.cat([task, svc, mf], dim=-1)
        x = F.silu(self.norm1(self.fc1(x, indices), indices))
        x = F.silu(self.norm2(self.fc2(x, indices), indices))
        return self.logits(x, indices)


class RefineActor(nn.Module):
    def __init__(self, task_state, service_state, mf_dim,
                 action_dim, hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes

        # TÁCH RIÊNG group_load và mf_load
        in_dim = (task_state + service_state + mf_dim +
                  action_dim +  # proposal_logits
                  service_state//2 +  # group_load (tải do chính nhóm này gây ra)
                  mf_dim)  # mf_load (tải do các edge khác gây ra)

        self.fc1 = MultiInstanceLinear(num_instances, in_dim, h1)
        self.norm1 = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2 = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2 = MultiInstanceRMSNorm(num_instances, h2)
        self.delta_logits = MultiInstanceLinear(num_instances, h2, action_dim)

        nn.init.zeros_(self.delta_logits.weight)
        nn.init.zeros_(self.delta_logits.bias)

    def forward(self, task, svc, mf, proposal_logits, group_load, mf_load, indices=None):
        x = torch.cat([
            task, svc, mf,
            proposal_logits,
            group_load,  
            mf_load 
        ], dim=-1)

        x = F.silu(self.norm1(self.fc1(x, indices), indices))
        x = F.silu(self.norm2(self.fc2(x, indices), indices))
        return self.delta_logits(x, indices)


class CriticNetwork(nn.Module):
    """
    Centralized Q-Network cho Credit Assignment (COMA-inspired).
    Đánh giá Q(s, h_node, a) cho TỪNG task, nhưng có nhận thức được tải trọng của CẢ NHÓM (h_node).
    """

    def __init__(self, service_state, mf_dim,
                 action_dim, hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes

        # s = svc + mf. Thêm h_node + tải trọng task + mask + omega + data_size (mean, std, q1, q2, q3) + deadline (mean, std, q1, q2, q3) + batch_size
        M = service_state // 2
        in_dim = service_state + mf_dim + M + M + M + 1 + 5 + 5 + 1

        self.fc1 = MultiInstanceLinear(num_instances, in_dim, h1)
        self.norm1 = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2 = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2 = MultiInstanceRMSNorm(num_instances, h2)
        self.v_value = MultiInstanceLinear(num_instances, h2, 1)

    def forward(self, svc, mf, h_node, workload, mask, ds_metrics, deadline_metrics, omega, batch_size, indices=None):
        x = torch.cat([svc, mf, h_node, workload, mask, ds_metrics, deadline_metrics, omega, batch_size], dim=-1)
        x = F.silu(self.norm1(self.fc1(x, indices), indices))
        x = F.silu(self.norm2(self.fc2(x, indices), indices))
        return self.v_value(x, indices).squeeze(-1)
