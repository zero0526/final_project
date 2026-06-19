import torch.nn as nn
import torch
import torch.nn.functional as F

from matrix_source.agents.base import MultiInstanceLinear, MultiInstanceRMSNorm


class MFNetwork(nn.Module):
    """Predict current mean field từ (state || prev_mf).
    Output: sigmoid -> [0,1]^mf_dim.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_sizes, num_instances: int = 1):
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
    """(task || svc || mf) → logits (Baseline Prior)"""

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
    """(task || svc || mf || proposal || hist || overload) → δlogits (Residual Correction)"""

    def __init__(self, task_state, service_state, mf_dim,
                 proposal_dim, action_dim,
                 hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        M = service_state // 2
        hist_dim = 2 * M  # histogram + overload
        in_dim = task_state + service_state + mf_dim + proposal_dim + hist_dim

        self.fc1 = MultiInstanceLinear(num_instances, in_dim, h1)
        self.norm1 = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2 = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2 = MultiInstanceRMSNorm(num_instances, h2)
        self.logits = MultiInstanceLinear(num_instances, h2, action_dim)

        # Khởi tạo bằng 0 để ở Phase 1 nó là hàm số 0 (không ảnh hưởng đến Proposal)
        nn.init.xavier_uniform_(self.delta_head.weight, gain=1.0)
        nn.init.normal_(self.delta_head.bias, std=0.1)

    def forward(self, task, svc, mf, current_logits, histogram, overload, indices=None):
        """current_logits: z_p (caller decides whether to .detach())"""
        x = torch.cat([
            task, svc, mf,
            current_logits,
            histogram, overload,
        ], dim=-1)
        x = F.silu(self.norm1(self.fc1(x, indices), indices))
        x = F.silu(self.norm2(self.fc2(x, indices), indices))
        return self.logits(x, indices)


class ResidualCritic(nn.Module):
    """(general_task || svc || mf) → V(s)

    MINIMALIST CRITIC: Không quan sát h* (histogram).
    Chỉ đánh giá giá trị trạng thái tổng thể của Node để tạo tín hiệu cho PPO.
    """

    def __init__(self, general_task_states, service_states,
                 mf_dim, hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        # ĐÃ BỎ hist_dim ra khỏi in_dim
        in_dim = general_task_states + service_states + mf_dim

        self.fc1 = MultiInstanceLinear(num_instances, in_dim, h1)
        self.norm1 = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2 = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2 = MultiInstanceRMSNorm(num_instances, h2)
        self.v = MultiInstanceLinear(num_instances, h2, 1)

    def forward(self, general_task, svc, mf, indices=None):
        # ĐÃ BỎ h_star khỏi forward
        x = torch.cat([general_task, svc, mf], dim=-1)
        x = F.silu(self.norm1(self.fc1(x, indices), indices))
        x = F.silu(self.norm2(self.fc2(x, indices), indices))
        return self.v(x, indices).squeeze(-1)