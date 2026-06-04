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
        # Dùng sigmoid nếu mf là tỷ lệ tải, hoặc softmax nếu là phân phối xác suất offload
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
                  2 +  # confidence_metrics (entropy, margin)
                  service_state//2 +  # group_load (tải do chính nhóm này gây ra)
                  action_dim)  # mf_load (tải do các edge khác gây ra)

        self.fc1 = MultiInstanceLinear(num_instances, in_dim, h1)
        self.norm1 = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2 = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2 = MultiInstanceRMSNorm(num_instances, h2)
        self.delta_logits = MultiInstanceLinear(num_instances, h2, action_dim)

        nn.init.zeros_(self.delta_logits.weight)
        nn.init.zeros_(self.delta_logits.bias)

    def forward(self, task, svc, mf, proposal_logits,
                confidence_metrics, group_load, mf_load, indices=None):
        x = torch.cat([
            task, svc, mf,
            proposal_logits,
            confidence_metrics,
            group_load,  # Đưa riêng
            mf_load  # Đưa riêng
        ], dim=-1)

        x = F.silu(self.norm1(self.fc1(x, indices), indices))
        x = F.silu(self.norm2(self.fc2(x, indices), indices))
        return self.delta_logits(x, indices)


class COMAQNetwork(nn.Module):
    """
    Centralized Q-Network cho Credit Assignment (COMA-inspired).
    Đánh giá Q(s, h_node, a) cho TỪNG task, nhưng có nhận thức được tải trọng của CẢ NHÓM (h_node).
    """

    def __init__(self, general_task_state, service_state, mf_dim,
                 action_dim, hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes

        # s = task + svc + mf. Thêm h_node để cung cấp ngữ cảnh tải trọng nhóm.
        in_dim = general_task_state + service_state + mf_dim + service_state//2

        self.fc1 = MultiInstanceLinear(num_instances, in_dim, h1)
        self.norm1 = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2 = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2 = MultiInstanceRMSNorm(num_instances, h2)
        self.q_values = MultiInstanceLinear(num_instances, h2, action_dim)

    def forward(self, task, svc, mf, h_node, indices=None):
        x = torch.cat([task, svc, mf, h_node], dim=-1)
        x = F.silu(self.norm1(self.fc1(x, indices), indices))
        x = F.silu(self.norm2(self.fc2(x, indices), indices))
        return self.q_values(x, indices)


def build_group_context(proposal_logits_group, masks_exp, mf_load_contribution, temperature=0.5):
    """
    Trả về các thành phần tách biệt để mạng neural tự học cách tương tác.
    """
    # 1. Confidence Metrics
    logits_masked = proposal_logits_group.clone()
    if masks_exp is not None:
        logits_masked = logits_masked.masked_fill(masks_exp == 0, -1e9)
    probs = F.softmax(logits_masked, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1, keepdim=True)
    top2_probs, _ = torch.topk(probs, k=2, dim=-1)
    margin = (top2_probs[:, 0] - top2_probs[:, 1]).unsqueeze(-1)
    confidence_metrics = torch.cat([entropy, margin], dim=-1)

    # 2. Group Load (Tải do chính nhóm này gây ra)
    sharpened_probs = F.softmax(proposal_logits_group / temperature, dim=-1)
    group_load = sharpened_probs.sum(dim=0, keepdim=True)  # (1, action_dim)

    # 3. Broadcast để mỗi task đều thấy được bức tranh toàn cảnh
    N = proposal_logits_group.shape[0]
    group_load_expanded = group_load.expand(N, -1)  # (N, action_dim)
    mf_load_expanded = mf_load_contribution.unsqueeze(0).expand(N, -1)  # (N, action_dim)

    return confidence_metrics, group_load_expanded, mf_load_expanded