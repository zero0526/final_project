import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from matrix_source.agents.base import MultiInstanceLinear, MultiInstanceRMSNorm



class MFNetwork(nn.Module):
    """Predict current mean field từ (state || prev_mf).
    Output: sigmoid -> [0,1]^mf_dim.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes, num_instances: int = 1):
        super().__init__()
        h1, h2 = hidden_sizes
        self.fc1  = MultiInstanceLinear(num_instances, input_dim, h1)
        self.norm = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2  = MultiInstanceLinear(num_instances, h1, h2)
        self.out  = MultiInstanceLinear(num_instances, h2, output_dim)

    def forward(self, x, indices=None):
        x = F.silu(self.norm(self.fc1(x, indices), indices))
        x = F.silu(self.fc2(x, indices))
        return torch.sigmoid(self.out(x, indices))


class ProposalActor(nn.Module):
    """(task || svc || mf) → logits"""
    def __init__(self, task_state, service_state, mf_dim, action_dim,
                 hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        in_dim = task_state + service_state + mf_dim
        self.fc1    = MultiInstanceLinear(num_instances, in_dim, h1)
        self.norm1  = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2    = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2  = MultiInstanceRMSNorm(num_instances, h2)
        self.logits = MultiInstanceLinear(num_instances, h2, action_dim)

    def forward(self, task, svc, mf, indices=None):
        x = torch.cat([task, svc, mf], dim=-1)
        x = F.silu(self.norm1(self.fc1(x, indices), indices))
        x = F.silu(self.norm2(self.fc2(x, indices), indices))
        return self.logits(x, indices)

    def evaluate(self, task, svc, mf, action, masks=None,
                 indices=None, residual_logits=None, exclude_zero=False):
        logits = self.forward(task, svc, mf, indices)
        if residual_logits is not None:
            logits = logits + residual_logits
        if masks is not None:
            logits = logits.masked_fill(masks == 0, -1e9)
        if exclude_zero and logits.shape[-1] > 1:
            logits[:, 0] = -1e9
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy()


class RefineActor(nn.Module):
    """(task || svc || mf || proposal || hist || overload) → δlogits"""
    def __init__(self, task_state, service_state, mf_dim,
                 proposal_dim, action_dim,
                 hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        M = service_state // 2  # service_state = 2*M
        hist_dim = 2 * M        # histogram + overload
        in_dim = task_state + service_state + mf_dim + proposal_dim + hist_dim

        self.fc1    = MultiInstanceLinear(num_instances, in_dim, h1)
        self.norm1  = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2    = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2  = MultiInstanceRMSNorm(num_instances, h2)
        self.logits = MultiInstanceLinear(num_instances, h2, action_dim)

        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def forward(self, task, svc, mf, proposal_logits,
                histogram, overload, indices=None):
        x = torch.cat([
            task, svc, mf,
            proposal_logits.detach(),
            histogram, overload,
        ], dim=-1)
        x = F.silu(self.norm1(self.fc1(x, indices), indices))
        x = F.silu(self.norm2(self.fc2(x, indices), indices))
        return self.logits(x, indices)


class ResidualCritic(nn.Module):
    """(general_task || svc || mf) → V(s)"""
    def __init__(self, general_task_states, service_states,
                 mf_dim, hist_dim, hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        in_dim = general_task_states + service_states + mf_dim + hist_dim
        self.fc1   = MultiInstanceLinear(num_instances, in_dim, h1)
        self.norm1 = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2   = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2 = MultiInstanceRMSNorm(num_instances, h2)
        self.v     = MultiInstanceLinear(num_instances, h2, 1)

    def forward(self, general_task, svc, mf, histogram, indices=None):
        x = torch.cat([general_task, svc, mf, histogram], dim=-1)
        x = F.silu(self.norm1(self.fc1(x, indices), indices))
        x = F.silu(self.norm2(self.fc2(x, indices), indices))
        return self.v(x, indices).squeeze(-1)
