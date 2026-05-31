import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from matrix_source.agents.base import (
    MultiInstanceLinear, MultiInstanceRMSNorm
)


class MultiInstanceActor(nn.Module):
    
    def __init__(self, state_dim, mf_dim, action_dim, hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        self.num_instances = num_instances

        self.fc1 = MultiInstanceLinear(num_instances, state_dim + mf_dim, h1)
        self.norm1 = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2 = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2 = MultiInstanceRMSNorm(num_instances, h2)
        self.actor_logits = MultiInstanceLinear(num_instances, h2, action_dim)

    def forward(self, state, mf, indices=None):
        x = torch.cat([state, mf], dim=-1)
        x = F.silu(self.norm1(self.fc1(x, indices), indices))
        x = F.silu(self.norm2(self.fc2(x, indices), indices))
        return self.actor_logits(x, indices)

    def evaluate(self, state, mf, action, masks=None, indices=None,
                 exclude_zero=False, zeta=1.0):
        logits = self.forward(state, mf, indices)

        # Temperature scaling
        if zeta != 1.0:
            logits = logits * zeta

        if masks is not None:
            logits = logits.masked_fill(masks == 0, -1e9)

        if exclude_zero and logits.shape[-1] > 1:
            zero_mask = torch.zeros_like(logits, dtype=torch.bool)
            zero_mask[:, 0] = True
            logits = logits.masked_fill(zero_mask, -1e9)

        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


class MultiInstanceCritic(nn.Module):
    def __init__(self, state_dim, mf_dim, hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        self.num_instances = num_instances

        self.fc1 = MultiInstanceLinear(num_instances, state_dim + mf_dim, h1)
        self.norm1 = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2 = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2 = MultiInstanceRMSNorm(num_instances, h2)
        self.critic = MultiInstanceLinear(num_instances, h2, 1)

    def forward(self, state, mf, indices=None):
        x = torch.cat([state, mf], dim=-1)
        x = F.silu(self.norm1(self.fc1(x, indices), indices))
        x = F.silu(self.norm2(self.fc2(x, indices), indices))
        return self.critic(x, indices).squeeze(-1)


class MFNetwork(nn.Module):
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

