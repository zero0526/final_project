import numpy as np
import torch.nn.functional as F
import torch

def to_binary(action_id, dim):
    """
    Converts an integer action ID into a binary vector of size dim.
    Used for multi-binary action spaces like placement.
    """
    binary_str = bin(int(action_id))[2:].zfill(dim)
    return np.array([int(char) for char in reversed(binary_str)])

def from_binary(binary_array):
    """
    Converts a binary vector back into an integer action ID.
    """
    res = 0
    for i, val in enumerate(binary_array):
        if val > 0:
            res += (1 << i)
    return res

def one_hot(idx, dim):
    """
    Returns a one-hot vector of size dim.
    """
    vec = np.zeros(dim)
    if 0 <= idx < dim:
        vec[idx] = 1
    return vec

def compute_kl(prop_logits, delta_logits, mask=None):
    p_detached = prop_logits.detach()
    final = p_detached + delta_logits

    if mask is not None:
        p_detached = p_detached.masked_fill(mask == 0, -1e9)
        final = final.masked_fill(mask == 0, -1e9)

    log_p = F.log_softmax(p_detached, dim=-1)
    log_q = F.log_softmax(final, dim=-1)
    p = F.softmax(p_detached, dim=-1)

    kl = (p * (log_p - log_q)).sum(dim=-1).mean()
    return kl

def compute_gae(rewards, next_values, values, dones, agent_ids, gamma, lmbda):
    """Generalized Advantage Estimation. Vectorized mask."""
    device = rewards.device
    num_steps = rewards.size(0)

    deltas = rewards + gamma * next_values * (1 - dones) - values
    advantages = torch.zeros_like(deltas)

    masks = (1 - dones) * (gamma * lmbda)
    boundary_mask = torch.ones(num_steps, device=device)
    if num_steps > 1:
        boundary_mask[:-1] = (agent_ids[:-1] == agent_ids[1:]).float()

    combined_mask = masks * boundary_mask

    curr_advantage = 0
    for t in reversed(range(num_steps)):
        curr_advantage = deltas[t] + curr_advantage * (
            combined_mask[t] if t < num_steps - 1 else 0
        )
        advantages[t] = curr_advantage

    return advantages


