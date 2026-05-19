import torch.nn as nn
import torch

def make_branch(in_dim: int, out_dim: int, hidden: int) -> nn.Sequential:
    """Two-layer MLP branch: Linear → LN → ELU → Linear → LN → ELU."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.ELU(),
        nn.Linear(hidden, out_dim),
        nn.LayerNorm(out_dim),
        nn.ELU(),
    )

def stats5(x: torch.Tensor) -> torch.Tensor:
    """
    Compute [mean, std, min, max, sum] over a 1-D tensor → (5,).
    Safe for N=1 (std→0).
    """
    if x.numel() == 0:
        return torch.zeros(5, device=x.device, dtype=x.dtype)
    return torch.stack([
        x.mean(),
        x.std(unbiased=False),
        x.min(),
        x.max(),
        x.sum(),
    ])