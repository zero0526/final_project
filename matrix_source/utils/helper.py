def get_grad_norm(module):
    """Calculate sum of gradient norm for a module."""
    total_norm = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5
