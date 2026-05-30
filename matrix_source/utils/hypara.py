import numpy as np


def cosine_interp(val_range, progress):
    """Cosine interpolation: val_range[0] → val_range[1]."""
    if isinstance(val_range, (int, float)):
        return val_range
    start, end = val_range
    return end + 0.5 * (start - end) * (1 + np.cos(np.pi * progress))


def get_phase(cycle):
    if cycle <= 13:
        return 'EXPLORE'
    elif cycle <= 33:
        return 'REFINE'
    return 'CONVERGE'


def get_phase_progress(cycle, phase, phase_cfg):
    bounds = phase_cfg.PHASE_BOUNDARIES[phase]
    start, end = bounds
    return (cycle - start) / max(1, end - start)


def apply_phase_config(agent, phase_cfg, cycle, phase):
    """
    Áp dụng phase config.
    Temperature được convert sang zeta = 1/T.
    Trả về zeta hiện tại.
    """
    progress = get_phase_progress(cycle, phase)

    # ===== Entropy =====
    agent.entropy_coef = cosine_interp(phase_cfg['entropy'], progress)

    # ===== Learning rates =====
    if isinstance(phase_cfg['lr_actor'], tuple):
        new_lr = cosine_interp(phase_cfg['lr_actor'], progress)
        for pg in agent.optimizer_actor.param_groups:
            pg['lr'] = new_lr
    if isinstance(phase_cfg['lr_critic'], tuple):
        new_lr = cosine_interp(phase_cfg['lr_critic'], progress)
        for pg in agent.optimizer_critic.param_groups:
            pg['lr'] = new_lr

    # ===== Scalar params =====
    agent.k_epochs = phase_cfg['k_epochs']
    agent.eps_clip = phase_cfg['clip_eps']
    agent.tau = phase_cfg['tau']

    # ===== Temperature → zeta =====
    # T cao → explore (zeta nhỏ, logits thu lại, phân phối phẳng)
    # T thấp → exploit (zeta lớn, logits phóng to, phân phối nhọn)
    temperature = cosine_interp(phase_cfg['temperature'], progress)
    zeta = 1.0 / temperature

    return zeta, temperature
