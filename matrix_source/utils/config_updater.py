"""config_updater.py — Phase config + entropy + LR scheduling.

Simplified: NO target networks.
PPO's clipped objective + on-policy data = sufficient stability.
"""

import numpy as np
from typing import Dict, Tuple, Optional


# ── Cosine Interpolation ──

def cosine_interp(val_range, progress: float):
    """Cosine interpolation. Float → return as-is."""
    if isinstance(val_range, (int, float)):
        return val_range
    start, end = val_range
    return end + 0.5 * (start - end) * (1 + np.cos(np.pi * progress))


# ── Temperature & Zeta ──

def get_temperature_and_zeta(phase_cfg, cycle: int, phase_bounds: Tuple):
    start, end = phase_bounds
    progress = (cycle - start) / max(1, end - start)
    temperature = cosine_interp(phase_cfg.temperature, progress)
    zeta = 1.0 / temperature
    return temperature, zeta


# ── Scalar Config ──

def apply_scalar_config(agent, phase_cfg):
    agent.k_epochs = phase_cfg.k_epochs
    agent.eps_clip = phase_cfg.clip_eps


# ── LR Config ──

def apply_lr_config(agent, phase_cfg, cycle, phase_bounds):
    """Apply LR with cosine interpolation. No target networks involved."""
    progress = (cycle - phase_bounds[0]) / max(1, phase_bounds[1] - phase_bounds[0])

    def _set_lr(optimizer, lr_val):
        if isinstance(lr_val, tuple):
            lr_val = cosine_interp(lr_val, progress)
        for pg in optimizer.param_groups:
            pg['lr'] = lr_val

    # MF LR — independent (FIX: not overwritten)
    mf_lr = getattr(phase_cfg, 'mf_lr', None)
    if mf_lr is not None and hasattr(agent, 'mf_optimizer'):
        _set_lr(agent.mf_optimizer, mf_lr)

    # Actor LR
    if hasattr(agent, 'optimizer_actor'):
        _set_lr(agent.optimizer_actor, phase_cfg.lr_actor)

    # Critic LR — auto detect layout
    if hasattr(agent, 'bone_optimizer'):
        _set_lr(agent.bone_optimizer, phase_cfg.lr_critic)
        if hasattr(agent, 'head_optimizer'):
            _set_lr(agent.head_optimizer, phase_cfg.lr_critic)
    elif hasattr(agent, 'optimizer_critic'):
        _set_lr(agent.optimizer_critic, phase_cfg.lr_critic)


# ── Universal Apply ──

def apply_config_to_agent(agent, phase_cfg, cycle, phase_bounds, is_converge):
    apply_scalar_config(agent, phase_cfg)
    apply_lr_config(agent, phase_cfg, cycle, phase_bounds)
    temperature, zeta = get_temperature_and_zeta(phase_cfg, cycle, phase_bounds)
    return temperature, zeta


# ── Entropy Scheduler ──

def create_entropy_scheduler(entropy_cfg, total_train_steps: int):
    phase_to_ratio = {'EXPLORE': 0.25, 'REFINE': 0.60, None: 0.0}
    rewarm_until = int(
        phase_to_ratio.get(entropy_cfg.rewarm_until_phase, 0.0) * total_train_steps)

    return _EntropyScheduler(
        initial=entropy_cfg.explore_start,
        vals=[entropy_cfg.explore_start, entropy_cfg.explore_end,
              entropy_cfg.refine_end, entropy_cfg.focus_end,
              entropy_cfg.freeze_end],
        rewarm_boost=entropy_cfg.rewarm_boost,
        min_coef=entropy_cfg.min_coef,
        total_steps=total_train_steps,
        rewarm_until=rewarm_until,
    )


class _EntropyScheduler:
    """4-phase cosine entropy with adaptive re-warm."""

    def __init__(self, initial, vals, rewarm_boost, min_coef,
                 total_steps, rewarm_until):
        self.coef = initial
        self.vals = vals
        self.total_steps = total_steps
        self.current_step = 0
        self.rewarm_boost = rewarm_boost
        self.min_coef = min_coef
        self.rewarm_until = rewarm_until
        self.reward_history = []
        self.last_rewarm = -200

        self.explore_end = int(total_steps * 0.25)
        self.refine_end = int(total_steps * 0.60)
        self.focus_end = int(total_steps * 0.90)

    def _cosine(self, v0, v1, step, length):
        if length <= 0:
            return v1
        p = min(1.0, step / max(1, length))
        return v1 + 0.5 * (v0 - v1) * (1 + np.cos(np.pi * p))

    def _base(self):
        s = self.current_step
        v = self.vals
        if s <= self.explore_end:
            return self._cosine(v[0], v[1], s, self.explore_end)
        elif s <= self.refine_end:
            return self._cosine(v[1], v[2], s - self.explore_end,
                                self.refine_end - self.explore_end)
        elif s <= self.focus_end:
            return self._cosine(v[2], v[3], s - self.refine_end,
                                self.focus_end - self.refine_end)
        return self._cosine(v[3], v[4], s - self.focus_end,
                            self.total_steps - self.focus_end)

    def step(self, reward=None):
        self.current_step += 1
        if reward is not None:
            self.reward_history.append(reward)
        base = self._base()
        boost = 0.0
        if (self.current_step <= self.rewarm_until and
                len(self.reward_history) >= 80 and
                self.current_step - self.last_rewarm >= 80):
            recent = np.mean(self.reward_history[-40:])
            older = np.mean(self.reward_history[-80:-40])
            if (recent - older) / (abs(older) + 1e-8) < 0.005:
                boost = self.rewarm_boost
                self.last_rewarm = self.current_step
        self.coef = min(base + boost, self.vals[0])
        self.coef = max(self.coef, self.min_coef)
        return self.coef

    def get_phase_name(self):
        s = self.current_step
        if s <= self.explore_end:
            return 'EXPLORE'
        elif s <= self.refine_end:
            return 'REFINE'
        elif s <= self.focus_end:
            return 'FOCUS'
        return 'FREEZE'


# ── LR Schedulers ──

def create_lr_schedulers(agent, lr_min_actor=1e-6, lr_min_mf=1e-6,
                          lr_min_critic=5e-6, total_steps=500):
    schedulers = {}
    lr_map = {
        'optimizer_actor': lr_min_actor,       # FIX: was swapped
        'optimizer_critic': lr_min_critic,
        'optimizer_proposal': lr_min_actor,
        'optimizer_refine': lr_min_actor * 0.5,
        'bone_optimizer': lr_min_critic,
        'head_optimizer': lr_min_critic,
        'mf_optimizer': lr_min_mf,             # FIX: was swapped
    }
    for attr, lr_min in lr_map.items():
        if hasattr(agent, attr):
            name = attr.replace('optimizer_', '').replace('_optimizer', '')
            schedulers[name] = _LRScheduler(
                getattr(agent, attr), lr_min, total_steps)
    return schedulers


def start_convergence(schedulers: Dict):
    for s in schedulers.values():
        s.start_annealing()


def step_all_schedulers(schedulers: Dict):
    for s in schedulers.values():
        s.step()


class _LRScheduler:
    def __init__(self, optimizer, lr_min, total_steps):
        self.optimizer = optimizer
        self.lr_min = lr_min
        self.total_steps = total_steps
        self.current_step = 0
        self.annealing = False
        self.start_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def start_annealing(self):
        self.start_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
        self.current_step = 0
        self.annealing = True

    def step(self):
        if not self.annealing:
            return
        self.current_step += 1
        p = min(1.0, self.current_step / self.total_steps)
        for i, pg in enumerate(self.optimizer.param_groups):
            pg['lr'] = self.lr_min + 0.5 * (self.start_lrs[i] - self.lr_min) * \
                       (1 + np.cos(np.pi * p))
