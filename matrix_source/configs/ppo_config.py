"""ppo_config.py — 3-phase curriculum config for simultaneous PPO."""

from matrix_source.configs.abstract_config import (
    AbstractTrainingConfig, BasePhaseConfig, EntropyConfig
)


class PPOTrainingConfig(AbstractTrainingConfig):

    MAX_TRAIN_STEPS: int = 200        # Số lần train thực sự (không phải episodes)
    CONVERGE_START: int = 140         # Bắt đầu converge từ step 140/200

    PHASE_BOUNDS = {
        'EXPLORE':  (0, 67),          # 0–33%   : 67 steps
        'REFINE':   (68, 139),        # 34–69%  : 72 steps
        'CONVERGE': (140, 199),       # 70–100% : 60 steps
    }

    # ── Upper Agent (Edge — placement) ──
    UPPER_PHASES = {
        'EXPLORE': BasePhaseConfig(
            k_epochs=5, clip_eps=0.4,
            temperature=(0.67, 1.0),
            lr_actor=(3e-4, 2e-4),
            lr_critic=(1e-3, 7e-4),
        ),
        'REFINE': BasePhaseConfig(
            k_epochs=4, clip_eps=0.3,
            temperature=(1.0, 1.43),
            lr_actor=(2e-4, 1e-4),
            lr_critic=(7e-4, 5e-4),
        ),
        'CONVERGE': BasePhaseConfig(
            k_epochs=3, clip_eps=0.2,
            temperature=(1.43, 2.0),
            lr_actor=(1e-4, 5e-5),
            lr_critic=(5e-4, 3e-4),
        ),
    }

    # ── Lower Agent (Terminal — routing) ──
    LOWER_PHASES = {
        'EXPLORE': BasePhaseConfig(
            k_epochs=7, clip_eps=0.4,
            temperature=(0.67, 1.0),
            lr_actor=(2e-4, 1.5e-4),
            lr_critic=(8e-4, 5e-4),
        ),
        'REFINE': BasePhaseConfig(
            k_epochs=5, clip_eps=0.3,
            temperature=(1.0, 1.43),
            lr_actor=(1.5e-4, 1e-4),
            lr_critic=(5e-4, 3e-4),
        ),
        'CONVERGE': BasePhaseConfig(
            k_epochs=4, clip_eps=0.2,
            temperature=(1.43, 2.0),
            lr_actor=(1e-4, 5e-5),
            lr_critic=(3e-4, 1e-4),
        ),
    }

    UPPER_ENTROPY = EntropyConfig(
        agent_type='upper',
        explore_start=0.05, explore_end=0.03,
        refine_end=0.015, focus_end=0.008, freeze_end=0.003,
        rewarm_boost=0.012, min_coef=0.003,
        rewarm_until_phase='REFINE',
    )

    LOWER_ENTROPY = EntropyConfig(
        agent_type='lower',
        explore_start=0.05, explore_end=0.03,
        refine_end=0.015, focus_end=0.008, freeze_end=0.003,
        rewarm_boost=0.012, min_coef=0.003,
        rewarm_until_phase='REFINE',
    )

    # ── Buffer thresholds ──
    # Lower: ~3800 transitions/episode → 2 episodes đủ 6400
    # Upper: ~80 transitions/episode   → 8 episodes đủ 640
    UPPER_BUFFER = {'min_size': 640, 'batch_size': 32}
    LOWER_BUFFER = {'min_size': 6400, 'batch_size': 128}
