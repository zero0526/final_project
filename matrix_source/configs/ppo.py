class PPOCfg:
    """
    50 cycles — 3 phase.
    Temperature (T)= 1/zeta
    """

    PHASE_BOUNDARIES = {
        'EXPLORE':  (1, 13),     # 13 cycles
        'REFINE':   (14, 33),    # 20 cycles
        'CONVERGE': (34, 50),    # 17 cycles
    }

    # ==================== UPPER AGENT ====================
    UPPER_CONFIG = {
        'EXPLORE': {
            'entropy': (0.06, 0.04),
            'lr_actor': 1e-4,
            'lr_critic': 5e-4,
            'k_epochs': 4,
            'clip_eps': 0.4,
            'temperature': (2.0, 1.5),    # T từ 2.0 → 1.5 (giảm dần, vẫn explore)
            'rewarm': True,
            'rewarm_boost': 0.015,
            'tau': 0.005,
        },
        'REFINE': {
            'entropy': (0.04, 0.012),
            'lr_actor': 1e-4,
            'lr_critic': 5e-4,
            'k_epochs': 3,
            'clip_eps': 0.3,
            'temperature': (1.5, 1.0),    # T từ 1.5 → 1.0 (bình thường)
            'rewarm': True,
            'rewarm_boost': 0.01,
            'tau': 0.005,
        },
        'CONVERGE': {
            'entropy': (0.012, 0.004),
            'lr_actor': (1e-4, 5e-7),
            'lr_critic': (5e-4, 2.5e-6),
            'k_epochs': 3,
            'clip_eps': 0.2,
            'temperature': (1.0, 0.3),    # T từ 1.0 → 0.3 (rất nhọn, exploit mạnh)
            'rewarm': False,
            'tau': 0.003,
        }
    }

    # ==================== LOWER AGENT ====================
    LOWER_CONFIG = {
        'EXPLORE': {
            'entropy': (0.05, 0.03),
            'lr_actor': 1e-4,
            'lr_critic': 5e-4,
            'k_epochs': 5,
            'clip_eps': 0.4,
            'temperature': (1.8, 1.3),    # Lower: ít explore hơn upper 1 chút
            'rewarm': True,
            'rewarm_boost': 0.01,
            'tau': 0.005,
        },
        'REFINE': {
            'entropy': (0.03, 0.008),
            'lr_actor': 1e-4,
            'lr_critic': 5e-4,
            'k_epochs': 4,
            'clip_eps': 0.3,
            'temperature': (1.3, 1.0),
            'rewarm': True,
            'rewarm_boost': 0.008,
            'tau': 0.005,
        },
        'CONVERGE': {
            'entropy': (0.008, 0.003),
            'lr_actor': (1e-4, 5e-7),
            'lr_critic': (5e-4, 2.5e-6),
            'k_epochs': 3,
            'clip_eps': 0.2,
            'temperature': (1.0, 0.3),
            'rewarm': False,
            'tau': 0.003,
        }
    }
