import numpy as np

class TemperatureScheduler:
    def __init__(
        self,
        T_start=1.0,
        T_min=0.3,
        decay=0.9995,
        recovery_T=0.5,
        td_threshold=0.1,
        ema_alpha=0.05,
    ):
        self.T = T_start
        self.T_start = T_start
        self.T_min = T_min
        self.decay = decay
        self.recovery_T = recovery_T
        self.td_threshold = td_threshold

        self.td_ema = None
        self.reward_ema = None
        self.ema_alpha = ema_alpha

        self.in_recovery = False
        self.recovery_steps = 0

    def update_ema(self, value, ema):
        if ema is None:
            return value
        return self.ema_alpha * value + (1 - self.ema_alpha) * ema

    def step(self, td_error, reward):
        # --- update EMA ---
        self.td_ema = self.update_ema(td_error, self.td_ema)
        self.reward_ema = self.update_ema(reward, self.reward_ema)

        # --- detect instability ---
        unstable = self.td_ema is not None and self.td_ema > self.td_threshold

        # --- recovery phase ---
        if unstable:
            self.in_recovery = True
            self.recovery_steps = 50  # số step recover

        if self.in_recovery:
            # tăng nhẹ T để cứu training
            self.T = min(self.recovery_T, self.T + 0.01)
            self.recovery_steps -= 1

            if self.recovery_steps <= 0:
                self.in_recovery = False

            return self.T

        # --- normal decay (smooth) ---
        if self.td_ema is not None and self.td_ema < self.td_threshold:
            # chỉ decay khi ổn định
            self.T = max(self.T_min, self.T * self.decay)

        return self.T