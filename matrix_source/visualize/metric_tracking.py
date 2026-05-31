import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from matrix_source.configs.configs import cfg

class PPOTracker:
    """
    Specialized tracker for PPO-specific metrics:
    1. Reward (Mean/Std)
    2. Normalized Entropy (H / log(|A|))
    3. Zeta (Temperature/Scaling)
    4. Alpha (Entropy Coefficient)
    5. Max Action Probability
    """
    def __init__(self, name="ppo_agent"):
        self.name = name
        self.history = defaultdict(list)
        self.plot_dir = cfg.plot_dir
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

    def record(self, reward_mean, reward_std, entropy, norm_entropy, zeta, alpha, max_prob):
        self.history["reward_mean"].append(reward_mean)
        self.history["reward_std"].append(reward_std)
        self.history["entropy"].append(entropy)
        self.history["norm_entropy"].append(norm_entropy)
        self.history["zeta"].append(zeta)
        self.history["alpha"].append(alpha)
        self.history["max_prob"].append(max_prob)

    def plot(self):
        if not self.history["reward_mean"]:
            return

        fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
        plt.suptitle(f"PPO Training Dashboard - {self.name.upper()}", fontsize=16)

        # 1. Reward
        ax = axes[0]
        means = np.array(self.history["reward_mean"])
        stds = np.array(self.history["reward_std"])
        steps = np.arange(len(means))
        ax.plot(steps, means, label="Reward Mean", color="blue")
        ax.fill_between(steps, means - stds, means + stds, color="blue", alpha=0.2, label="Reward Std")
        ax.set_ylabel("Reward")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # 2. Entropy
        ax = axes[1]
        ax.plot(self.history["norm_entropy"], label="Normalized Entropy", color="green")
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
        ax.axhline(y=0.0, color='r', linestyle='--', alpha=0.3)
        ax.set_ylabel("Norm. Entropy")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # 3. Zeta
        ax = axes[2]
        ax.plot(self.history["zeta"], label="Zeta (Temperature)", color="purple")
        ax.set_ylabel("Zeta")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # 4. Alpha (Entropy Coef)
        ax = axes[3]
        ax.plot(self.history["alpha"], label="Entropy Coef (Alpha)", color="orange")
        ax.set_ylabel("Alpha")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # 5. Max Probability
        ax = axes[4]
        ax.plot(self.history["max_prob"], label="Max Action Prob", color="brown")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Max Prob")
        ax.set_xlabel("Update Step")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = f"{self.name.lower()}_metrics_dashboard.png"
        plt.savefig(os.path.join(self.plot_dir, filename), dpi=150)
        plt.close()
