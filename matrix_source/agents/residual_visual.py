import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from matrix_source.configs.configs import cfg

class ResidualTracker:
    def __init__(self, name="ResidualDiagnostics"):
        self.name = name
        self.history = {
            # Nhóm 1: LEARNING PROGRESS
            'reward': [],
            'ppo_ratio': [],
            'value_loss': [],
            'expl_var': [],
            
            # Nhóm 2: GRADIENT HEALTH
            'grad_prop': [],
            'grad_refine': [],
            'grad_ratio': [],
            'nan_count': [],
            'grad_sim': [],
            
            # Nhóm 3: COMPONENT ACTIVITY
            'prop_norm': [],
            'delta_norm': [],
            'final_norm': [],
            'final_max': [],
            'contrib_ratio': [],
            
            # Nhóm 4: ROUTING DYNAMICS
            'alpha_prop': [],
            'alpha_std': [],
            'pct_prop_dom': [],
            'pct_ref_dom': [],
            
            # Nhóm 5: STABILITY + ANTI-LAZINESS
            'entropy': [],
            'pct_low_ent': [],
            'entropy_pen': [],
            'norm_pen': [],
            'lazy_pen': [],
            'pct_bad': [],
            'pct_zero_delta': [],
        }
        self.step = 0

    def log(self, **kwargs):
        """
        Ghi lại các metrics.
        Sử dụng: tracker.log(reward=0.5, grad_prop=0.1, ...)
        """
        for k, v in kwargs.items():
            if k in self.history:
                # Chuyển đổi tensor về float nếu cần
                if torch.is_tensor(v):
                    v = v.item()
                elif isinstance(v, np.ndarray):
                    v = float(v.mean())
                self.history[k].append(v)
        self.step += 1

    def _moving_average(self, data, window=10):
        if len(data) < window: return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    def plot_dashboard(self, ep=None):
        """
        Vẽ dashboard chuẩn đoán 5 hàng theo yêu cầu.
        """
        if not self.history['reward']:
            return

        fig, axes = plt.subplots(5, 2, figsize=(18, 25))
        plt.suptitle(f"Residual Routing Diagnostics - {self.name} (Step {self.step})", fontsize=20, y=0.98)
        
        steps = range(len(self.history['reward']))
        window = 20

        # --- ROW 1: LEARNING PROGRESS ---
        # 1.1 Reward + PPO Ratio (Twin-X)
        ax1 = axes[0, 0]
        lns1 = ax1.plot(self.history['reward'], 'g-', alpha=0.3, label='Reward (raw)')
        if len(self.history['reward']) >= window:
            ax1.plot(range(window-1, len(self.history['reward'])), self._moving_average(self.history['reward'], window), 'g-', linewidth=2, label=f'Reward MA-{window}')
        ax1.set_ylabel('Reward', color='g')
        ax1.set_title('Reward Convergence & PPO Ratio')
        ax1.grid(True, alpha=0.3)

        ax1_twin = ax1.twinx()
        lns2 = ax1_twin.plot(self.history['ppo_ratio'], 'b--', alpha=0.5, label='PPO Ratio')
        ax1_twin.axhline(y=1.0, color='gray', linestyle=':', alpha=0.8)
        ax1_twin.axhline(y=1.3, color='r', linestyle=':', alpha=0.5)
        ax1_twin.axhline(y=0.7, color='r', linestyle=':', alpha=0.5)
        ax1_twin.set_ylabel('PPO Ratio', color='b')
        ax1_twin.set_ylim(0.5, 1.5)
        
        # Merge legends
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper left')

        # 1.2 Value Loss + Explained Variance
        ax2 = axes[0, 1]
        ax2.plot(self.history['value_loss'], 'r-', alpha=0.3, label='Value Loss')
        if len(self.history['value_loss']) >= window:
             ax2.plot(range(window-1, len(self.history['value_loss'])), self._moving_average(self.history['value_loss'], window), 'r-', linewidth=2)
        ax2.set_yscale('log')
        ax2.set_title('Value Loss & Explained Var')
        ax2.set_ylabel('Loss (log)')
        ax2.grid(True, which="both", alpha=0.3)

        ax2_twin = ax2.twinx()
        ax2_twin.plot(self.history['expl_var'], 'k-', alpha=0.7, label='Expl. Var')
        ax2_twin.axhline(y=0.5, color='g', linestyle='--')
        ax2_twin.set_ylabel('Variance', color='k')
        ax2_twin.set_ylim(-0.1, 1.1)
        ax2.legend(loc='upper right')

        # --- ROW 2: GRADIENT HEALTH ---
        # 2.1 Grad Norms (Log Scale)
        ax3 = axes[1, 0]
        ax3.plot(self.history['grad_prop'], 'b-', label='∇Proposal')
        ax3.plot(self.history['grad_refine'], 'r-', label='∇Refine')
        ax3.set_yscale('log')
        ax3.set_title('Gradient Norms')
        ax3.legend()
        ax3.grid(True, which="both")

        # 2.2 Grad Ratio + NaN Count
        ax4 = axes[1, 1]
        ax4.plot(self.history['grad_ratio'], 'purple', label='Ratio R/P')
        ax4.axhline(y=1.0, color='gray', linestyle='--')
        ax4.set_yscale('log')
        ax4.set_title('Gradient Balance (R/P) & NaN')
        
        ax4_twin = ax4.twinx()
        ax4_twin.bar(steps, self.history['nan_count'], color='red', alpha=0.5, label='NaNs')
        ax4_twin.set_ylabel('NaN Count', color='r')
        ax4_twin.set_ylim(0, 5)
        ax4.legend(loc='upper left')

        # --- ROW 3: COMPONENT ACTIVITY ---
        # 3.1 Logit Norms
        ax5 = axes[2, 0]
        ax5.plot(self.history['prop_norm'], 'b-', label='||Prop||')
        ax5.plot(self.history['delta_norm'], 'r-', label='||Δ||')
        ax5.plot(self.history['final_norm'], 'g-', label='||Final||')
        ax5.axhline(y=5.0, color='orange', linestyle='--', label='Target Ceiling', alpha=0.6)
        ax5.set_title('Component Norms (Logits)')
        ax5.legend()
        ax5.grid(True)

        # 3.2 Contribution Ratio
        ax6 = axes[2, 1]
        ax6.plot(self.history['contrib_ratio'], 'teal')
        ax6.axhline(y=0.5, color='gray', linestyle='--')
        ax6.set_ylim(0, 1)
        ax6.set_title('Contribution Ratio (Δ / (Prop + Δ))')
        ax6.grid(True)

        # --- ROW 4: ROUTING + STABILITY ---
        # 4.1 Alpha Prop Mean ± Std
        ax7 = axes[3, 0]
        data_alpha = np.array(self.history['alpha_prop'])
        data_std = np.array(self.history['alpha_std'])
        ax7.plot(data_alpha, 'orange', label='α_prop mean')
        ax7.fill_between(range(len(data_alpha)), data_alpha - data_std, data_alpha + data_std, color='orange', alpha=0.2)
        ax7.axhline(y=0.8, color='r', linestyle=':', label='Prop Dom')
        ax7.axhline(y=0.2, color='b', linestyle=':', label='Refine Dom')
        ax7.set_ylim(0, 1)
        ax7.set_title('Routing Dynamics (α_prop)')
        ax7.legend()
        ax7.grid(True)

        # 4.2 Entropy + Penalties
        ax8 = axes[3, 1]
        ax8.plot(self.history['entropy'], 'cyan', label='Entropy')
        ax8.axhline(y=0.2, color='r', linestyle='--', label='Floor')
        ax8.set_ylabel('Entropy')
        ax8_twin = ax8.twinx()
        ax8_twin.plot(self.history['entropy_pen'], 'r:', alpha=0.5, label='Ent Pen')
        ax8_twin.plot(self.history['norm_pen'], 'b:', alpha=0.5, label='Norm Pen')
        ax8_twin.set_ylabel('Penalty Value')
        ax8.set_title('Final Entropy & Penalties')
        ax8.legend(loc='upper right')

        # --- ROW 5: ANTI-LAZINESS ---
        # 5.1 Lazy Penalty + % Zero Delta (Combining two views)
        ax9 = axes[4, 0]
        ax9.plot(self.history['lazy_pen'], 'orange', label='Lazy Penalty')
        ax9.set_title('Anti-Laziness Penalty')
        ax9.grid(True)
        ax9.legend()

        ax10 = axes[4, 1]
        ax10.plot(self.history['pct_zero_delta'], 'r-', label='% Zero Δ')
        ax10.plot(self.history['pct_bad'], 'b-', label='% Bad Tasks (A < 0)')
        ax10.axhline(y=70, color='gray', linestyle=':', label='Collapse Threshold')
        ax10.set_ylim(0, 100)
        ax10.set_title('Refine Collapse Monitoring')
        ax10.legend()
        ax10.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        save_dir = cfg.plot_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        timestamp = ep if ep is not None else self.step
        filename = f"residual_diagnostics_{self.name.lower()}_step.png"
        save_path = os.path.join(save_dir, filename)
        
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"📊 Dashboard chẩn đoán đã lưu tại: {save_path}")
        return save_path

# Utility function for explained variance
def explained_variance(y_pred, y_true):
    """Computes fraction of variance that y_pred explains about y_true."""
    var_y = torch.var(y_true)
    return 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
