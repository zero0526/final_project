"""
KStepMonitor
============
Lớp riêng theo dõi và vẽ biểu đồ các chỉ số hội tụ K-step equilibrium:
  - last_residual      : ‖a^k − a^(k-1)‖  (nên giảm về 0)
  - proposal_load_var  : load variance của Proposal  (trước refinement)
  - equilibrium_load_var: load variance sau K bước   (nên nhỏ hơn proposal)

Sử dụng độc lập, KHÔNG qua MetricsAggregator.
Plot được ghi đè lên cùng 1 file → luôn hiển thị trạng thái mới nhất.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend, an toàn cho training loop
import matplotlib.pyplot as plt
from collections import deque


class KStepMonitor:
    """
    Theo dõi và vẽ biểu đồ K-step diagnostics trong quá trình training.

    Parameters
    ----------
    save_dir : str
        Thư mục lưu file PNG (mặc định dùng cfg.plot_dir nếu không truyền).
    name : str
        Prefix cho tên file PNG, ví dụ ``"residual_ppo"``.
    plot_every : int
        Số lần ``record()`` được gọi giữa 2 lần vẽ lại biểu đồ.
    window : int
        Cửa sổ Moving Average.
    maxlen : int | None
        Độ dài tối đa history (None = không giới hạn).
    """

    def __init__(
        self,
        save_dir: str,
        name: str = "kstep",
        plot_every: int = 10,
        window: int = 20,
        maxlen: int | None = None,
    ):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir   = save_dir
        self.name       = name
        self.plot_every = plot_every
        self.window     = window

        kw = dict(maxlen=maxlen) if maxlen else {}
        # ── History (across updates) ──
        self.residuals:  deque[float] = deque(**kw)
        self.lv_prop:    deque[float] = deque(**kw)
        self.lv_equil:   deque[float] = deque(**kw)
        
        # ── Latest traces (inner step convergence) ──
        self.latest_res_trace: list[float] = []
        self.latest_lv_trace:  list[float] = []
        self.latest_hist_trace: list[float] = []
        
        self._call_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, agent) -> None:
        """
        Đọc diagnostics từ ``agent`` và lưu vào history.
        Vẽ biểu đồ mỗi ``plot_every`` lần gọi.

        Args:
            agent: bất kỳ object nào có thuộc tính
                   ``last_residual``, ``proposal_load_var``,
                   ``equilibrium_load_var``.
        """
        residual = getattr(agent, "last_residual",        None)
        lv_p     = getattr(agent, "proposal_load_var",    None)
        lv_e     = getattr(agent, "equilibrium_load_var", None)

        if residual is None:
            return   # không phải ResidualRoutingAgent → bỏ qua

        self.residuals.append(float(residual))
        self.lv_prop.append(float(lv_p) if lv_p is not None else 0.0)
        self.lv_equil.append(float(lv_e) if lv_e is not None else 0.0)

        # Capture traces for "Inner-loop" visualization
        self.latest_res_trace  = list(getattr(agent, "residual_trace",    []))
        self.latest_lv_trace   = list(getattr(agent, "load_var_trace",    []))
        self.latest_hist_trace = list(getattr(agent, "hist_change_trace", []))

        self._call_count += 1
        if self._call_count % self.plot_every == 0:
            self.plot()

    def plot(self) -> None:
        """Vẽ và ghi đè file PNG diagnostics."""
        if len(self.residuals) < 2:
            return

        res   = list(self.residuals)
        lv_p  = list(self.lv_prop)
        lv_e  = list(self.lv_equil)
        xs    = list(range(len(res)))
        w     = self.window

        reduction = [
            (p - e) / (p + 1e-8) for p, e in zip(lv_p, lv_e)
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f"K-step Equilibrium Diagnostics  (n={len(res)} updates) - Stage: {self.name}",
            fontsize=14, fontweight="bold",
        )

        # ── FIRST ROW: Evolution Across Updates ──────────────────────────────────
        
        # Panel 1: Residual
        ax = axes[0, 0]
        ax.plot(xs, res, color="steelblue", alpha=0.3, linewidth=1, label="Raw")
        if len(res) >= w:
            ma = self._ma(res, w)
            ax.plot(range(w - 1, len(res)), ma,
                    color="steelblue", linewidth=2, label=f"MA-{w}")
        ax.set_title("‖Δz‖ Last Residual (→ 0 = hội tụ)", fontsize=11)
        ax.set_ylabel("Norm")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # Panel 2: Load variance comparison
        ax = axes[0, 1]
        ax.plot(xs, lv_p, color="tomato",   alpha=0.3, linewidth=1, label="Proposal")
        ax.plot(xs, lv_e, color="seagreen", alpha=0.3, linewidth=1, label="Equilibrium")
        if len(lv_p) >= w:
            x_ma = range(w - 1, len(lv_p))
            ax.plot(x_ma, self._ma(lv_p, w), color="tomato",   linewidth=2)
            ax.plot(x_ma, self._ma(lv_e, w), color="seagreen", linewidth=2)
        ax.set_title("Load Variance: Prop (đỏ) vs Equil (xanh)", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # Panel 3: Reduction ratio
        ax = axes[0, 2]
        ax.plot(xs, reduction, color="mediumpurple", alpha=0.3, linewidth=1, label="Raw")
        if len(reduction) >= w:
            ax.plot(range(w - 1, len(reduction)), self._ma(reduction, w),
                    color="mediumpurple", linewidth=2, label=f"MA-{w}")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title("Load Var Reduction Ratio", fontsize=11)
        ax.grid(True, alpha=0.3)

        # ── SECOND ROW: Latest Batch Inner Convergence (Traces) ─────────────────
        
        # Panel 4: Inner Residual Trace
        ax = axes[1, 0]
        if self.latest_res_trace:
            ks = range(len(self.latest_res_trace))
            ax.plot(ks, self.latest_res_trace, marker="o", markersize=4, color="steelblue")
            ax.set_title("Trace: ‖z_{k+1} - z_k‖ per step", fontsize=11)
        ax.set_xlabel("Refinement K-steps")
        ax.grid(True, alpha=0.3)

        # Panel 5: Inner Load Variance Trace
        ax = axes[1, 1]
        if self.latest_lv_trace:
            ks = range(len(self.latest_lv_trace))
            ax.plot(ks, self.latest_lv_trace, marker="s", markersize=4, color="seagreen")
            ax.axhline(self.latest_lv_trace[0], color="tomato", linestyle="--", alpha=0.5, label="Prop")
            ax.set_title("Trace: Load Variance per step", fontsize=11)
        ax.set_xlabel("Refinement K-steps")
        ax.grid(True, alpha=0.3)

        # Panel 6: Inner Histogram Change Trace
        ax = axes[1, 2]
        if self.latest_hist_trace:
            ks = range(len(self.latest_hist_trace))
            ax.plot(ks, self.latest_hist_trace, marker="^", markersize=4, color="goldenrod")
            ax.set_title("Trace: ‖h_{k+1} - h_k‖ per step", fontsize=11)
        ax.set_xlabel("Refinement K-steps")
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(self.save_dir, f"{self.name}_kstep_diagnostics.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ma(data: list[float], w: int) -> np.ndarray:
        return np.convolve(data, np.ones(w) / w, mode="valid")
