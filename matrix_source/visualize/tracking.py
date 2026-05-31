"""
Unified Training Dashboard — Metric tracking + auto-plotting for 4 architectures.
All algorithms use the same class, only differing in metric list.

Usage:
    dashboard = TrainingDashboard(
        agent_name="upper_ppo",
        metrics_config=PPOMetricsConfig,
        save_dir="metrics/",
        plot_every=50,          # Plot every 50 learn() steps
    )

    # In learn():
    dashboard.log({...})

    # Auto-plot khi step_counter % plot_every == 0
    # Or call manually:
    dashboard.plot()
"""
import json
import time
from pathlib import Path
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Agg')


# 1. METRIC DEFINITIONS
@dataclass
class MetricDef:
    """Định nghĩa MỘT metric cần track."""
    key: str                          # Key in dict log
    title: str                        # Plot title
    color: str = 'blue'               # Line color
    smooth_alpha: float = 0.1         # EMA alpha (0 = no smooth)
    yscale: str = 'linear'            # 'linear' or 'log'
    ylabel: str = ''                  # Y-axis label
    hlines: Dict[float, str] = field(default_factory=dict)
    # hlines = {0.5: 'clip threshold', 1.0: 'baseline'}
    invert: bool = False              # True = better when LOWER (loss, energy)
    rolling_window: int = 0           # Rolling average window (0 = use EMA)


@dataclass
class MetricGroup:
    """Group of metrics displayed on the same subplot."""
    title: str
    metrics: List[MetricDef]
    layout: str = 'overlay'           # 'overlay' (same axis) or 'stack'


# 2. PREDEFINED METRIC CONFIGS FOR 4 ARCHITECTURES
M_VALUE_LOSS = MetricDef('v_loss', 'Value Loss', 'blue', 0.05, 'log')
M_ACTOR_LOSS = MetricDef('actor_loss', 'Actor Loss', 'red', 0.05)
M_ENTROPY = MetricDef('entropy', 'Policy Entropy', 'teal', 0.05,
                       hlines={0.01: 'collapse'})
M_ENTROPY_COEF = MetricDef('entropy_coef', 'Entropy Coef', 'darkblue', 0.0)
M_REWARD = MetricDef('avg_reward', 'Mean Reward', 'green', 0.05, invert=True)
M_RATIO = MetricDef('ratio_mean', 'PPO Ratio', 'purple', 0.05,
                     hlines={1.0: 'baseline'})
M_KL = MetricDef('approx_kl', 'KL Divergence', 'navy', 0.05,
                  hlines={0.02: 'target'})
M_CLIP_FRAC = MetricDef('clip_fraction', 'Clip Fraction', 'brown', 0.05,
                          hlines={0.1: 'good', 0.3: 'bad'})
M_GRAD_ACTOR = MetricDef('grad_norm_actor', 'Actor Grad Norm', 'red', 0.05, 'log')
M_GRAD_CRITIC = MetricDef('grad_norm_critic', 'Critic Grad Norm', 'blue', 0.05, 'log')
M_ADV_MEAN = MetricDef('advantage_mean', 'Adv Mean', 'purple', 0.05)
M_ADV_STD = MetricDef('advantage_std', 'Adv Std', 'orange', 0.05)
M_VALUE_PRED = MetricDef('avg_value_pred', 'V(s) Predicted', 'blue', 0.05)
M_RETURNS = MetricDef('avg_returns', 'Returns', 'orange', 0.05)
M_TEMPERATURE = MetricDef('temperature', 'Temperature (T)', 'darkorange', 0.0)
M_ZETA = MetricDef('zeta', 'Zeta = 1/T', 'crimson', 0.0)
M_LR_ACTOR = MetricDef('lr_actor', 'Actor LR', 'red', 0.0, 'log')
M_LR_CRITIC = MetricDef('lr_critic', 'Critic LR', 'blue', 0.0, 'log')
M_K_EPOCHS = MetricDef('k_epochs', 'k_epochs', 'darkgreen', 0.0)


# ─── Domain metrics ───
M_ENERGY = MetricDef('energy', 'Energy', 'red', 0.05, invert=True)
M_DELAY = MetricDef('delay', 'Avg Delay', 'blue', 0.05, invert=True)
M_QOS = MetricDef('qos_satisfaction', 'QoS Satisfaction', 'green', 0.05)
M_THROUGHPUT = MetricDef('throughput', 'Task Throughput', 'orange', 0.05)
M_BACKLOG = MetricDef('backlog', 'Backlog Queue', 'brown', 0.05, invert=True)
M_LYAPUNOV_DIFF = MetricDef('lyapunov_diff', 'Lyapunov Diff', 'magenta', 0.05, invert=True)


# ─── SCAFFOLD-specific ───
M_SCAFFOLD_SYNC = MetricDef('scaffold_sync_count', 'SCAFFOLD Syncs', 'darkviolet', 0.0)
M_BACKBONE_LR = MetricDef('lr_backbone', 'Backbone LR', 'darkred', 0.0, 'log')
M_HEAD_LR = MetricDef('lr_head', 'Head LR', 'darkblue', 0.0, 'log')


# ─── Residual-specific ───
M_KL_COEF = MetricDef('kl_coef', 'KL Coef', 'navy', 0.0)
M_L2_COEF = MetricDef('l2_coef', 'L2 Coef', 'brown', 0.0)
M_DELTA_NORM = MetricDef('delta_logits_norm', '|Δz| Mean', 'crimson', 0.05)
M_PROPOSAL_LR = MetricDef('lr_proposal', 'Proposal LR', 'red', 0.0, 'log')
M_REFINE_LR = MetricDef('lr_refine', 'Refine LR', 'darkorange', 0.0, 'log')


# ─── GRU-specific ───
M_GRU_GRAD = MetricDef('grad_norm_gru', 'GRU Grad Norm', 'darkred', 0.05, 'log')
M_HIDDEN_NORM = MetricDef('hidden_state_norm', 'Hidden State Norm', 'teal', 0.05)
M_COLLECT_SIZE = MetricDef('collect_size', 'Collect Size', 'gray', 0.0)


# ============================================================
# 3. DEFAULT CONFIGS CHO 4 KIẾN TRÚC
# ============================================================

PPOMetricGroups = [
    MetricGroup('Reward & Losses', [M_REWARD, M_VALUE_LOSS, M_ACTOR_LOSS]),
    MetricGroup('Exploration', [M_ENTROPY, M_ENTROPY_COEF, M_TEMPERATURE, M_ZETA]),
    MetricGroup('PPO Diagnostics', [M_RATIO, M_KL, M_CLIP_FRAC]),
    MetricGroup('Gradients', [M_GRAD_ACTOR, M_GRAD_CRITIC]),
    MetricGroup('Advantage & Value', [M_ADV_MEAN, M_ADV_STD, M_VALUE_PRED, M_RETURNS]),
    MetricGroup('Learning Rates', [M_LR_ACTOR, M_LR_CRITIC, M_K_EPOCHS]),
    MetricGroup('Domain: Energy', [M_ENERGY]),
    MetricGroup('Domain: Delay', [M_DELAY]),
    MetricGroup('Domain: QoS & Throughput', [M_QOS, M_THROUGHPUT, M_LYAPUNOV_DIFF]),
]

SCAFFOLDFedRepMetricGroups = [
    MetricGroup('Reward & Losses', [M_REWARD, M_VALUE_LOSS, M_ACTOR_LOSS]),
    MetricGroup('Exploration', [M_ENTROPY, M_ENTROPY_COEF, M_TEMPERATURE, M_ZETA]),
    MetricGroup('PPO Diagnostics', [M_RATIO, M_KL, M_CLIP_FRAC]),
    MetricGroup('Gradients', [M_GRAD_ACTOR, M_GRAD_CRITIC]),
    MetricGroup('Learning Rates', [M_LR_ACTOR, M_BACKBONE_LR, M_HEAD_LR, M_K_EPOCHS]),
    MetricGroup('SCAFFOLD', [M_SCAFFOLD_SYNC]),
    MetricGroup('Domain Metrics', [M_ENERGY, M_DELAY, M_QOS, M_THROUGHPUT, M_LYAPUNOV_DIFF]),
]

GRUSeqMetricGroups = [
    MetricGroup('Reward & Losses', [M_REWARD, M_VALUE_LOSS, M_ACTOR_LOSS]),
    MetricGroup('Exploration', [M_ENTROPY, M_ENTROPY_COEF, M_TEMPERATURE, M_ZETA]),
    MetricGroup('PPO Diagnostics', [M_RATIO, M_KL, M_CLIP_FRAC]),
    MetricGroup('GRU Health', [M_GRU_GRAD, M_GRAD_CRITIC, M_HIDDEN_NORM]),
    MetricGroup('Collection', [M_COLLECT_SIZE, M_K_EPOCHS]),
    MetricGroup('Learning Rates', [M_LR_ACTOR, M_LR_CRITIC]),
    MetricGroup('Domain Metrics', [M_ENERGY, M_DELAY, M_QOS, M_THROUGHPUT, M_LYAPUNOV_DIFF]),
]

ResidualMetricGroups = [
    MetricGroup('Reward & Losses', [M_REWARD, M_VALUE_LOSS, M_ACTOR_LOSS]),
    MetricGroup('Exploration', [M_ENTROPY, M_ENTROPY_COEF, M_TEMPERATURE, M_ZETA]),
    MetricGroup('Residual Architecture', [M_DELTA_NORM, M_KL_COEF, M_L2_COEF]),
    MetricGroup('PPO Diagnostics', [M_RATIO, M_KL, M_CLIP_FRAC]),
    MetricGroup('Gradients', [M_GRAD_ACTOR, M_GRAD_CRITIC]),
    MetricGroup('Learning Rates', [M_PROPOSAL_LR, M_REFINE_LR, M_LR_CRITIC, M_K_EPOCHS]),
    MetricGroup('Domain Metrics', [M_ENERGY, M_DELAY, M_QOS, M_THROUGHPUT, M_LYAPUNOV_DIFF]),
]


# 4. MAIN CLASS
class TrainingDashboard:
    """
    Unified dashboard for ALL architectures.

    Auto:
      - Collect metrics via log()
      - Filter out metrics with no data (avoid empty subplots)
      - Calculate dynamic grid layout based on metric groups with data
      - Plot every plot_every steps
      - Save JSON + PNG
    """

    def __init__(
        self,
        agent_name: str,
        metric_groups: List[MetricGroup],
        save_dir: str = "metrics/",
        plot_every: int = 50,
        smooth_alpha: float = 0.1,
    ):
        """
        Args:
            agent_name: agent name (e.g., "upper_ppo", "lower_scaffold")
            metric_groups: list of MetricGroup from config
            save_dir: output directory
            plot_every: plot every N learn() calls (0 = manual only)
            smooth_alpha: default EMA alpha if metric does not override
        """
        self.agent_name = agent_name
        self.metric_groups = metric_groups
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.plot_every = plot_every
        self.default_smooth_alpha = smooth_alpha

        # ─── Raw data ───
        self.data = defaultdict(list)
        self.steps = []
        self.cycle_markers = []   # [(step, phase_name), ...]

        # ─── Metadata ───
        self.start_time = time.time()
        self.step_counter = 0

        # ─── Build flat metric lookup ───
        self._all_metrics = OrderedDict()
        for group in metric_groups:
            for m in group.metrics:
                if m.key not in self._all_metrics:
                    self._all_metrics[m.key] = m

    # LOG — Gọi trong learn()
    def log(self, metrics: Dict[str, Any], step: int = None):
        """
        Log metrics for 1 learn() step.
        Only save metrics declared in metric_groups.
        Auto-plot when reaching plot_every.
        """
        if step is None:
            step = self.step_counter
        self.step_counter = step + 1

        self.steps.append(step)

        for key in self._all_metrics:
            value = metrics.get(key)
            if value is not None:
                self.data[key].append(float(value))
            else:
                self.data[key].append(None)

        # Auto-plot
        if self.plot_every > 0 and self.step_counter % self.plot_every == 0:
            self.plot()

    def log_cycle(self, cycle: int, phase: str, **extra):
        """Mark phase boundary."""
        step = self.steps[-1] if self.steps else 0
        self.cycle_markers.append((step, phase))

        for key, value in extra.items():
            if key in self._all_metrics and value is not None:
                if len(self.data[key]) < len(self.steps):
                    self.data[key].append(float(value))

    # ════════════════════════════════════════════════════
    # PLOT — Dashboard tổng hợp
    # ════════════════════════════════════════════════════

    def plot(self, filename: str = None):
        """
        Plot all metric groups into one dashboard.
        Dynamic grid layout: only plot groups with data.
        """
        if not self.steps:
            return

        # ─── Lọc groups có data ───
        active_groups = []
        for group in self.metric_groups:
            has_data = any(
                self._has_valid_data(m.key) for m in group.metrics
            )
            if has_data:
                active_groups.append(group)

        if not active_groups:
            return

        # ─── Grid layout  ───
        n = len(active_groups)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols

        fig = plt.figure(figsize=(7 * cols, 4.5 * rows))
        fig.suptitle(
            f'{self.agent_name} — Training Dashboard',
            fontsize=14, fontweight='bold', y=0.98
        )

        gs = gridspec.GridSpec(rows, cols, figure=fig,
                                hspace=0.35, wspace=0.3,
                                top=0.93, bottom=0.06,
                                left=0.06, right=0.97)

        for i, group in enumerate(active_groups):
            ax = fig.add_subplot(gs[i // cols, i % cols])
            self._plot_group(ax, group)

        # ─── Info box right corner ───
        info = self._build_info_text()
        fig.text(0.98, 0.02, info, fontsize=7, va='bottom', ha='right',
                 family='monospace', alpha=0.5,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray',
                           alpha=0.3))

        # ─── Save ───
        if filename is None:
            filename = f"{self.agent_name}_dashboard.png"
        filepath = self.save_dir / filename
        fig.savefig(filepath, dpi=130, bbox_inches='tight')
        plt.close(fig)

        # Also save JSON
        self._save_json()

    def _plot_group(self, ax, group: MetricGroup):
        """Plot 1 metric group into 1 subplot."""
        ax.set_title(group.title, fontsize=10, fontweight='bold')
        plotted_any = False

        for m in group.metrics:
            if not self._has_valid_data(m.key):
                continue

            steps_clean, values_clean = self._get_clean_data(m.key)
            if len(steps_clean) == 0:
                continue

            plotted_any = True

            # Raw data (nhạt)
            ax.plot(steps_clean, values_clean, alpha=0.2,
                    color=m.color, linewidth=0.8)

            # Smoothed
            if m.smooth_alpha > 0 and len(values_clean) > 5:
                if m.rolling_window > 0:
                    smoothed = self._rolling_avg(values_clean, m.rolling_window)
                    ax.plot(steps_clean[:len(smoothed)], smoothed,
                            color=m.color, linewidth=1.8, label=m.title)
                else:
                    alpha = m.smooth_alpha or self.default_smooth_alpha
                    smoothed = self._ema(values_clean, alpha)
                    ax.plot(steps_clean, smoothed,
                            color=m.color, linewidth=1.8, label=m.title)
            else:
                ax.plot(steps_clean, values_clean,
                        color=m.color, linewidth=1.5, label=m.title)

            # Horizontal lines
            for y_val, label in m.hlines.items():
                ax.axhline(y=y_val, color='gray', linestyle='--',
                           alpha=0.4, linewidth=0.8)

        if not plotted_any:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(group.title, fontsize=10, color='gray')
            return

        # ─── Axes formatting ───
        # Y-scale: dùng scale của metric đầu tiên có data
        for m in group.metrics:
            if self._has_valid_data(m.key):
                if m.yscale == 'log':
                    ax.set_yscale('log', nonpositive='clip')
                break

        ax.set_xlabel('Learn Step', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc='best')

        # Phase markers
        for step, phase in self.cycle_markers:
            ax.axvline(x=step, color='gray', alpha=0.15, linewidth=0.5)

    # ════════════════════════════════════════════════════
    # DATA HELPERS
    # ════════════════════════════════════════════════════

    def _has_valid_data(self, key: str) -> bool:
        """Metric có ít nhất 1 giá trị hợp lệ?"""
        if key not in self.data:
            return False
        return any(v is not None for v in self.data[key])

    def _get_clean_data(self, key: str) -> Tuple[list, list]:
        """Lấy (steps, values) bỏ qua None."""
        if key not in self.data:
            return [], []

        raw = self.data[key]
        steps_clean = []
        values_clean = []

        for i, v in enumerate(raw):
            if v is not None and i < len(self.steps):
                steps_clean.append(self.steps[i])
                values_clean.append(v)

        return steps_clean, values_clean

    def _build_info_text(self) -> str:
        """Build info text hiển thị góc dashboard."""
        elapsed = time.time() - self.start_time
        lines = [
            f"Agent: {self.agent_name}",
            f"Steps: {self.step_counter}",
            f"Time: {elapsed/60:.1f} min",
            f"Groups: {len(self.metric_groups)}",
        ]

        if self.cycle_markers:
            last_phase = self.cycle_markers[-1][1]
            lines.append(f"Phase: {last_phase}")

        return '\n'.join(lines)

    # ════════════════════════════════════════════════════
    # SMOOTHING
    # ════════════════════════════════════════════════════

    @staticmethod
    def _ema(data: list, alpha: float) -> list:
        if not data:
            return []
        result = [data[0]]
        for val in data[1:]:
            result.append(alpha * val + (1 - alpha) * result[-1])
        return result

    @staticmethod
    def _rolling_avg(data: list, window: int) -> list:
        if len(data) < window:
            return data
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode='valid').tolist()

    # ════════════════════════════════════════════════════
    # SAVE / LOAD
    # ════════════════════════════════════════════════════

    def _save_json(self):
        filepath = self.save_dir / f"{self.agent_name}_data.json"

        serializable = {
            'agent_name': self.agent_name,
            'steps': self.steps,
            'cycle_markers': self.cycle_markers,
            'data': {},
            'elapsed': time.time() - self.start_time,
        }

        for key, values in self.data.items():
            serializable['data'][key] = [
                float(v) if v is not None else None for v in values
            ]

        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)

    @classmethod
    def load(cls, filepath: str, metric_groups: List[MetricGroup] = None):
        """Load dashboard từ JSON để vẽ lại."""
        with open(filepath) as f:
            raw = json.load(f)

        if metric_groups is None:
            metric_groups = PPOMetricGroups

        dashboard = cls(
            agent_name=raw['agent_name'],
            metric_groups=metric_groups,
            plot_every=0,
        )
        dashboard.steps = raw['steps']
        dashboard.cycle_markers = [tuple(m) for m in raw.get('cycle_markers', [])]
        dashboard.data = defaultdict(list, raw.get('data', {}))
        dashboard.step_counter = len(dashboard.steps)
        return dashboard

    def save(self):
        """Manual save."""
        self._save_json()

    # ════════════════════════════════════════════════════
    # FACTORY: Tạo từ architecture name
    # ════════════════════════════════════════════════════

    @classmethod
    def create(
        cls,
        architecture: str,
        agent_name: str,
        save_dir: str = "data/metrics/",
        plot_every: int = 50,
        extra_groups: List[MetricGroup] = None,
    ) -> 'TrainingDashboard':
        """
        Factory: tạo dashboard từ tên kiến trúc.

        Args:
            architecture: 'ppo', 'scaffold', 'gru', 'residual'
            agent_name: tên agent
            extra_groups: thêm metric groups custom
        """
        config_map = {
            'ppo': PPOMetricGroups,
            'scaffold': SCAFFOLDFedRepMetricGroups,
            'gru': GRUSeqMetricGroups,
            'residual': ResidualMetricGroups,
        }

        groups = list(config_map.get(architecture, PPOMetricGroups))

        if extra_groups:
            groups.extend(extra_groups)

        return cls(
            agent_name=agent_name,
            metric_groups=groups,
            save_dir=save_dir,
            plot_every=plot_every,
        )
