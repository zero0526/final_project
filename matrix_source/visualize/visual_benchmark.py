from typing import List, Dict, Optional, Tuple
import numpy as np
import os
import csv
import glob
import matplotlib.pyplot as plt
from matplotlib import cm
METRICS = [
    ("total_reward",       "Total Reward",              None),
    ("qos_success_rate",   "QoS Success Rate",          None),
    ("total_energy",       "Energy Consumption (J)",    None),
    ("avg_backlog_drift",  "Avg Backlog Drift",         None),
    ("completion_rate",    "Completion Rate",           None),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _moving_average(data: List[float], window: int = 20) -> np.ndarray:
    """Smoothed moving average (SMA)."""
    if len(data) < window:
        return np.array(data)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def _exponential_moving_average(data: List[float], span: int = 20) -> np.ndarray:
    """Exponential moving average (EMA)."""
    if not data:
        return np.array([])
    data_arr = np.array(data)
    alpha = 2 / (span + 1)
    ema = np.zeros_like(data_arr)
    ema[0] = data_arr[0]
    for i in range(1, len(data_arr)):
        ema[i] = alpha * data_arr[i] + (1 - alpha) * ema[i - 1]
    return ema


def _load_csv(path: str) -> Dict[str, List[float]]:
    """
    Đọc một file CSV do MetricsAggregator.save_history_csv() tạo ra.
    Trả về dict: {column_name -> [float, ...]}
    """
    result: Dict[str, List[float]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                try:
                    parsed = float(val)
                except (ValueError, TypeError):
                    parsed = 0.0
                result.setdefault(key, []).append(parsed)
    return result


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BenchmarkPlotter:
    """
    So sánh nhiều lần train trên cùng một biểu đồ.

    Workflow:
        plotter = BenchmarkPlotter(smooth_window=20)
        plotter.add_csv("run1/training_metrics.csv", label="PPO-v1")
        plotter.add_csv("run2/training_metrics.csv", label="PPO-v2")
        plotter.plot(save_path="benchmark.png", show=True)
    """

    def __init__(self, smooth_window: int = 20):
        self.smooth_window = smooth_window
        self._runs: List[Tuple[str, Dict[str, List[float]]]] = []  # [(label, data), ...]

    # ------------------------------------------------------------------
    def add_csv(self, csv_path: str, label: Optional[str] = None) -> "BenchmarkPlotter":
        """Thêm một file CSV vào plotter."""
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Không tìm thấy file: {csv_path}")
        data = _load_csv(csv_path)
        lbl = label or os.path.splitext(os.path.basename(csv_path))[0]
        self._runs.append((lbl, data))
        print(f"  [✓] Đã nạp: {lbl}  ({len(data.get('episode', data.get('total_reward', [])))} episodes)")
        return self

    def add_folder(
        self,
        folder: str,
        pattern: str = "**/training_metrics.csv",
        label_from: str = "parent",  # "parent" | "filename" | "folder"
    ) -> "BenchmarkPlotter":
        """
        Tự động tìm tất cả CSV trong một thư mục (tìm đệ quy).

        label_from:
            "parent"   -> dùng tên thư mục cha của file CSV làm label
            "filename" -> dùng tên file (không phần mở rộng)
            "folder"   -> dùng tên thư mục gốc được truyền vào
        """
        paths = sorted(glob.glob(os.path.join(folder, pattern), recursive=True))
        if not paths:
            print(f"  [!] Không tìm thấy file nào khớp '{pattern}' trong: {folder}")
            return self
        for p in paths:
            if label_from == "parent":
                lbl = os.path.basename(os.path.dirname(p))
            elif label_from == "filename":
                lbl = os.path.splitext(os.path.basename(p))[0]
            else:
                lbl = os.path.basename(folder)
            self.add_csv(p, label=lbl)
        return self

    # ------------------------------------------------------------------
    def plot(
        self,
        metrics: Optional[List[Tuple[str, str, Optional[str]]]] = None,
        save_path: Optional[str] = None,
        show: bool = False,
        ncols: int = 3,
        figsize_per_cell: Tuple[int, int] = (7, 4),
        title: str = "Model Comparison",
        show_raw: bool = True,
        show_smooth: bool = True,
        smooth_type: str = "sma",  # "sma" | "ema"
        linewidth_smooth: float = 1.5,
        smooth_overrides: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Vẽ biểu đồ so sánh.

        Parameters
        ----------
        metrics          : list [(col_name, subplot_title, color_hint), ...]
                           Mặc định dùng METRICS toàn cục.
        save_path        : đường dẫn lưu ảnh (PNG). Nếu None chỉ show.
        show             : True để hiện cửa sổ matplotlib.
        ncols            : số cột trong lưới subplot.
        figsize_per_cell : kích thước (width, height) mỗi ô subplot (inch).
        title            : tiêu đề chung của figure.
        show_raw         : hiện đường raw (mờ).
        show_smooth      : hiện đường smoothed (đậm).
        smooth_type      : loại smoothing ("sma" hoặc "ema").
        linewidth_smooth : độ dày đường smoothed.
        smooth_overrides : dict ghi đè cửa sổ smoothing cho từng metric {col_name: window}.
        """
        if not self._runs:
            print("  [!] Chưa có dữ liệu nào được nạp. Dùng add_csv() trước.")
            return

        metrics = metrics or METRICS
        smooth_overrides = smooth_overrides or {}
        n = len(metrics)
        nrows = (n + ncols - 1) // ncols
        fig_w = figsize_per_cell[0] * ncols
        fig_h = figsize_per_cell[1] * nrows + 1  # +1 for suptitle

        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)

        # Bảng màu tự động
        palette = cm.tab10.colors if len(self._runs) <= 10 else cm.tab20.colors
        colors = [palette[i % len(palette)] for i in range(len(self._runs))]

        for idx, (col_name, subplot_title, _) in enumerate(metrics):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]

            ax.set_title(subplot_title, fontsize=11, fontweight="bold")
            ax.set_xlabel("Episode", fontsize=9)
            ax.grid(True, alpha=0.3, linestyle="--")

            for run_idx, (label, data) in enumerate(self._runs):
                color = colors[run_idx]
                values = data.get(col_name, [])
                if not values:
                    continue

                episodes = list(range(1, len(values) + 1))
                sw = smooth_overrides.get(col_name, self.smooth_window)

                # Raw (mờ)
                if show_raw:
                    ax.plot(
                        episodes, values,
                        alpha=0.2, color=color, linewidth=0.8,
                    )

                # Smoothed (đậm)
                if show_smooth:
                    if smooth_type == "ema":
                        sm = _exponential_moving_average(values, sw)
                        ep_sm = episodes
                    else:
                        if len(values) >= sw:
                            sm = _moving_average(values, sw)
                            ep_sm = list(range(sw, len(values) + 1))
                        else:
                            sm, ep_sm = values, episodes

                    ax.plot(
                        ep_sm, sm,
                        color=color, linewidth=linewidth_smooth,
                        label=label,
                    )
                elif not show_raw:
                    # Nếu tắt cả hai thì vẫn nên hiện cái gì đó, mặc định hiện raw đường mảnh
                    ax.plot(episodes, values, color=color, linewidth=0.8, alpha=0.5, label=label)

            ax.legend(fontsize=8, loc="best", framealpha=0.6)

        # Ẩn các ô subplot thừa
        for empty_idx in range(n, nrows * ncols):
            r, c = divmod(empty_idx, ncols)
            axes[r][c].set_visible(False)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  [✓] Đã lưu biểu đồ: {save_path}")

        if show:
            plt.show()

        plt.close(fig)

    # ------------------------------------------------------------------
    def summary_table(self) -> None:
        """In bảng tóm tắt giá trị trung bình (toàn bộ run) cho từng model."""
        cols = [m[0] for m in METRICS]
        col_w = 22

        header = f"{'Model':<30}" + "".join(f"{c:>{col_w}}" for c in cols)
        print("\n" + "=" * len(header))
        print("  BENCHMARK SUMMARY (mean over all episodes)")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        for label, data in self._runs:
            row = f"{label:<30}"
            for col in cols:
                vals = data.get(col, [])
                mean_val = np.mean(vals) if vals else float("nan")
                row += f"{mean_val:>{col_w}.4f}"
            print(row)

        print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# CLI / Demo
# ---------------------------------------------------------------------------

def _demo():
    """
    Ví dụ: tự động tìm tất cả CSV trong thư mục logs và vẽ so sánh.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Visual Benchmark: so sánh nhiều model training")
    parser.add_argument(
        "csvs", nargs="*",
        help="Danh sách đường dẫn CSV (vd: logs/run1.csv logs/run2.csv)"
    )
    parser.add_argument(
        "--dir", type=str, default=None,
        help="Thư mục chứa các CSV (tìm đệ quy theo pattern **/training_metrics.csv)"
    )
    parser.add_argument(
        "--labels", nargs="*", default=None,
        help="Tên nhãn tương ứng với từng file CSV (nếu có)"
    )
    parser.add_argument(
        "--save", type=str, default="benchmark_comparison.png",
        help="Đường dẫn lưu ảnh (mặc định: benchmark_comparison.png)"
    )
    parser.add_argument(
        "--smooth", type=int, default=20,
        help="Cửa sổ moving average / span EMA (mặc định: 20)"
    )
    parser.add_argument(
        "--ema", action="store_true",
        help="Dùng Exponential Moving Average thay vì SMA"
    )
    parser.add_argument(
        "--no-raw", action="store_false", dest="raw",
        help="Không vẽ đường raw dạo động (chỉ hiện đường smoothed)"
    )
    parser.add_argument(
        "--lw", type=float, default=1.2,
        help="Độ dày đường smoothed (mặc định: 1.2)"
    )
    parser.add_argument(
        "--smooth-overrides", type=str, default="",
        help="Ghi đè smoothing cho từng metric, vd: total_energy=50,qos_success_rate=40"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Hiện cửa sổ matplotlib"
    )
    parser.add_argument(
        "--title", type=str, default="Model Comparison",
        help="Tiêu đề biểu đồ"
    )
    args = parser.parse_args()

    plotter = BenchmarkPlotter(smooth_window=args.smooth)

    # Nạp từ thư mục
    if args.dir:
        plotter.add_folder(args.dir)

    # Nạp từ danh sách file
    for i, csv_path in enumerate(args.csvs):
        label = args.labels[i] if (args.labels and i < len(args.labels)) else None
        plotter.add_csv(csv_path, label=label)

    if not plotter._runs:
        print("\n[!] Không có dữ liệu. Dùng: python visual_benchmark.py <file1.csv> <file2.csv>")
        print("    hoặc: python visual_benchmark.py --dir logs/")
        return

    plotter.summary_table()

    # Parse overrides
    overrides = {}
    if args.smooth_overrides:
        for item in args.smooth_overrides.split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                try:
                    overrides[k.strip()] = int(v.strip())
                except ValueError:
                    pass

    plotter.plot(
        save_path=args.save,
        show=args.show,
        title=args.title,
        show_raw=args.raw,
        smooth_type="ema" if args.ema else "sma",
        linewidth_smooth=args.lw,
        smooth_overrides=overrides
    )


if __name__ == "__main__":
    # Dùng CLI mặc định khi chạy file
    _demo()

