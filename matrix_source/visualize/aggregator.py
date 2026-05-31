import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import os
import csv
import logging
from matrix_source.configs.configs import cfg
import torch

class MetricsAggregator:
    def __init__(self, name="metrics"):
        self.history = defaultdict(list)
        self.episode_count = 0
        self.name = name
        self._setup_logger()
        self.reset_episode()

    def _setup_logger(self):
        """Sets up a logger that writes to both terminal and a file."""
        log_dir = cfg.logs
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, "metrics.log")
        
        self.logger = logging.getLogger("MetricsAggregator")
        self.logger.setLevel(logging.INFO)
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(ch)

    def log(self, message):
        self.logger.info(message)

    def reset_episode(self):
        """Resets the accumulators for a new episode."""
        # Performance: Use lists to collect tensors lazily
        self.episode_upper_rewards = []
        self.episode_lower_rewards = []
        self.episode_energy = []
        self.episode_violations = []
        self.episode_success_qos = []
        self.episode_violate_qos = []
        self.episode_backlog_drift = []
        self.episode_remaining_tasks = []
        self.episode_assigned_list = []
        self.episode_failed_list = []
        self.episode_realized_delay = []
        
        # Training/State tracking
        self.episode_upper_mf_losses = []
        self.episode_lower_mf_losses = []
        self.episode_upper_td_losses = []
        self.episode_lower_td_losses = []
        self.episode_upper_states = []
        self.episode_lower_states = []
        
        # Q-Stats
        self.episode_upper_q_min = []
        self.episode_upper_q_max = []
        self.episode_upper_q_mean = []
        self.episode_lower_q_min = []
        self.episode_lower_q_max = []
        self.episode_lower_q_mean = []

        self.episode_terminal_fails = None
        
        self.episode_step_fail_reasons = [] # Dicts of tensors
        self.curr_zeta_lower = 1.0
        self.curr_zeta_upper = 1.0
        
        # Matrix tracking
        self.eps_f_alloc = []
        self.eps_arrivals = []
        self.eps_backlog = []
        
        # Derived stats (cleared each episode)
        self.eps_hw_deficit = None
        self.eps_hw_fail_count = None
        self.episode_fail_reasons = {'deadline': 0, 'hardware': 0, 'queue_full': 0, 'invalid_placement': 0}


    def add_upper(self, step_output, mf_loss=0, state=None):
        if isinstance(step_output, dict):
            self.episode_upper_rewards.append(step_output.get("reward_global", 0))
        elif isinstance(step_output, torch.Tensor):
            self.episode_upper_rewards.append(step_output)
        self.episode_upper_mf_losses.append(mf_loss)
        if state is not None:
            s = state.detach().cpu().numpy() if hasattr(state, "detach") else np.array(state)
            self.episode_upper_states.append(s.flatten())

    def add_lower(self, step_output, mf_loss=0, state=None):
        self.episode_lower_rewards.append(step_output.get("reward", 0.0))
        self.episode_lower_mf_losses.append(mf_loss)
        if state is not None:
            s = state.detach().cpu().numpy() if hasattr(state, "detach") else np.array(state)
            self.episode_lower_states.append(s.flatten())
            
        info = step_output.get("info", {})
        obs = step_output.get("obs", {})
        if obs: self.episode_backlog_drift.append(obs.get("total_drift", 0))
        
        self.episode_energy.append(step_output.get("energy", 0))
        self.episode_violations.append(step_output.get("violations", 0))

        # Lazy summation of QoS vectors
        def _sum(v): return v.float().sum() if isinstance(v, torch.Tensor) else v
        self.episode_success_qos.append(_sum(info.get("success_qos", 0)))
        self.episode_violate_qos.append(_sum(info.get("violate_qos", 0)))
        
        # Task counters
        self.episode_remaining_tasks.append(info.get("remaining", 0))
        self.episode_assigned_list.append(info.get("num_tasks", 0))
        self.episode_failed_list.append(info.get("immediate_fails", 0) + info.get("expired_count", 0))
        
        term_fails = info.get("terminal_fail_counts")
        if term_fails is not None:
            if hasattr(term_fails, "detach"): term_fails = term_fails.detach().cpu().numpy()
            if self.episode_terminal_fails is None:
                self.episode_terminal_fails = np.zeros_like(term_fails)
            self.episode_terminal_fails += term_fails
        
        r_delay = info.get("realized_delay", {})
        if r_delay:
            # Vectorized mean of all delays in the dict (each value is a tensor)
            delays = [torch.mean(v) for v in r_delay.values()]
            self.episode_realized_delay.append(torch.stack(delays).mean())

        # Failure reasons
        reasons = info.get("fail_reasons")
        if reasons: self.episode_step_fail_reasons.append(reasons)

    def add_step_matrices(self, f_alloc, arrivals, backlog):
        self.eps_f_alloc.append(f_alloc)
        self.eps_arrivals.append(arrivals)
        self.eps_backlog.append(backlog)

    def record_td_losses(self, upper_losses=None, lower_losses=None):
        def _extract(x):
            if isinstance(x, dict): x = x.get("loss", 0.0)
            return x.detach() if hasattr(x, "detach") else torch.tensor(float(x))
        if upper_losses is not None:
            self.episode_upper_td_losses.append(_extract(upper_losses))
        if lower_losses is not None:
            self.episode_lower_td_losses.append(_extract(lower_losses))

    def record_zeta(self, lower, upper):
        self.curr_zeta_lower = lower
        self.curr_zeta_upper = upper
        
    def record_q_stats(self, node_type, q_min, q_max, q_mean):
        if node_type == "Edge_Group":
            self.episode_upper_q_min.append(q_min.detach() if hasattr(q_min, "detach") else torch.tensor(q_min))
            self.episode_upper_q_max.append(q_max.detach() if hasattr(q_max, "detach") else torch.tensor(q_max))
            self.episode_upper_q_mean.append(q_mean.detach() if hasattr(q_mean, "detach") else torch.tensor(q_mean))
        else:
            self.episode_lower_q_min.append(q_min.detach() if hasattr(q_min, "detach") else torch.tensor(q_min))
            self.episode_lower_q_max.append(q_max.detach() if hasattr(q_max, "detach") else torch.tensor(q_max))
            self.episode_lower_q_mean.append(q_mean.detach() if hasattr(q_mean, "detach") else torch.tensor(q_mean))

    def store_history(self):
        """Vectorized aggregation on GPU to avoid CPU sync points."""
        def _get_avg(arr, key):
            # Filter None values
            arr = [x for x in arr if x is not None]
            if not arr: return self.history[key][-1] if self.history.get(key) else 0.0
            # Perform stack and mean on device
            t = torch.stack([x if isinstance(x, torch.Tensor) else torch.tensor(float(x), device=self.device if hasattr(self, 'device') else 'cpu') for x in arr])
            return float(t.float().mean().item())

        def _get_sum(arr, key):
            # Filter None values
            arr = [x for x in arr if x is not None]
            if not arr: return 0.0
            t = torch.stack([x if isinstance(x, torch.Tensor) else torch.tensor(float(x), device=self.device if hasattr(self, 'device') else 'cpu') for x in arr])
            return float(t.float().sum().item())

        # Sync Rewards and Energy
        total_u_rw = _get_sum(self.episode_upper_rewards, "u_rw")
        total_l_rw = _get_sum(self.episode_lower_rewards, "l_rw")
        self.history["total_reward"].append(total_u_rw + total_l_rw)
        self.history["total_energy"].append(_get_sum(self.episode_energy, "total_energy"))

        # Training Averages (Vectorized)
        self.history["avg_upper_mf_loss"].append(_get_avg(self.episode_upper_mf_losses, "avg_upper_mf_loss"))
        self.history["avg_lower_mf_loss"].append(_get_avg(self.episode_lower_mf_losses, "avg_lower_mf_loss"))
        self.history["avg_upper_td_loss"].append(_get_avg(self.episode_upper_td_losses, "avg_upper_td_loss"))
        self.history["avg_lower_td_loss"].append(_get_avg(self.episode_lower_td_losses, "avg_lower_td_loss"))
        
        self.history["zeta_lower"].append(self.curr_zeta_lower)
        self.history["zeta_upper"].append(self.curr_zeta_upper)

        for k in ["upper_q_min", "upper_q_max", "upper_q_mean", "lower_q_min", "lower_q_max", "lower_q_mean"]:
            self.history[k].append(_get_avg(getattr(self, f"episode_{k}"), k))

        # QoS and Completion
        success = _get_sum(self.episode_success_qos, "success")
        violate = _get_sum(self.episode_violate_qos, "violate")
        self.history["qos_success_rate"].append(success / (success + violate) if (success + violate) > 0 else 0)

        # Completion rate: tasks successfully processed / all tasks that arrived
        # rem_val = số task còn trong queue cuối episode (chưa xử lý xong)
        if self.episode_remaining_tasks:
            rem_val = float(self.episode_remaining_tasks[-1].item() if hasattr(self.episode_remaining_tasks[-1], "item") else self.episode_remaining_tasks[-1])
        else:
            rem_val = 0.0

        total_resolved = success + violate + rem_val
        self.history["completion_rate"].append(success / total_resolved if total_resolved > 0 else 0.0)
        self.history["avg_backlog_drift"].append(_get_avg(self.episode_backlog_drift, "avg_backlog_drift"))
        self.history["avg_remaining_tasks"].append(_get_avg(self.episode_remaining_tasks, "avg_remaining_tasks"))
        self.history["avg_realized_delay"].append(_get_avg(self.episode_realized_delay, "avg_realized_delay"))
        self.history["total_violations"].append(violate)
        
        # Additional metrics for restored plots
        self.history["qos_rate"].append(success / (violate if violate > 0 else 1.0))

        self.episode_count += 1
        if self.episode_count % 50 == 0:
            self.plot_history(ep=self.episode_count)
            self.save_history_csv()

    def report_episode(self, ep):
        """Prints summary. history[-1] contains processed results for current ep."""
        tr = self.history["total_reward"][-1] if self.history["total_reward"] else 0
        en = self.history["total_energy"][-1] if self.history["total_energy"] else 0
        qos = self.history["qos_success_rate"][-1] if self.history["qos_success_rate"] else 0
        cr = self.history["completion_rate"][-1] if self.history["completion_rate"] else 0
        
        # Lấy số lượng tuyệt đối từ list hiện tại (trước khi reset)
        total_success = sum(self.episode_success_qos)
        total_failed = sum(self.episode_violate_qos)
        
        self.log(f"EP {ep:4d} | Rew: {tr:8.2f} | Energy: {en:8.2f} | QoS: {qos:6.2%} | Comp: {cr:6.2%} | OK: {total_success:4.0f} | FAIL: {total_failed:4.0f}")
        
        if self.episode_terminal_fails is not None and np.sum(self.episode_terminal_fails) > 0:
            pass
            # self.log(f" --- Per-Terminal Failure Breakdown ---")
            # header = " Term ID | Fail Count"
            # self.log(header)
            # self.log("-" * len(header))
            # for tid in range(len(self.episode_terminal_fails)):
            #     if np.any(self.episode_terminal_fails[tid] > 0):
            #         self.log(f" {tid:<7} | {np.sum(self.episode_terminal_fails[tid]):<10.0f}")

        # Log fail reason summary
        if self.episode_step_fail_reasons:
            d_total = sum(float(r.get('deadline', 0)) for r in self.episode_step_fail_reasons)
            h_total = sum(float(r.get('hardware', 0)) for r in self.episode_step_fail_reasons)
            q_total = sum(float(r.get('queue_full', 0)) for r in self.episode_step_fail_reasons)
            p_total = sum(float(r.get('invalid_placement', 0)) for r in self.episode_step_fail_reasons)
            e_total = sum(float(r.get('expired', 0)) for r in self.episode_step_fail_reasons)
            self.log(f" --- Fail Reason Breakdown ---")
            self.log(f"  Deadline exceeded : {d_total:.0f}")
            self.log(f"  Hardware (f_max)  : {h_total:.0f}")
            self.log(f"  Queue full        : {q_total:.0f}")
            self.log(f"  Invalid placement : {p_total:.0f}")
            self.log(f"  Expired in Queue  : {e_total:.0f}")

        self.reset_episode()

    def _moving_average(self, data, window=10):
        if len(data) < window: return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    def plot_history(self, ep=None):
        if not self.history["total_reward"]: return
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        plt.suptitle(f"Training Progress (Episode {len(self.history['total_reward'])})", fontsize=16)
        
        window = 10
        metrics = [
            ("total_reward", "Reward Convergence", "blue"),
            ("avg_backlog_drift", "System Stability (Drift)", "green"),
            ("total_energy", "Energy Consumption", "orange"),
            ("avg_realized_delay", "Delay Evolution", "red"),
            ("qos_success_rate", "QoS Satisfaction", "purple"),
            ("completion_rate", "Task Throughput", "blue")
        ]

        for i, (key, title, color) in enumerate(metrics):
            ax = axes[i // 3, i % 3]
            data = self.history[key]
            
            # Plot raw data with transparency
            ax.plot(data, color=color, alpha=0.3, label="Raw")
            
            # Plot MA-10
            if len(data) >= window:
                ma_data = self._moving_average(data, window)
                ax.plot(range(window-1, len(data)), ma_data, color=color, linewidth=2, label=f"MA-{window}")
            
            ax.set_title(title)
            ax.legend()
            ax.grid(True)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = f"{self.name.lower()}_progress_training.png"
        plt.savefig(os.path.join(cfg.plot_dir, filename))
        plt.close()

    def save_history_csv(self):
        path = os.path.join(cfg.results, "training_history.csv")
        if not os.path.exists(cfg.results): os.makedirs(cfg.results)
        keys = sorted(self.history.keys())
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for i in range(len(self.history[keys[0]])):
                writer.writerow([self.history[k][i] for k in keys])
