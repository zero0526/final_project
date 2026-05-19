import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import os
import csv
import logging
from matrix_source.configs.configs import cfg

class MetricsAggregator:
    def __init__(self):
        self.history = defaultdict(list)
        self.episode_count = 0
        self._setup_logger()
        self.history["zeta_lower"] = []
        self.history["zeta_upper"] = []
        self.history["avg_backlog_drift"] = []
        self.history["completion_rate"] = []
        
        # Q-Value History
        self.history["upper_q_min"] = []
        self.history["upper_q_max"] = []
        self.history["upper_q_mean"] = []
        self.history["lower_q_min"] = []
        self.history["lower_q_max"] = []
        self.history["lower_q_mean"] = []
        
        self.reset_episode()

    def _setup_logger(self):
        """Sets up a logger that writes to both terminal and a file."""
        log_dir = cfg.logs
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, "metrics.log")
        
        self.logger = logging.getLogger("MetricsAggregator")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers if any
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        # File handler
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(ch)

    def log(self, message):
        """Logs a message to both console and file."""
        self.logger.info(message)

    def reset_episode(self):
        """Resets the accumulators for a new episode."""
        self.episode_upper_rewards = []
        self.episode_lower_rewards = []
        self.episode_remaining_tasks = []
        
        # Info metrics
        self.episode_energy = []
        self.episode_violations = [] # Added for clarity
        self.episode_success_qos = []
        self.episode_violate_qos = []
        self.episode_backlog_drift = [] # Track backlog drift over time

        # Task Statistics Counters
        self.eps_assigned = 0
        self.eps_failed = 0
        self.last_remaining = 0

        # Node-Service Matrices Tracking
        self.eps_f_alloc = []
        self.eps_arrivals = []
        self.eps_backlog = []

        # Training Losses
        self.episode_upper_mf_losses = []
        self.episode_lower_mf_losses = []
        self.episode_upper_td_losses = []
        self.episode_lower_td_losses = []

        # State Tracking for Normalization Analysis
        self.episode_upper_states = []
        self.episode_lower_states = []
        
        self.curr_zeta_lower = 1.0
        self.curr_zeta_upper = 1.0
        
        # Q-Stats buffers
        self.episode_upper_q_min = []
        self.episode_upper_q_max = []
        self.episode_upper_q_mean = []
        self.episode_lower_q_min = []
        self.episode_lower_q_max = []
        self.episode_lower_q_mean = []
        
        # Offloading flow matrix: [SourceNode][TargetNode] -> count
        self.episode_offloading_matrix = defaultdict(lambda: defaultdict(int))
        
        # Node-Service Matrices Tracking
        self.eps_f_alloc = []
        self.eps_arrivals = []
        self.eps_backlog = []

    def add_upper(self, step_output, mf_loss=None, state=None):
        """Adds data from an upper-level step."""
        self.episode_upper_rewards.append(step_output.get("reward_global", 0))
        if mf_loss is not None:
            self.episode_upper_mf_losses.append(float(mf_loss))
        if state is not None:
            if hasattr(state, "detach"):
                s_np = state.detach().cpu().numpy()
            else:
                s_np = np.array(state)
            self.episode_upper_states.append(s_np.flatten())

    def add_lower(self, step_output, mf_loss=None, state=None):
        """Adds data from a lower-level step."""
        self.episode_lower_rewards.append(step_output.get("reward", 0))
        if mf_loss is not None:
            self.episode_lower_mf_losses.append(float(mf_loss))
        if state is not None:
            if hasattr(state, "detach"):
                s_np = state.detach().cpu().numpy()
            else:
                s_np = np.array(state)
            self.episode_lower_states.append(s_np.flatten())
            
        info = step_output.get("info", {})
        obs = step_output.get("obs", {})
        if obs:
            backlog_drift = obs.get("total_drift", 0)
            self.episode_backlog_drift.append(backlog_drift)
            
        energy_dist = step_output.get("energy", {})
        if energy_dist:
            self.episode_energy.append(energy_dist)

        # Violations count
        violations = step_output.get("violations", 0)
        self.episode_violations.append(violations)

        success_qos = info.get("success_qos", {})
        if success_qos:
            self.episode_success_qos.append(sum(np.sum(v) for v in success_qos.values()))
            
        violate_qos = info.get("violate_qos", {})
        if violate_qos:
            self.episode_violate_qos.append(sum(np.sum(v) for v in violate_qos.values()))
            
        # Accumulate task stats
        self.eps_assigned += info.get("num_tasks", 0)
        self.eps_failed += info.get("immediate_fails", 0) + info.get("expired_count", 0)
        self.last_remaining = info.get("remaining", 0)

    def add_step_matrices(self, f_alloc, arrivals, backlog):
        """Accumulates (Node x Service) matrices for averaging at end of episode."""
        if hasattr(f_alloc, "detach"): f_alloc = f_alloc.detach().cpu().numpy()
        if hasattr(arrivals, "detach"): arrivals = arrivals.detach().cpu().numpy()
        if hasattr(backlog, "detach"): backlog = backlog.detach().cpu().numpy()
        
        self.eps_f_alloc.append(f_alloc)
        self.eps_arrivals.append(arrivals)
        self.eps_backlog.append(backlog)

    def record_td_losses(self, upper_losses=None, lower_losses=None):
        """Records TD losses at their respective timescales (e.g. per-slot for lower, per-frame for upper)."""
        if upper_losses is not None:
            if isinstance(upper_losses, list) and len(upper_losses) > 0:
                self.episode_upper_td_losses.append(np.mean(upper_losses))
            elif isinstance(upper_losses, (float, int)):
                self.episode_upper_td_losses.append(float(upper_losses))
                
        if lower_losses is not None:
            if isinstance(lower_losses, list) and len(lower_losses) > 0:
                self.episode_lower_td_losses.append(np.mean(lower_losses))
            elif isinstance(lower_losses, (float, int)):
                self.episode_lower_td_losses.append(float(lower_losses))

    def record_zeta(self, lower, upper):
        self.curr_zeta_lower = lower
        self.curr_zeta_upper = upper
        
    def record_q_stats(self, node_type, q_min, q_max, q_mean):
        """Records Q-value statistics for the specified node type."""
        if node_type == "Edge_Group": # Upper
            self.episode_upper_q_min.append(q_min)
            self.episode_upper_q_max.append(q_max)
            self.episode_upper_q_mean.append(q_mean)
        elif node_type == "Terminal_Group": # Lower
            self.episode_lower_q_min.append(q_min)
            self.episode_lower_q_max.append(q_max)
            self.episode_lower_q_mean.append(q_mean)

    def store_history(self):
        """Saves episode averages to history and RESETS intra-episode data."""
        self.history["upper_reward"].append(np.sum(self.episode_upper_rewards))
        self.history["lower_reward"].append(np.sum(self.episode_lower_rewards))
        self.history["total_reward"].append(np.sum(self.episode_upper_rewards) + np.sum(self.episode_lower_rewards))
        
        self.history["total_energy"].append(np.sum(self.episode_energy) if self.episode_energy else 0)
        self.history["total_violations"].append(np.sum(self.episode_violations) if self.episode_violations else 0)
        
        self.history["total_success_qos"].append(np.sum(self.episode_success_qos) if self.episode_success_qos else 0)
        self.history["total_violate_qos"].append(np.sum(self.episode_violate_qos) if self.episode_violate_qos else 0)
        
        # Training Losses
        self.history["avg_upper_mf_loss"].append(np.mean(self.episode_upper_mf_losses) if self.episode_upper_mf_losses else 0)
        self.history["avg_lower_mf_loss"].append(np.mean(self.episode_lower_mf_losses) if self.episode_lower_mf_losses else 0)
        self.history["avg_upper_td_loss"].append(np.mean(self.episode_upper_td_losses) if self.episode_upper_td_losses else 0)
        self.history["avg_lower_td_loss"].append(np.mean(self.episode_lower_td_losses) if self.episode_lower_td_losses else 0)

        self.history["zeta_lower"].append(self.curr_zeta_lower)
        self.history["zeta_upper"].append(self.curr_zeta_upper)

        # Q-Value History Averages
        self.history["upper_q_min"].append(np.mean(self.episode_upper_q_min) if self.episode_upper_q_min else 0)
        self.history["upper_q_max"].append(np.mean(self.episode_upper_q_max) if self.episode_upper_q_max else 0)
        self.history["upper_q_mean"].append(np.mean(self.episode_upper_q_mean) if self.episode_upper_q_mean else 0)
        self.history["lower_q_min"].append(np.mean(self.episode_lower_q_min) if self.episode_lower_q_min else 0)
        self.history["lower_q_max"].append(np.mean(self.episode_lower_q_max) if self.episode_lower_q_max else 0)
        self.history["lower_q_mean"].append(np.mean(self.episode_lower_q_mean) if self.episode_lower_q_mean else 0)

        # QoS Success Rate
        success = np.sum(self.episode_success_qos) if self.episode_success_qos else 0
        violate = np.sum(self.episode_violate_qos) if self.episode_violate_qos else 0
        qos_success_rate = success / (success + violate) if (success + violate) > 0 else 0
        self.history["qos_success_rate"].append(qos_success_rate)
        
        # Completion Rate (vs Assigned)
        total_completed = self.eps_assigned - self.eps_failed - self.last_remaining
        completion_rate = (total_completed / self.eps_assigned) if self.eps_assigned > 0 else 0
        self.history["completion_rate"].append(completion_rate)
        
        self.history["avg_backlog_drift"].append(np.mean(self.episode_backlog_drift) if self.episode_backlog_drift else 0)
        
        self.history["avg_remaining_tasks"].append(np.mean(self.episode_remaining_tasks) if self.episode_remaining_tasks else 0)

        # Keep old qos_rate for backward compatibility if needed, but we focus on success rate
        qos_rate = success / (violate if violate > 0 else 1.0)
        self.history["qos_rate"].append(qos_rate)
        
        self.episode_count += 1

        # Auto-plot every 50 episodes
        if self.episode_count % 50 == 0:
            self.plot_history(ep=self.episode_count)
            self.save_history_csv()

    def report_episode(self, ep, success_counts=None, failure_counts=None):
        """Prints a summary of the current episode."""
        upper_reward = np.mean(self.episode_upper_rewards) if self.episode_upper_rewards else 0
        lower_reward = np.mean(self.episode_lower_rewards) if self.episode_lower_rewards else 0
        total_reward = self.history["total_reward"][-1] if self.history["total_reward"] else 0
        energy = self.history["total_energy"][-1] if self.history["total_energy"] else 0
        qos_success_rate = self.history["qos_success_rate"][-1] if self.history["qos_success_rate"] else 0
        
        self.log(f"\n--- Episode {ep} Summary ---")
        self.log(f"Avg Upper Reward: {upper_reward:.4f}")
        self.log(f"Avg Lower Reward: {lower_reward:.4f}")
        self.log(f"Total Reward:     {total_reward:.2f}")
        self.log(f"Total Energy:     {energy:.4f} J")
        self.log(f"QoS Success Rate: {qos_success_rate:.2%}")
        self.log(f"Avg Remaining Tasks: {self.history['avg_remaining_tasks'][-1] if self.history['avg_remaining_tasks'] else 0:.2f}")

        # State statistics reporting
        # if self.episode_upper_states or self.episode_lower_states:
        #     self.log("\n--- Input State Statistics (Mean ± Std) ---")
        #     self._print_state_stats("Upper Agents", self.episode_upper_states)
        #     self._print_state_stats("Lower Agents", self.episode_lower_states)

        if success_counts:
            self.log("\n--- Successful Tasks count per Node and Service ---")
            self._print_per_node_table(success_counts)

        if failure_counts:
            self.log("\n--- Failed Tasks count (Dropped/Overdue) per Node and Service ---")
            self._print_per_node_table(failure_counts)

        if self.episode_offloading_matrix:
            self.log("\n--- Offloading Traffic Matrix (Source -> Target) ---")
            self._print_traffic_matrix()

        # Added Combined Node-Service Average Matrix Log
        # if self.eps_f_alloc:
        #     self.log("\n" + "="*120)
        #     self.log("         NODE-SERVICE RESOURCE ATTRIBUTION (Avg: f_alloc | Arrival | Backlog)")
        #     self.log("="*120)
        #
        #     avg_f = np.mean(self.eps_f_alloc, axis=0)
        #     avg_arr = np.mean(self.eps_arrivals, axis=0)
        #     avg_back = np.mean(self.eps_backlog, axis=0)
        #
        #     self._print_combined_node_service_matrix(avg_f, avg_arr, avg_back)
        #     self.log("="*120 + "\n")

        # Task Completion Summary
        if self.eps_assigned > 0:
            total_completed = self.eps_assigned - self.eps_failed - self.last_remaining
            completion_rate = (total_completed / self.eps_assigned) * 100
            
            # QoS Success Rate from actual success/violation counts
            success_total = np.sum(self.episode_success_qos) if self.episode_success_qos else 0
            violate_total = np.sum(self.episode_violate_qos) if self.episode_violate_qos else 0
            qos_rate = (success_total / (success_total + violate_total) * 100) if (success_total + violate_total) > 0 else 0

            self.log("\n" + "="*50)
            self.log("         EPISODE EXECUTION SUMMARY")
            self.log("="*50)
            self.log(f" Total Tasks Assigned   : {self.eps_assigned}")
            self.log(f" Total Tasks Failed     : {self.eps_failed}")
            self.log(f" Tasks Remaining (Queue): {self.last_remaining}")
            self.log(f" Total Tasks Completed  : {total_completed}")
            self.log("-" * 50)
            self.log(f" Completion Rate (vs Assigned): {completion_rate:.2f}%")
            self.log(f" QoS Success Rate (vs Proc):   {qos_rate:.2f}%")
            self.log("="*50 + "\n")

        self.log("---------------------------\n")

        # Reset episode data after reporting
        self.reset_episode()

    def _print_per_node_table(self, data_dict, is_delay=False):
        """Helper to print per-node per-service tables."""
        # Find the number of services
        sample_val = next(iter(data_dict.values()))
        num_services = len(sample_val) if isinstance(sample_val, (list, np.ndarray)) else len(sample_val)
        
        header = "Node ID | " + " | ".join([f"Svc {i}" for i in range(num_services)])
        self.log(header)
        self.log("-" * len(header))

        for nid in sorted(data_dict.keys()):
            values = data_dict[nid]
            row = f"{nid:<7} | "
            fmt_values = []
            for sid in range(num_services):
                if is_delay:
                    val = np.mean(values[sid]) if (isinstance(values[sid], (list, deque)) and len(values[sid]) > 0) else 0.0
                    fmt_values.append(f"{val:7.4f}")
                else:
                    # For counts
                    val = values[sid]
                    fmt_values.append(f"{int(val):7}")
            self.log(row + " | ".join(fmt_values))

    def _print_combined_node_service_matrix(self, f_matrix, arr_matrix, back_matrix):
        """Prints a Node x Service matrix where each cell is f_alloc | Arrival | Backlog."""
        num_nodes, num_services = f_matrix.shape
        # Header with service IDs
        header = "Node | " + "             | ".join([f"Svc {i:<2}" for i in range(num_services)])
        self.log(header)
        self.log("-" * len(header))
        
        for n in range(num_nodes):
            row = f"N{n:<3} | "
            fmt_cells = []
            for s in range(num_services):
                # Format: f_alloc | Arrival | Backlog
                cell = f"{f_matrix[n, s]:5.1f}|{arr_matrix[n, s]:4.1f}|{back_matrix[n, s]:5.1f}"
                fmt_cells.append(cell)
            self.log(row + " | ".join(fmt_cells))

    def _print_state_stats(self, label, states):
        """Calculates and prints mean/std per feature from a list of flattened states."""
        if not states:
            self.log(f"{label}: No state data.")
            return

        # Stack into [Samples, Features]
        states_matrix = np.stack(states)
        means = np.mean(states_matrix, axis=0)
        stds = np.std(states_matrix, axis=0)
        mins = np.min(states_matrix, axis=0)
        maxs = np.max(states_matrix, axis=0)

        self.log(f"\n[{label}] Feature distribution (Samples: {len(states)}):")
        header = f"{'Feat':<5} | {'Mean ± Std':<20} | {'Range [Min, Max]':<25}"
        self.log(header)
        self.log("-" * len(header))

        for i in range(len(means)):
            stat_str = f"{means[i]:8.3f} ± {stds[i]:8.3f}"
            range_str = f"[{mins[i]:9.3f}, {maxs[i]:9.3f}]"
            self.log(f"{i:<5} | {stat_str:<20} | {range_str:<25}")

    def _print_traffic_matrix(self):
        """Prints the [SourceNode][TargetNode] count matrix."""
        # Get all source nodes and target nodes encountered
        sources = sorted(self.episode_offloading_matrix.keys())
        all_targets = set()
        for s in sources:
            all_targets.update(self.episode_offloading_matrix[s].keys())
        targets = sorted(list(all_targets), key=lambda x: (len(x), x)) # Sort targets: N1, N2, N10...

        header = "Src \\ Tgt | " + " | ".join([f"{t:<4}" for t in targets])
        self.log(header)
        self.log("-" * len(header))

        for s in sources:
            row = f"{s:<9} | "
            counts = []
            for t in targets:
                counts.append(f"{self.episode_offloading_matrix[s][t]:4}")
            self.log(row + " | ".join(counts))

    def _moving_average(self, data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    def plot_history(self, save_dir=cfg.plot_dir, ep=None):
        """Generates and saves performance charts showing evolution."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        episodes = range(1, len(self.history["total_reward"]) + 1)
        window = 10 # Window for smoothing
        
        plt.figure(figsize=(24, 18))
        plt.suptitle(f"Model Evolution Over Episodes (up to {len(episodes)})", fontsize=20)
        
        # Plot 1: Rewards
        plt.subplot(3, 4, 1)
        plt.plot(episodes, self.history["total_reward"], alpha=0.3, color='blue', label="Raw Total")
        if len(episodes) >= window:
            ma = self._moving_average(self.history["total_reward"], window)
            plt.plot(range(window, len(self.history["total_reward"]) + 1), ma, color='blue', label=f"MA-{window}")
        plt.title("Reward Evolution")
        plt.xlabel("Episode")
        plt.legend()
        
        # Plot 2: Energy
        plt.subplot(3, 4, 2)
        plt.plot(episodes, self.history["total_energy"], alpha=0.3, color='orange')
        if len(episodes) >= window:
            ma = self._moving_average(self.history["total_energy"], window)
            plt.plot(range(window, len(self.history["total_energy"]) + 1), ma, color='orange', label=f"MA-{window}")
        plt.title("Energy Consumption Trend")
        plt.xlabel("Episode")
        
        # Plot 3: QoS Success Rate
        plt.subplot(3, 4, 3)
        plt.plot(episodes, self.history["qos_success_rate"], alpha=0.3, color='purple')
        if len(episodes) >= window:
            ma = self._moving_average(self.history["qos_success_rate"], window)
            plt.plot(range(window, len(self.history["qos_success_rate"]) + 1), ma, color='purple', linewidth=2)
        # plt.ylim(0, 1.05) # Removed for auto-scaling
        plt.title("QoS Success Rate (vs Processed)")
        plt.xlabel("Episode")
 
        # Plot 4: Remaining Tasks
        plt.subplot(3, 4, 4)
        plt.plot(episodes, self.history["avg_remaining_tasks"], alpha=0.3, color='brown')
        if len(episodes) >= window:
            ma = self._moving_average(self.history["avg_remaining_tasks"], window)
            plt.plot(range(window, len(self.history["avg_remaining_tasks"]) + 1), ma, color='brown')
        plt.title("Task Clearing Efficiency")
        plt.xlabel("Episode")
 
        # Plot 5: MF Training Losses
        plt.subplot(3, 4, 5)
        plt.plot(episodes, self.history["avg_upper_mf_loss"], alpha=0.3, color='cyan', label="Upper")
        plt.plot(episodes, self.history["avg_lower_mf_loss"], alpha=0.3, color='magenta', label="Lower")
        if len(episodes) >= window:
            if len(self.history["avg_upper_mf_loss"]) >= window:
                ma_u = self._moving_average(self.history["avg_upper_mf_loss"], window)
                plt.plot(range(window, len(self.history["avg_upper_mf_loss"]) + 1), ma_u, color='cyan')
            if len(self.history["avg_lower_mf_loss"]) >= window:
                ma_l = self._moving_average(self.history["avg_lower_mf_loss"], window)
                plt.plot(range(window, len(self.history["avg_lower_mf_loss"]) + 1), ma_l, color='magenta')
        plt.yscale('log')
        plt.title("Mean Field Loss (Log)")
        plt.xlabel("Episode")
        plt.legend()
 
        # Plot 6: TD Training Losses (Q-Network)
        plt.subplot(3, 4, 6)
        plt.plot(episodes, self.history["avg_upper_td_loss"], alpha=0.3, color='teal', label="Upper")
        plt.plot(episodes, self.history["avg_lower_td_loss"], alpha=0.3, color='olive', label="Lower")
        if len(episodes) >= window:
            if len(self.history["avg_upper_td_loss"]) >= window:
                ma_u = self._moving_average(self.history["avg_upper_td_loss"], window)
                plt.plot(range(window, len(self.history["avg_upper_td_loss"]) + 1), ma_u, color='teal')
            if len(self.history["avg_lower_td_loss"]) >= window:
                ma_l = self._moving_average(self.history["avg_lower_td_loss"], window)
                plt.plot(range(window, len(self.history["avg_lower_td_loss"]) + 1), ma_l, color='olive')
        plt.yscale('log')
        plt.title("TD Loss (Log)")
        plt.xlabel("Episode")
        plt.legend()

        # Plot 7: Upper Q-Values
        plt.subplot(3, 4, 7)
        plt.plot(episodes, self.history["upper_q_min"], alpha=0.3, color='blue', label="Min")
        plt.plot(episodes, self.history["upper_q_max"], alpha=0.3, color='red', label="Max")
        plt.plot(episodes, self.history["upper_q_mean"], alpha=0.8, color='green', label="Mean")
        plt.title("Upper Q-Value Stats")
        plt.xlabel("Episode")
        plt.legend()

        # Plot 8: Lower Q-Values
        plt.subplot(3, 4, 8)
        plt.plot(episodes, self.history["lower_q_min"], alpha=0.3, color='blue', label="Min")
        plt.plot(episodes, self.history["lower_q_max"], alpha=0.3, color='red', label="Max")
        plt.plot(episodes, self.history["lower_q_mean"], alpha=0.8, color='green', label="Mean")
        plt.title("Lower Q-Value Stats")
        plt.xlabel("Episode")
        plt.legend()

        # Plot 9: Backlog Drift
        plt.subplot(3, 4, 9)
        plt.plot(episodes, self.history["avg_backlog_drift"], alpha=0.3, color='crimson', label="Raw Drift")
        if len(episodes) >= window:
            ma = self._moving_average(self.history["avg_backlog_drift"], window)
            plt.plot(range(window, len(self.history["avg_backlog_drift"]) + 1), ma, color='crimson', label=f"MA-{window}")
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title("Backlog Drift Evolution")
        plt.xlabel("Episode")
        plt.legend()

        # Plot 10: Completion Rate (vs Assigned)
        plt.subplot(3, 4, 10)
        plt.plot(episodes, self.history["completion_rate"], alpha=0.3, color='forestgreen', label="Raw Rate")
        if len(episodes) >= window:
            ma = self._moving_average(self.history["completion_rate"], window)
            plt.plot(range(window, len(self.history["completion_rate"]) + 1), ma, color='forestgreen', linewidth=2, label=f"MA-{window}")
        # plt.ylim(0, 1.05) # Removed for auto-scaling
        plt.title("Completion Rate (vs Assigned)")
        plt.xlabel("Episode")
        plt.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save latest
        plt.savefig(os.path.join(save_dir, "training_metrics.png"))
        
        # Save archival copy
        if ep is not None:
            archive_dir = os.path.join(save_dir, "archive")
            if not os.path.exists(archive_dir):
                os.makedirs(archive_dir)
            plt.savefig(os.path.join(archive_dir, f"metrics_ep_{ep}.png"))
            
        plt.close()

    def plot_state_distributions(self, save_dir=cfg.plot_dir, ep=None):
        """Generates boxplots for Upper and Lower agent input features."""
        if not self.episode_upper_states and not self.episode_lower_states:
            return
            
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        plt.figure(figsize=(20, 10))
        plt.suptitle(f"Agent Input Feature Distributions (Episode {self.episode_count})", fontsize=20)
        
        # Plot Upper Agents
        plt.subplot(2, 1, 1)
        if self.episode_upper_states:
            data = np.stack(self.episode_upper_states)
            plt.boxplot(data, vert=True, patch_artist=True)
            plt.title(f"Upper Agent State Features (Samples: {len(self.episode_upper_states)})")
            plt.ylabel("Value Range")
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, "No data for Upper Agents", ha='center')

        # Plot Lower Agents
        plt.subplot(2, 1, 2)
        if self.episode_lower_states:
            data = np.stack(self.episode_lower_states)
            plt.boxplot(data, vert=True, patch_artist=True)
            plt.title(f"Lower Agent State Features (Samples: {len(self.episode_lower_states)})")
            plt.ylabel("Value Range")
            plt.xlabel("Feature Index")
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, "No data for Lower Agents", ha='center')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save latest
        plt.savefig(os.path.join(save_dir, "state_distributions.png"))
        
        # Save archival copy
        if ep is not None:
            archive_dir = os.path.join(save_dir, "archive")
            if not os.path.exists(archive_dir):
                os.makedirs(archive_dir)
            plt.savefig(os.path.join(archive_dir, f"state_dist_ep_{ep}.png"))
            
        plt.close()

    def save_history_csv(self, save_dir=cfg.logs, filename="training_metrics.csv"):
        """Saves the entire history to a CSV file for model comparison."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        file_path = os.path.join(save_dir, filename)
        
        # Determine headers from history keys
        keys = sorted(self.history.keys())
        num_episodes = len(self.history["total_reward"])
        
        try:
            with open(file_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Header: Episode + History Keys
                writer.writerow(["episode"] + keys)
                
                # Rows: Episode 1..N
                for i in range(num_episodes):
                    row = [i + 1]
                    for k in keys:
                        # Handle potential length mismatches (shouldn't happen but for safety)
                        if i < len(self.history[k]):
                            val = self.history[k][i]
                            # Format floats for readability
                            if isinstance(val, (float, np.float32, np.float64, np.ndarray)):
                                # Use higher precision for CSV
                                if isinstance(val, np.ndarray):
                                    row.append(str(val.tolist()))
                                else:
                                    row.append(f"{val:.8f}")
                            else:
                                row.append(val)
                        else:
                            row.append("")
                    writer.writerow(row)
            self.log(f"Successfully exported training history to {file_path}")
        except Exception as e:
            self.log(f"Error exporting CSV: {e}")

