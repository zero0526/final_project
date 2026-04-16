import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

class MetricsAggregator:
    def __init__(self, bin_size=300):
        self.bin_size = bin_size
        self.episode_count = 0
        self.buffer = defaultdict(list)
        self.history = defaultdict(list)
        
        # Per-service and per-node history
        self.history["service_delays"] = defaultdict(list)
        self.history["node_delays"] = defaultdict(list)
        
        self.reset_episode()

    def reset_episode(self):
        """Resets the accumulators for a new episode."""
        self.episode_upper_rewards = []
        self.episode_lower_rewards = []
        self.episode_remaining_tasks = []
        
        # Info metrics
        self.episode_f1 = []
        self.episode_energy = []
        self.episode_virtual_delay = []
        self.episode_realized_delay = []
        self.episode_success_qos = []
        self.episode_violate_qos = []
        
        # Detailed delay tracking (step-level)
        self.episode_node_raw_delays = defaultdict(list)
        self.episode_service_raw_delays = defaultdict(list)

    def add_upper(self, step_output):
        """Adds data from an upper-level step."""
        self.episode_upper_rewards.append(step_output.get("reward", 0))
        
        remaining = step_output.get("remaining_task", {})
        if remaining:
            total_remaining = sum(np.sum(v) for v in remaining.values())
            self.episode_remaining_tasks.append(total_remaining)

    def add_lower(self, step_output):
        """Adds data from a lower-level step."""
        self.episode_lower_rewards.append(step_output.get("reward", 0))
        
        info = step_output.get("info", {})
        
        # F1 and Energy
        f1_dist = info.get("f1", {})
        if f1_dist:
            self.episode_f1.append(sum(f1_dist.values()))
            
        energy_dist = info.get("energy", {})
        if energy_dist:
            self.episode_energy.append(sum(energy_dist.values()))
            
        # Realized Delay - Detailed
        r_delay = info.get("realized_delay", {}) # {node_id: np.ndarray(num_services)}
        if r_delay:
            # Overall mean for the episode summary
            self.episode_realized_delay.append(np.mean([np.mean(v) for v in r_delay.values()]))
            
            # Per-node average across services in this step
            for nid, delay_vec in r_delay.items():
                self.episode_node_raw_delays[nid].append(np.mean(delay_vec))
                
            # Per-service average across nodes in this step
            # We assume delay_vec indices match service IDs
            if len(r_delay) > 0:
                # Transpose to get per-service [node1_delay, node2_delay, ...]
                all_delays = np.stack(list(r_delay.values())) # (num_nodes, num_services)
                avg_per_service = np.mean(all_delays, axis=0) # (num_services,)
                for sid, avg_d in enumerate(avg_per_service):
                    self.episode_service_raw_delays[sid].append(avg_d)
            
        # QoS info
        success_qos = info.get("success_qos", {})
        if success_qos:
            self.episode_success_qos.append(sum(np.sum(v) for v in success_qos.values()))
            
        violate_qos = info.get("violate_qos", {})
        if violate_qos:
            self.episode_violate_qos.append(sum(np.sum(v) for v in violate_qos.values()))

    def store_history(self):
        """Saves episode data to BUFFER. Commits to history every bin_size eps."""
        # Calculate episode-level summaries
        ep_total_reward = np.sum(self.episode_upper_rewards) + np.sum(self.episode_lower_rewards)
        ep_energy = np.sum(self.episode_energy)
        
        success = np.sum(self.episode_success_qos)
        violate = np.sum(self.episode_violate_qos)
        ep_qos_rate = success / (violate if violate > 0 else 1.0)
        
        ep_avg_f1 = np.mean(self.episode_f1) if self.episode_f1 else 0
        ep_avg_remaining = np.mean(self.episode_remaining_tasks) if self.episode_remaining_tasks else 0
        
        # Store in buffer
        self.buffer["total_reward"].append(ep_total_reward)
        self.buffer["total_energy"].append(ep_energy)
        self.buffer["qos_rate"].append(ep_qos_rate)
        self.buffer["avg_f1"].append(ep_avg_f1)
        self.buffer["avg_remaining"].append(ep_avg_remaining)
        
        # Detailed delays buffer
        for nid, d_list in self.episode_node_raw_delays.items():
            self.buffer[f"node_delay_{nid}"].append(np.mean(d_list))
        for sid, d_list in self.episode_service_raw_delays.items():
            self.buffer[f"service_delay_{sid}"].append(np.mean(d_list))
            
        self.episode_count += 1
        
        # Commit to history every bin_size episodes
        if self.episode_count % self.bin_size == 0:
            self._commit_buffer()
            
        self.reset_episode()

    def _commit_buffer(self):
        """Commits the average of buffered episodes to long-term history."""
        self.history["total_reward"].append(np.mean(self.buffer["total_reward"]))
        self.history["total_energy"].append(np.mean(self.buffer["total_energy"]))
        self.history["qos_rate"].append(np.mean(self.buffer["qos_rate"]))
        self.history["avg_f1"].append(np.mean(self.buffer["avg_f1"]))
        self.history["avg_remaining"].append(np.mean(self.buffer["avg_remaining"]))
        
        # Commmit detailed delays
        for key in list(self.buffer.keys()):
            if key.startswith("node_delay_"):
                nid = key.replace("node_delay_", "")
                self.history["node_delays"][nid].append(np.mean(self.buffer[key]))
            elif key.startswith("service_delay_"):
                sid = int(key.replace("service_delay_", ""))
                self.history["service_delays"][sid].append(np.mean(self.buffer[key]))
        
        # Clear buffer
        self.buffer.clear()

    def report_episode(self, ep):
        """Prints a summary of the current episode (using the last buffered values)."""
        # We need to access values BEFORE they were cleared
        reward = self.buffer["total_reward"][-1] if self.buffer["total_reward"] else 0
        qos_rate = self.buffer["qos_rate"][-1] if self.buffer["qos_rate"] else 0
        
        print(f"\n--- Episode {ep} Summary ---")
        print(f"Total Reward: {reward:.2f}")
        print(f"QoS Rate (S/V): {qos_rate:.2f}")
        print(f"Batch Progress: {self.episode_count % self.bin_size}/{self.bin_size}")
        print("---------------------------\n")

    def _moving_average(self, data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    def plot_history(self, save_dir="src/visualize/plots", ep=None):
        """Generates performance charts using binned historical data."""
        if not self.history["total_reward"]:
            return # Nothing to plot yet
            
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        bins = range(1, len(self.history["total_reward"]) + 1)
        labels = [f"Bin {b}\n({b*self.bin_size} eps)" for b in bins]
        
        plt.figure(figsize=(20, 15))
        plt.suptitle(f"Model Evolution - Binned every {self.bin_size} Episodes", fontsize=18)
        
        # Plot 1: Reward
        plt.subplot(3, 2, 1)
        plt.plot(bins, self.history["total_reward"], marker='o', color='blue')
        plt.title("Reward Evolution (Binned Average)")
        plt.xticks(bins, labels)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: QoS Rate
        plt.subplot(3, 2, 2)
        plt.plot(bins, self.history["qos_rate"], marker='o', color='purple')
        plt.title("QoS Success Rate (Success/Violate)")
        plt.xticks(bins, labels)
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Energy
        plt.subplot(3, 2, 3)
        plt.plot(bins, self.history["total_energy"], marker='o', color='orange')
        plt.title("Total Energy Consumption")
        plt.xticks(bins, labels)
        plt.grid(True, alpha=0.3)

        # Plot 4: Remaining Tasks
        plt.subplot(3, 2, 4)
        plt.plot(bins, self.history["avg_remaining"], marker='o', color='brown')
        plt.title("Task Clearing Efficiency (Avg Remaining)")
        plt.xticks(bins, labels)
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Delay per Service
        plt.subplot(3, 2, 5)
        for sid, delay_history in self.history["service_delays"].items():
            plt.plot(bins, delay_history, marker='.', label=f"Svc {sid}")
        plt.title("Average Delay per Service")
        plt.xlabel("Training Bins")
        plt.xticks(bins, labels)
        plt.legend(ncol=2, fontsize='small')
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Delay per Node
        plt.subplot(3, 2, 6)
        for nid, delay_history in self.history["node_delays"].items():
            plt.plot(bins, delay_history, marker='.', label=f"Node {nid}")
        plt.title("Average Delay per Node")
        plt.xlabel("Training Bins")
        plt.xticks(bins, labels)
        plt.legend(ncol=2, fontsize='small')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save latest
        plt.savefig(os.path.join(save_dir, "training_metrics.png"))
        
        # Archive
        if ep is not None:
            archive_dir = os.path.join(save_dir, "archive")
            if not os.path.exists(archive_dir):
                os.makedirs(archive_dir)
            plt.savefig(os.path.join(archive_dir, f"metrics_ep_{ep}.png"))
            
        plt.close()
