import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

class MetricsAggregator:
    def __init__(self):
        self.history = defaultdict(list)
        self.episode_count = 0
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
        
        # Per-node per-service delays (realized)
        self.episode_node_service_delays = defaultdict(lambda: defaultdict(list))

    def add_upper(self, step_output):
        """Adds data from an upper-level step."""
        self.episode_upper_rewards.append(step_output.get("reward", 0))
        
        remaining = step_output.get("remaining_task", {})
        # Aggregate remaining tasks across all nodes (summing the vectors)
        if remaining:
            total_remaining = sum(np.sum(v) for v in remaining.values())
            self.episode_remaining_tasks.append(total_remaining)

    def add_lower(self, step_output):
        """Adds data from a lower-level step."""
        self.episode_lower_rewards.append(step_output.get("reward", 0))
        
        info = step_output.get("info", {})
        
        # F1 and Energy distributions (per node)
        f1_dist = info.get("f1", {})
        if f1_dist:
            self.episode_f1.append(sum(f1_dist.values()))
            
        energy_dist = info.get("energy", {})
        if energy_dist:
            self.episode_energy.append(sum(energy_dist.values()))
            
        # Delay and QoS info (per node vectors)
        v_delay = info.get("virtual_delay", {})
        if v_delay:
            self.episode_virtual_delay.append(np.mean([np.mean(v) for v in v_delay.values()]))
            
        r_delay = info.get("realized_delay", {})
        if r_delay:
            self.episode_realized_delay.append(np.mean([np.mean(v) for v in r_delay.values()]))
            # Track per-node per-service avg delay
            for nid, delays in r_delay.items():
                for sid, d in enumerate(delays):
                    if d > 1e-9: # Only track actual delays
                        self.episode_node_service_delays[nid][sid].append(d)
            
        success_qos = info.get("success_qos", {})
        if success_qos:
            self.episode_success_qos.append(sum(np.sum(v) for v in success_qos.values()))
            
        violate_qos = info.get("violate_qos", {})
        if violate_qos:
            self.episode_violate_qos.append(sum(np.sum(v) for v in violate_qos.values()))

    def store_history(self):
        """Saves episode averages to history and RESETS intra-episode data."""
        self.history["upper_reward"].append(np.sum(self.episode_upper_rewards))
        self.history["lower_reward"].append(np.sum(self.episode_lower_rewards))
        self.history["total_reward"].append(np.sum(self.episode_upper_rewards) + np.sum(self.episode_lower_rewards))
        
        self.history["avg_f1"].append(np.mean(self.episode_f1) if self.episode_f1 else 0)
        self.history["total_energy"].append(np.sum(self.episode_energy) if self.episode_energy else 0)
        self.history["avg_virtual_delay"].append(np.mean(self.episode_virtual_delay) if self.episode_virtual_delay else 0)
        self.history["avg_realized_delay"].append(np.mean(self.episode_realized_delay) if self.episode_realized_delay else 0)
        self.history["total_success_qos"].append(np.sum(self.episode_success_qos) if self.episode_success_qos else 0)
        self.history["total_violate_qos"].append(np.sum(self.episode_violate_qos) if self.episode_violate_qos else 0)
        
        # Calculate QoS Rate: Success / (Success + Violate)
        success = np.sum(self.episode_success_qos) if self.episode_success_qos else 0
        violate = np.sum(self.episode_violate_qos) if self.episode_violate_qos else 0
        qos_success_rate = success / (success + violate) if (success + violate) > 0 else 0
        self.history["qos_success_rate"].append(qos_success_rate)
        
        # Keep old qos_rate for backward compatibility if needed, but we focus on success rate
        qos_rate = success / (violate if violate > 0 else 1.0)
        self.history["qos_rate"].append(qos_rate)
        
        self.history["avg_remaining_tasks"].append(np.mean(self.episode_remaining_tasks) if self.episode_remaining_tasks else 0)
        
        self.episode_count += 1
        
        # Auto-plot every 100 episodes
        if self.episode_count % 100 == 0:
            self.plot_history(ep=self.episode_count)
            
        # Auto-reset for next episode
        # Note: We reset AFTER report_episode is called usually, 
        # but store_history is called FIRST in train.py. 
        # So we should probably NOT reset here if report_episode needs the data.
        # However, report_episode in this class uses self.history which is NOT reset.
        # But per-node delays ARE in episode_node_service_delays.
        # I'll move reset_episode() to the end of report_episode or make report_episode use history.
        # Actually, let's store the node delays in history too if we want them persistent.
        # For now, I'll let train.py call report_episode then store_history maybe? 
        # No, train.py calls store_history then report_episode.
        # I'll keep the episode data until reset_episode is explicitly called or at start of next store cycle.
        # Let's just NOT reset here, and instead reset at the start of add_upper if it's a new episode.
        # Or even better, reset at the end of report_episode.

    def report_episode(self, ep):
        """Prints a summary of the current episode."""
        upper_reward = np.mean(self.episode_upper_rewards) if self.episode_upper_rewards else 0
        lower_reward = np.mean(self.episode_lower_rewards) if self.episode_lower_rewards else 0
        total_reward = self.history["total_reward"][-1]
        energy = self.history["total_energy"][-1]
        qos_success_rate = self.history["qos_success_rate"][-1]
        
        print(f"\n--- Episode {ep} Summary ---")
        print(f"Avg Upper Reward: {upper_reward:.4f}")
        print(f"Avg Lower Reward: {lower_reward:.4f}")
        print(f"Total Reward:     {total_reward:.2f}")
        print(f"Total Energy:     {energy:.4f} J")
        print(f"QoS Success Rate: {qos_success_rate:.2%}")
        print(f"Avg Remaining Tasks: {self.history['avg_remaining_tasks'][-1]:.2f}")
        
        print("\n--- Average Delay per Node and Service ---")
        if not self.episode_node_service_delays:
            print("No delay data recorded for this episode.")
        else:
            # Find the number of services from the first node's data
            num_services = len(next(iter(self.episode_node_service_delays.values())))
            header = "Node ID | " + " | ".join([f"Svc {i}" for i in range(num_services)])
            print(header)
            print("-" * len(header))
            
            # Sort nodes for consistent output
            for nid in sorted(self.episode_node_service_delays.keys()):
                delays = self.episode_node_service_delays[nid]
                row = f"{nid:<7} | "
                svc_delays = []
                for sid in sorted(delays.keys()):
                    d_list = delays[sid]
                    avg_d = np.mean(d_list) if d_list else 0.0
                    svc_delays.append(f"{avg_d:7.4f}")
                print(row + " | ".join(svc_delays))
        print("---------------------------\n")
        
        # Reset episode data after reporting
        self.reset_episode()

    def _moving_average(self, data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    def plot_history(self, save_dir="src/visualize/plots", ep=None):
        """Generates and saves performance charts showing evolution."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        episodes = range(1, len(self.history["total_reward"]) + 1)
        window = 10 # Window for smoothing
        
        plt.figure(figsize=(18, 12))
        plt.suptitle(f"Model Evolution Over Episodes (up to {len(episodes)})", fontsize=16)
        
        # Plot 1: Rewards (with moving average)
        plt.subplot(2, 3, 1)
        plt.plot(episodes, self.history["total_reward"], alpha=0.3, color='blue', label="Raw Total")
        if len(episodes) >= window:
            ma = self._moving_average(self.history["total_reward"], window)
            plt.plot(range(window, len(self.history["total_reward"]) + 1), ma, color='blue', label=f"MA-{window}")
        plt.title("Reward Evolution")
        plt.xlabel("Episode")
        plt.legend()
        
        # Plot 2: F1 Score
        plt.subplot(2, 3, 2)
        plt.plot(episodes, self.history["avg_f1"], alpha=0.3, color='green')
        if len(episodes) >= window:
            ma = self._moving_average(self.history["avg_f1"], window)
            plt.plot(range(window, len(self.history["avg_f1"]) + 1), ma, color='green', label=f"MA-{window}")
        plt.title("F1 Score Improvement")
        plt.xlabel("Episode")
        
        # Plot 3: Energy
        plt.subplot(2, 3, 3)
        plt.plot(episodes, self.history["total_energy"], alpha=0.3, color='orange')
        if len(episodes) >= window:
            ma = self._moving_average(self.history["total_energy"], window)
            plt.plot(range(window, len(self.history["total_energy"]) + 1), ma, color='orange', label=f"MA-{window}")
        plt.title("Energy Consumption Trend")
        plt.xlabel("Episode")
        
        # Plot 4: Delays
        plt.subplot(2, 3, 4)
        plt.plot(episodes, self.history["avg_realized_delay"], alpha=0.3, color='red', label="Realized")
        if len(episodes) >= window:
            ma = self._moving_average(self.history["avg_realized_delay"], window)
            plt.plot(range(window, len(self.history["avg_realized_delay"]) + 1), ma, color='red', label=f"MA-{window}")
        plt.title("Delay Evolution")
        plt.xlabel("Episode")
        plt.legend()
        
        # Plot 5: QoS Rate (Success / Violate)
        plt.subplot(2, 3, 5)
        plt.plot(episodes, self.history["qos_rate"], alpha=0.3, color='purple')
        if len(episodes) >= window:
            ma = self._moving_average(self.history["qos_rate"], window)
            plt.plot(range(window, len(self.history["qos_rate"]) + 1), ma, color='purple', linewidth=2)
        plt.title("QoS Rate (Success/Violate)")
        plt.xlabel("Episode")

        # Plot 6: Remaining Tasks
        plt.subplot(2, 3, 6)
        plt.plot(episodes, self.history["avg_remaining_tasks"], alpha=0.3, color='brown')
        if len(episodes) >= window:
            ma = self._moving_average(self.history["avg_remaining_tasks"], window)
            plt.plot(range(window, len(self.history["avg_remaining_tasks"]) + 1), ma, color='brown')
        plt.title("Task Clearing Efficiency")
        plt.xlabel("Episode")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save latest
        plt.savefig(os.path.join(save_dir, "training_metrics.png"))
        
        # Save archival copy if episode number is provided
        if ep is not None:
            archive_dir = os.path.join(save_dir, "archive")
            if not os.path.exists(archive_dir):
                os.makedirs(archive_dir)
            plt.savefig(os.path.join(archive_dir, f"metrics_ep_{ep}.png"))
            
        plt.close()
