import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

class MetricsAggregator:
    def __init__(self):
        self.history = defaultdict(list)
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
            
        success_qos = info.get("success_qos", {})
        if success_qos:
            self.episode_success_qos.append(sum(np.sum(v) for v in success_qos.values()))
            
        violate_qos = info.get("violate_qos", {})
        if violate_qos:
            self.episode_violate_qos.append(sum(np.sum(v) for v in violate_qos.values()))

    def store_history(self):
        """Saves episode averages to history."""
        self.history["upper_reward"].append(np.sum(self.episode_upper_rewards))
        self.history["lower_reward"].append(np.sum(self.episode_lower_rewards))
        self.history["total_reward"].append(np.sum(self.episode_upper_rewards) + np.sum(self.episode_lower_rewards))
        
        self.history["avg_f1"].append(np.mean(self.episode_f1) if self.episode_f1 else 0)
        self.history["total_energy"].append(np.sum(self.episode_energy) if self.episode_energy else 0)
        self.history["avg_virtual_delay"].append(np.mean(self.episode_virtual_delay) if self.episode_virtual_delay else 0)
        self.history["avg_realized_delay"].append(np.mean(self.episode_realized_delay) if self.episode_realized_delay else 0)
        self.history["total_success_qos"].append(np.sum(self.episode_success_qos) if self.episode_success_qos else 0)
        self.history["total_violate_qos"].append(np.sum(self.episode_violate_qos) if self.episode_violate_qos else 0)
        self.history["avg_remaining_tasks"].append(np.mean(self.episode_remaining_tasks) if self.episode_remaining_tasks else 0)

    def report_episode(self, ep):
        """Prints a summary of the current episode."""
        reward = self.history["total_reward"][-1]
        f1 = self.history["avg_f1"][-1]
        energy = self.history["total_energy"][-1]
        qos_v = self.history["total_violate_qos"][-1]
        
        print(f"\n--- Episode {ep} Summary ---")
        print(f"Total Reward: {reward:.2f}")
        print(f"Avg F1 Score: {f1:.4f}")
        print(f"Total Energy: {energy:.2f}")
        print(f"QoS Violations: {qos_v}")
        print(f"Avg Remaining Tasks: {self.history['avg_remaining_tasks'][-1]:.2f}")
        print("---------------------------\n")

    def plot_history(self, save_dir="src/visualize/plots"):
        """Generates and saves performance charts."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        episodes = range(1, len(self.history["total_reward"]) + 1)
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Rewards
        plt.subplot(2, 3, 1)
        plt.plot(episodes, self.history["total_reward"], label="Total")
        plt.plot(episodes, self.history["upper_reward"], label="Upper")
        plt.plot(episodes, self.history["lower_reward"], label="Lower")
        plt.title("Rewards")
        plt.xlabel("Episode")
        plt.legend()
        
        # Plot 2: F1 Score
        plt.subplot(2, 3, 2)
        plt.plot(episodes, self.history["avg_f1"], color='green')
        plt.title("Average F1 Score")
        plt.xlabel("Episode")
        
        # Plot 3: Energy
        plt.subplot(2, 3, 3)
        plt.plot(episodes, self.history["total_energy"], color='orange')
        plt.title("Total Energy Consumption")
        plt.xlabel("Episode")
        
        # Plot 4: Delays
        plt.subplot(2, 3, 4)
        plt.plot(episodes, self.history["avg_virtual_delay"], label="Virtual")
        plt.plot(episodes, self.history["avg_realized_delay"], label="Realized")
        plt.title("Average Delays")
        plt.xlabel("Episode")
        plt.legend()
        
        # Plot 5: QoS
        plt.subplot(2, 3, 5)
        plt.plot(episodes, self.history["total_success_qos"], label="Success", color='blue')
        plt.plot(episodes, self.history["total_violate_qos"], label="Violate", color='red')
        plt.title("QoS metrics")
        plt.xlabel("Episode")
        plt.legend()

        # Plot 6: Remaining Tasks
        plt.subplot(2, 3, 6)
        plt.plot(episodes, self.history["avg_remaining_tasks"], color='purple')
        plt.title("Avg Remaining Tasks")
        plt.xlabel("Episode")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_metrics.png"))
        plt.close()
