import re

target = r"d:\code\ai_infras\matrix_source\trainers\d3qn_scaffold_strategy_v2.py"

with open(target, "r", encoding="utf-8") as f:
    src = f.read()

# 1. Patch run_training signature
old_sig = "    def run_training(self, trainer: Trainer):"
new_sig = "    def run_training(self, trainer: Trainer, resume_from: str = None, save_every: int = 1, save_dir: str = \"checkpoints\"):"
src = src.replace(old_sig, new_sig, 1)

# 2. Add resume logic
old_ep = '        num_eps = trainer.config.hyper_neural[\'NUMOF_TRAIN_EP\']\n        max_slots = trainer.env.time_manager.max_steps\n\n        for ep in tqdm(range(num_eps), desc="Training"):'
new_ep = (
    "        num_eps = trainer.config.hyper_neural['NUMOF_TRAIN_EP']\n"
    "        max_slots = trainer.env.time_manager.max_steps\n"
    "\n"
    "        start_ep = 0\n"
    "        if resume_from is not None:\n"
    "            start_ep = self.load_checkpoint(trainer, save_dir=resume_from)\n"
    "\n"
    "        for ep in tqdm(range(start_ep, num_eps), desc=\"Training\"):"
)
src = src.replace(old_ep, new_ep, 1)

# 3. Auto-save location
old_print = "            print(f\"Current Epsilon upper: {trainer.eps_upper:.4f} lower: {trainer.eps_lower:.4f}\")"
new_print = (
    "            print(f\"Current Epsilon upper: {trainer.eps_upper:.4f} lower: {trainer.eps_lower:.4f}\")\n"
    "\n"
    "            if (ep + 1) % save_every == 0:\n"
    "                self.save_checkpoint(trainer, ep + 1, save_dir=save_dir)"
)
src = src.replace(old_print, new_print, 1)

# 4. Remove existing methods to avoid duplicate appends (if script ran multiple times)
eval_idx = src.find("\n    # ── Checkpoint")
if eval_idx != -1:
    src = src[:eval_idx]

# 5. Append new methods
methods = r"""
    # ── Checkpoint and Evaluation ─────────────────────────────────────────────

    def save_checkpoint(self, trainer, ep: int, save_dir: str = "checkpoints"):
        import os as _os
        import torch as _torch
        _os.makedirs(save_dir, exist_ok=True)
        trainer.shared_upper_agent.save(_os.path.join(save_dir, "upper_agent.pt"))
        trainer.shared_lower_agent.save(_os.path.join(save_dir, "lower_agent.pt"))
        trainer_state = {
            "ep": ep,
            "total_upper_steps": trainer.total_upper_steps,
            "total_lower_steps": trainer.total_lower_steps,
            "eps_upper": trainer.eps_upper,
            "eps_lower": trainer.eps_lower,
            "zeta_upper": getattr(trainer, 'zeta_upper', 0.0),
            "zeta_lower": getattr(trainer, 'zeta_lower', 0.0)
        }
        _torch.save(trainer_state, _os.path.join(save_dir, "trainer_state.pt"))
        print(f"[Checkpoint] Saved at episode {ep} -> {save_dir}")

    def load_checkpoint(self, trainer, save_dir: str = "checkpoints") -> int:
        import os as _os
        import torch as _torch
        upper_path = _os.path.join(save_dir, "upper_agent.pt")
        lower_path = _os.path.join(save_dir, "lower_agent.pt")
        trainer_path = _os.path.join(save_dir, "trainer_state.pt")

        if not _os.path.exists(upper_path) or not _os.path.exists(lower_path):
            print(f"[Checkpoint] No checkpoint found in '{save_dir}', starting fresh.")
            return 0

        trainer.shared_upper_agent.load(upper_path)
        trainer.shared_lower_agent.load(lower_path)
        
        resumed_ep = 0
        if _os.path.exists(trainer_path):
            state = _torch.load(trainer_path, map_location="cpu")
            trainer.total_upper_steps = state.get("total_upper_steps", trainer.total_upper_steps)
            trainer.total_lower_steps = state.get("total_lower_steps", trainer.total_lower_steps)
            trainer.eps_upper = state.get("eps_upper", trainer.eps_upper)
            trainer.eps_lower = state.get("eps_lower", trainer.eps_lower)
            if hasattr(trainer, 'zeta_upper') and "zeta_upper" in state:
                trainer.zeta_upper = state["zeta_upper"]
            if hasattr(trainer, 'zeta_lower') and "zeta_lower" in state:
                trainer.zeta_lower = state["zeta_lower"]
            resumed_ep = state.get("ep", 0)

        print(f"[Checkpoint] Resumed from episode {resumed_ep} <- {save_dir}")
        return resumed_ep

    METRICS_BAR = ['total_energy', 'avg_backlog_drift', 'realized_delay', 'qos_success_rate']

    def evaluate(self, trainer, checkpoint_dir: str, num_eval_eps: int = 10, label: str = "D3QN-SCAFFOLD", plot_dir: str = "eval_results", compare_results: dict = None) -> dict:
        import numpy as np
        import os as _os
        import torch

        _os.makedirs(plot_dir, exist_ok=True)
        self.load_checkpoint(trainer, save_dir=checkpoint_dir)
        print(f"\n--- Starting Evaluation ({num_eval_eps} Episodes) ---")

        trainer.eps_upper = 0.0
        trainer.eps_lower = 0.0
        max_slots = trainer.env.time_manager.max_steps
        collected = {m: [] for m in self.METRICS_BAR}

        for ep_i in range(num_eval_eps):
            obs = trainer.env.reset()
            obs_upper = obs['upper']
            prev_lower_res = obs['lower']
            current_upper_state = self.build_upper_state(trainer, obs_upper) 

            for slot in range(max_slots):
                if trainer.env.time_manager.is_new_frame():
                    # Set deterministic True essentially
                    # D3QNAgent code changed to accept deterministic directly!
                    mf_global = obs_upper.get('mean_fields', torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device))
                    edge_states = current_upper_state[trainer.edge_node_ids]
                    edge_mfs = mf_global[trainer.edge_node_ids]
                    instance_indices = torch.tensor([trainer.node_to_instance[nid] for nid in trainer.edge_node_ids], device=trainer.device)
                    batch_a_ids = trainer.shared_upper_agent.choose_action_batch(
                        edge_states, edge_mfs, epsilon=0.0, zeta=getattr(trainer, 'zeta_upper', 0.1), agent_indices=instance_indices, deterministic=True
                    )
                    
                    u_acts_matrix = torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device)
                    from matrix_source.utils.math_utils import to_binary
                    for i, nid in enumerate(trainer.edge_node_ids):
                        u_acts_matrix[nid] = torch.tensor(to_binary(batch_a_ids[i], trainer.num_services), device=trainer.device)
                    for nid in trainer.env.static_matrices.get("cloud_ids", []):
                        u_acts_matrix[nid] = torch.ones(trainer.num_services, device=trainer.device)
                        
                    trainer.env.step_upper(u_acts_matrix)

                t_idx, s_idx, batch_sizes, tasks_min_accuracy, task_deadlines = trainer.workload_gen.generate_step()
                if len(t_idx) > 0:
                    obs_dict = prev_lower_res['obs']
                    mf_terminals = prev_lower_res['mean_field']
                    meta = trainer.env.metadata
                    unit_sizes = meta['service_input_size']
                    placement_matrix = trainer.env.engine.placement_matrix

                    data_sizes = batch_sizes * unit_sizes[s_idx].squeeze(-1)
                    s_tasks = torch.stack([data_sizes, tasks_min_accuracy, task_deadlines, meta['service_omega'][s_idx].squeeze(-1)], dim=1).float()
                    service_placements = placement_matrix[:, s_idx].T
                    s_backlogs = obs_dict['backlog'][:, s_idx].T * service_placements
                    s_cpus = obs_dict['cpu_alloc'][:, s_idx].T * service_placements
                    
                    states = torch.cat([s_tasks, s_backlogs, s_cpus], dim=1)
                    states[:, 0] /= trainer.config.norm_data_size
                    states[:, 1] /= 100.0
                    if states.shape[1] > 4:
                        states[:, 4:4+2*trainer.num_nodes] /= trainer.config.norm_gflop
                        
                    masks = D3QNScaffoldStrategy.calculate_lower_masks(trainer, t_idx, s_idx, tasks_min_accuracy, placement_matrix)
                    batch_actions = trainer.shared_lower_agent.choose_action_batch(
                        states, mf_terminals[t_idx], epsilon=0.0, zeta=getattr(trainer, 'zeta_lower', 0.1), masks_batch=masks.to(trainer.device),
                        agent_indices=torch.arange(trainer.num_terminals, device=trainer.device), deterministic=True
                    )
                    a_ids = torch.tensor(batch_actions, device=trainer.device)
                    n_idx, m_idx = a_ids // trainer.max_models, a_ids % trainer.max_models

                    results = trainer.env.step_lower(t_idx, s_idx, batch_sizes, n_idx, m_idx, task_deadlines, tasks_min_accuracy)
                    
                    trainer.aggregator.add_step_matrices(
                        f_alloc=trainer.env.engine.cpu_alloc_matrix,
                        arrivals=results['info']['arrival_matrix'],
                        backlog=trainer.env.engine.backlog_queue.sum(dim=-1)
                    )
                    trainer.aggregator.add_lower(results)
                    prev_lower_res = results
                else:
                    trainer.env.time_manager.tick()

                if trainer.env.time_manager.is_new_frame():
                    res_upper_final = trainer.env.collect_upper_metrics()
                    trainer.aggregator.add_upper(res_upper_final)
                    current_upper_state = self.build_upper_state(trainer, res_upper_final)
                    obs_upper = res_upper_final

            trainer.aggregator.store_history()
            for m in self.METRICS_BAR:
                hist = trainer.aggregator.history.get(m, [])
                if hist: collected[m].append(hist[-1])
            trainer.aggregator.reset_episode()
            
            line = "  ".join(f"{m.split('_')[-1]}={collected[m][-1]:.4f}" for m in self.METRICS_BAR if collected[m])
            print(f"[Eval] Episode {ep_i + 1}/{num_eval_eps} - {line}")

        results_summary = {}
        print("\n[Eval] === Results Summary ===")
        for m in self.METRICS_BAR:
            arr = np.array(collected[m], dtype=np.float64)
            results_summary[m] = {
                "mean": float(arr.mean()) if len(arr) > 0 else 0.0,
                "std":  float(arr.std())  if len(arr) > 0 else 0.0,
                "raw":  arr.tolist(),
            }
            print(f"  {m:25s}: mean={results_summary[m]['mean']:.4f}  std+-{results_summary[m]['std']:.4f}")

        self._plot_eval_bar(results_summary, label, compare_results, plot_dir)
        return results_summary

    def _plot_eval_bar(self, results: dict, label: str, compare_results: dict, plot_dir: str):
        import numpy as np
        import matplotlib.pyplot as plt
        import os as _os

        METRIC_DISPLAY = {
            'total_energy':      'Total Energy',
            'avg_backlog_drift': 'Avg Backlog Drift',
            'realized_delay':    'Realized Delay',
            'qos_success_rate':  'QoS Success Rate',
        }
        all_algos = {label: {m: (results[m]['mean'], results[m]['std']) for m in self.METRICS_BAR}}
        if compare_results: all_algos.update(compare_results)
        algos  = list(all_algos.keys())
        n_algo = len(algos)
        n_eps  = len(next(iter(results.values()))['raw'])
        colors = plt.cm.tab10.colors

        fig, axes = plt.subplots(1, len(self.METRICS_BAR), figsize=(5.5 * len(self.METRICS_BAR), 5))
        if len(self.METRICS_BAR) == 1: axes = [axes]

        for ax_i, metric in enumerate(self.METRICS_BAR):
            ax = axes[ax_i]
            x     = np.arange(n_algo)
            means = [all_algos[a].get(metric, (0.0, 0.0))[0] for a in algos]
            stds  = [all_algos[a].get(metric, (0.0, 0.0))[1] for a in algos]

            bars = ax.bar(
                x, means, yerr=stds,
                color=[colors[i % len(colors)] for i in range(n_algo)],
                capsize=7, width=0.55, alpha=0.88,
                error_kw={"elinewidth": 2.0, "ecolor": "black"},
                zorder=3,
            )

            for bar, mv, sv in zip(bars, means, stds):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + sv + abs(bar.get_height()) * 0.02 + 1e-9,
                    f"{mv:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                )

            ax.set_title(METRIC_DISPLAY.get(metric, metric), fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(algos, rotation=15, ha="right", fontsize=10)
            ax.set_ylabel("Value", fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle(f"Evaluation Metrics  (n={n_eps} episodes)", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        out_path = _os.path.join(plot_dir, "eval_metrics_bar.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Eval] Bar chart saved -> {out_path}")
"""

final = src.rstrip() + "\n" + methods + "\n"
with open(target, "w", encoding="utf-8", newline="\n") as f:
    f.write(final)

print("Patch complete D3QN Scaffold!")
