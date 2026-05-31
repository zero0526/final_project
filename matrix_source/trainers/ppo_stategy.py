"""ppo_strategy.py — Simultaneous dual-agent PPO, collect-then-train.

Training protocol:
    REPEAT 200 times:
        1. Run episodes until BOTH buffers full
           - Lower needs ≥ 6400 transitions (~2 episodes)
           - Upper needs ≥ 640 transitions (~8 episodes)
        2. Train both agents
        3. Clear both buffers
        4. Advance phase config

    Each "train step" = N episodes of collection + 1 train per agent
"""
import time
import torch
from tqdm import tqdm
import os
import numpy as np
from datetime import datetime

from matrix_source.agents.ppo import PPOAgent
from matrix_source.trainers.strategies import AlgorithmStrategy
from matrix_source.utils.math_utils import to_binary
from matrix_source.utils.config_updater import (
    apply_config_to_agent, create_entropy_scheduler,
    create_lr_schedulers, start_convergence,
)
from matrix_source.visualize.tracking import TrainingDashboard
from matrix_source.configs.ppo_config import PPOTrainingConfig

# ═══════════════════════════════════════════════════════
#  PPO STRATEGY
# ═══════════════════════════════════════════════════════

class PPOStrategy(AlgorithmStrategy):

    def __init__(self):
        super().__init__()
        self.config = PPOTrainingConfig()

        self.lower_min = self.config.LOWER_BUFFER['min_size']
        self.upper_min = self.config.UPPER_BUFFER['min_size']
        self.max_train_steps = self.config.MAX_TRAIN_STEPS

        self.lower_train_count = 0
        self.upper_train_count = 0
        self.is_evaluating = False
        self._last_phase = None

    # ══════════════════════════════════════════════════
    #  INITIALIZE
    # ══════════════════════════════════════════════════

    def initialize_agents(self, trainer):
        lower_mf_dim = trainer.num_nodes + trainer.max_models

        self._upper_dashboard = TrainingDashboard.create(
            'ppo', 'upper_ppo', plot_every=50)
        self._lower_dashboard = TrainingDashboard.create(
            'ppo', 'lower_ppo', plot_every=50)

        trainer.shared_upper_agent = PPOAgent(
            node_id=-2, node_type="Edge_Group",
            state_dim=trainer.upper_state_dim,
            action_dim=trainer.upper_action_dim,
            u_action_dim=trainer.upper_u_action_dim,
            mf_hidden_sizes=tuple(trainer.config.hyper_neural["MF_HIDDEN_LAYER"]),
            mf_lr=float(trainer.config.hyper_neural['MF_LR']),
            buffer_min_size=self.upper_min,
            hidden_sizes=trainer.config.hyper_neural['AGENT_HIDDEN_LAYER'],
            lr=float(trainer.config.hyper_neural['UPPER_LR']),
            gamma=trainer.config.hyper_neural['DISCOUNT_FACTOR'],
            lam=trainer.config.hyper_neural.get('LAMBDA', 0.95),
            clip_eps=self.config.UPPER_PHASES['EXPLORE'].clip_eps,
            k_epochs=self.config.UPPER_PHASES['EXPLORE'].k_epochs,
            batch_size=self.config.UPPER_BUFFER['batch_size'],
            entropy_coef=self.config.UPPER_ENTROPY.explore_start,
            num_instances=trainer.num_edge_agents,
            device=trainer.device,
            dashboard=self._upper_dashboard,
        )

        trainer.shared_lower_agent = PPOAgent(
            node_id=-1, node_type="Terminal_Group",
            state_dim=trainer.lower_state_dim,
            action_dim=lower_mf_dim,
            u_action_dim=trainer.lower_u_action_dim,
            mf_hidden_sizes=tuple(trainer.config.hyper_neural["MF_HIDDEN_LAYER"]),
            mf_lr=float(trainer.config.hyper_neural['MF_LR']),
            buffer_min_size=self.lower_min,
            hidden_sizes=tuple(trainer.config.hyper_neural['AGENT_HIDDEN_LAYER']),
            lr=float(trainer.config.hyper_neural['LOWER_LR']),
            gamma=trainer.config.hyper_neural['DISCOUNT_FACTOR'],
            lam=trainer.config.hyper_neural.get('LAMBDA', 0.95),
            clip_eps=self.config.LOWER_PHASES['EXPLORE'].clip_eps,
            k_epochs=self.config.LOWER_PHASES['EXPLORE'].k_epochs,
            batch_size=self.config.LOWER_BUFFER['batch_size'],
            entropy_coef=self.config.LOWER_ENTROPY.explore_start,
            num_instances=trainer.num_terminals,
            device=trainer.device,
            dashboard=self._lower_dashboard,
        )

        # Setup entropy + LR schedulers (no target networks)
        for agent, ent_cfg in [
            (trainer.shared_upper_agent, self.config.UPPER_ENTROPY),
            (trainer.shared_lower_agent, self.config.LOWER_ENTROPY),
        ]:
            agent.entropy_sched = create_entropy_scheduler(
                ent_cfg, total_train_steps=self.max_train_steps)
            agent.lr_schedulers = create_lr_schedulers(
                agent, lr_min_actor=1e-6, lr_min_critic=5e-6,
                total_steps=self.max_train_steps // 2)

        self._upper = trainer.shared_upper_agent
        self._lower = trainer.shared_lower_agent

        print(f"\n{'='*60}")
        print(f"  PPO Collect-Then-Train")
        print(f"  Max train steps : {self.max_train_steps}")
        print(f"  Lower buffer    : collect {self.lower_min} → train")
        print(f"  Upper buffer    : collect {self.upper_min} → train")
        print(f"  Phases          : EXPLORE(0-67) → REFINE(68-139) → CONVERGE(140-199)")
        print(f"{'='*60}\n")

    # ══════════════════════════════════════════════════
    #  PHASE CONFIG
    # ══════════════════════════════════════════════════

    def _apply_phase_config(self, trainer, train_step):
        phase = self.config.get_phase(train_step)
        phase_bounds = self.config.PHASE_BOUNDS[phase]
        is_converge = train_step >= self.config.CONVERGE_START

        if train_step == self.config.CONVERGE_START:
            for agent in [trainer.shared_upper_agent, trainer.shared_lower_agent]:
                if agent.lr_schedulers:
                    start_convergence(agent.lr_schedulers)
            print(f"\n[LR] Convergence annealing started at step {train_step}")

        _, zeta_u = apply_config_to_agent(
            trainer.shared_upper_agent,
            self.config.UPPER_PHASES[phase],
            train_step, phase_bounds, is_converge)
        _, zeta_l = apply_config_to_agent(
            trainer.shared_lower_agent,
            self.config.LOWER_PHASES[phase],
            train_step, phase_bounds, is_converge)

        return zeta_l, zeta_u, phase

    # ══════════════════════════════════════════════════
    #  COLLECT ONE EPISODE
    # ══════════════════════════════════════════════════

    def _collect_episode(self, trainer):
        """Run one full episode. Store transitions to both buffers.
        Returns: (ep_reward_lower, ep_reward_upper, n_lower_stores)
        """
        max_slots = trainer.env.time_manager.max_steps

        obs = trainer.env.reset()
        obs_upper = obs['upper']
        prev_lower_res = obs['lower']
        lower_mf_dim = trainer.num_nodes + trainer.max_models
        prev_lower_res["mean_field"] = torch.zeros(
            (trainer.num_terminals, lower_mf_dim), device=trainer.device)
        current_upper_state = self.build_upper_state(trainer, obs_upper)

        ep_rew_l = 0.0
        ep_rew_u = 0.0
        n_lower = 0

        for slot in range(max_slots):
            # ── Upper: act + step (new frame) ──
            u_acts = u_lp = u_v = None
            if trainer.env.time_manager.is_new_frame():
                u_acts, u_lp, u_v = self.get_upper_actions(
                    trainer, current_upper_state, obs_upper)
                trainer.env.step_upper(u_acts)

            # ── Lower: act + store (tasks arrive) ──
            t_idx, s_idx, batch_sizes, t_acc, t_dl = \
                trainer.workload_gen.generate_step()

            if len(t_idx) > 0:
                n_idx, m_idx, masks, l_lp, l_v = self.get_lower_actions(
                    trainer, prev_lower_res, t_idx, s_idx,
                    t_acc, t_dl, batch_sizes)

                results = trainer.env.step_lower(
                    t_idx, s_idx, batch_sizes, n_idx, m_idx, t_dl, t_acc)

                curr_mf = self.compute_lower_mean_fields(
                    trainer, t_idx, s_idx, n_idx, m_idx)
                results['mean_field'] = prev_lower_res['mean_field'].clone()
                results['mean_field'][t_idx] = curr_mf

                self.store_lower_transitions(
                    trainer, prev_lower_res, results,
                    t_idx, s_idx, n_idx, m_idx, masks, l_lp, l_v)
                n_lower += len(t_idx)
                ep_rew_l += results.get('reward', 0.0)

                trainer.aggregator.add_step_matrices(
                    f_alloc=trainer.env.engine.cpu_alloc_matrix,
                    arrivals=results['info']['arrival_matrix'],
                    backlog=trainer.env.engine.backlog_queue.sum(dim=-1))
                prev_lower_res = results
            else:
                trainer.env.time_manager.tick()

            # ── Upper: store (new frame) ──
            if (trainer.env.time_manager.current_step-1)% trainer.env.time_manager.timeframe_size==0:
                res_upper = trainer.env.collect_upper_metrics()
                next_upper_state = self.build_upper_state(trainer, res_upper)
                is_done = (slot == max_slots - 1)

                self.store_upper_transitions(
                    trainer, current_upper_state, next_upper_state,
                    obs_upper, res_upper, u_acts, u_lp, u_v, is_done)

                ep_rew_u += res_upper.get('reward_global', 0.0)
                current_upper_state = next_upper_state
                obs_upper = res_upper

        return ep_rew_l, ep_rew_u, n_lower

    # ══════════════════════════════════════════════════
    #  MAIN TRAINING LOOP — Collect-Then-Train
    # ══════════════════════════════════════════════════

    def run_training(self, trainer):
        """Collect-then-train protocol.

        Each train step:
            1. Run episodes until BOTH buffers >= min_size
            2. Apply phase config
            3. Train both agents
            4. Clear both buffers
        """
        pbar = tqdm(total=self.max_train_steps,
                    desc="PPO Collect-Then-Train")

        total_episodes = 0

        for train_step in range(self.max_train_steps):

            # ── Phase config ──
            zeta_l, zeta_u, phase = self._apply_phase_config(
                trainer, train_step)

            if self._last_phase is not None and phase != self._last_phase:
                print(f"\n{'='*60}")
                print(f"  PHASE: {self._last_phase} → {phase} "
                      f"(step {train_step})")
                print(f"{'='*60}")
            self._last_phase = phase

            # ═══════════════════════════════════════
            #  COLLECTION — run episodes until both buffers ready
            # ═══════════════════════════════════════

            episodes_this_step = 0
            total_lower_stores = 0
            total_rew_l = 0.0
            total_rew_u = 0.0
            start= time.time()
            while True:
                rew_l, rew_u, n_l = self._collect_episode(trainer)
                episodes_this_step += 1
                total_lower_stores += n_l
                total_rew_l += rew_l
                total_rew_u += rew_u

                lower_ready = len(self._lower.memory) >= self.lower_min
                upper_ready = len(self._upper.memory) >= self.upper_min

                if lower_ready and upper_ready:
                    break
            print(f"Collect 64 eps took {time.time() - start} seconds.")
            total_episodes += episodes_this_step
            buf_l = len(self._lower.memory)
            buf_u = len(self._upper.memory)

            # ═══════════════════════════════════════
            #  TRAINING — both agents
            # ═══════════════════════════════════════

            # ── Train Lower ──
            start= time.time()
            lower_loss = trainer.shared_lower_agent.learn(
                zeta=zeta_l,
                agents_ids=torch.arange(
                    trainer.num_terminals, device=trainer.device))
            print(f"trainer lower took {time.time() - start} seconds.")
            if lower_loss is not None:
                trainer.total_lower_steps += 1
                self.lower_train_count += 1
                trainer.aggregator.record_td_losses(
                    lower_losses=lower_loss)

                if self.lower_train_count % 10 == 0:
                    os.makedirs('checkpoints', exist_ok=True)
                    trainer.shared_lower_agent.save(
                        f'checkpoints/ppo_lower_{self.lower_train_count}.pth')

            # ── Train Upper ──
            instance_indices = torch.tensor(
                [trainer.node_to_instance[nid]
                 for nid in trainer.edge_node_ids],
                device=trainer.device)
            start= time.time()
            upper_loss = trainer.shared_upper_agent.learn(
                zeta=zeta_u,
                agents_ids=instance_indices)
            print(f"trainer upper took {time.time() - start} seconds.")

            if upper_loss is not None:
                trainer.total_upper_steps += 1
                self.upper_train_count += 1
                trainer.aggregator.record_td_losses(
                    upper_losses=upper_loss)

                if self.upper_train_count % 10 == 0:
                    os.makedirs('checkpoints', exist_ok=True)
                    trainer.shared_upper_agent.save(
                        f'checkpoints/ppo_upper_{self.upper_train_count}.pth')

            # ═══════════════════════════════════════
            #  LOGGING
            # ═══════════════════════════════════════

            trainer.aggregator.store_history()

            ent_l = self._lower.entropy_coef
            ent_u = self._upper.entropy_coef
            k_l = self._lower.k_epochs
            k_u = self._upper.k_epochs
            l_mark = "✓" if lower_loss is not None else "✗"
            u_mark = "✓" if upper_loss is not None else "✗"

            print(
                f"[Step {train_step:3d}/{self.max_train_steps}] "
                f"Phase={phase:8s} | "
                f"ζ_L={zeta_l:.3f} ζ_U={zeta_u:.3f}\n"
                f"  Episodes={episodes_this_step:2d} "
                f"(total={total_episodes}) | "
                f"Stores: L={total_lower_stores}\n"
                f"  Buffer: L={buf_l} U={buf_u} | "
                f"Train: L={l_mark}({lower_loss}) "
                f"U={u_mark}({upper_loss})\n"
                f"  EntL={ent_l:.5f} EntU={ent_u:.5f} | "
                f"k_L={k_l} k_U={k_u} | "
                f"Trains: L={self.lower_train_count} "
                f"U={self.upper_train_count}"
            )

            # Store history
            if hasattr(trainer, 'aggregator') and hasattr(trainer.aggregator, 'store_history'):
                trainer.aggregator.report_episode(total_episodes)

            pbar.update(1)

        pbar.close()
        self.run_evaluation(trainer, num_episodes=5)

    # ══════════════════════════════════════════════════
    #  UPPER ACTIONS
    # ══════════════════════════════════════════════════

    def get_upper_actions(self, trainer, current_upper_state, obs_upper):
        act_matrix = torch.zeros(
            (trainer.num_nodes, trainer.num_services),
            device=trainer.device)

        mf_global = obs_upper.get(
            'mean_fields',
            torch.zeros((trainer.num_nodes, trainer.num_services),
                        device=trainer.device))

        edge_states = current_upper_state[trainer.edge_node_ids]
        edge_mfs = mf_global[trainer.edge_node_ids]
        instance_indices = torch.tensor(
            [trainer.node_to_instance[nid]
             for nid in trainer.edge_node_ids],
            device=trainer.device)

        batch_a_ids, log_probs, values = \
            trainer.shared_upper_agent.choose_action_batch(
                edge_states, edge_mfs,
                agent_indices=instance_indices,
                deterministic=self.is_evaluating,
                zeta=1.0)

        for i, nid in enumerate(trainer.edge_node_ids):
            act_matrix[nid] = torch.tensor(
                to_binary(batch_a_ids[i], trainer.num_services),
                device=trainer.device)

        for nid in trainer.env.static_matrices.get("cloud_ids", []):
            act_matrix[nid] = torch.ones(
                trainer.num_services, device=trainer.device)

        return act_matrix, log_probs, values

    # ══════════════════════════════════════════════════
    #  LOWER ACTIONS
    # ══════════════════════════════════════════════════

    def _get_batch_placements(self, trainer, s_idx, placement_matrix=None):
        if placement_matrix is None:
            placement_matrix = trainer.env.engine.placement_matrix
        return placement_matrix[:, s_idx].T

    def calculate_lower_masks(self, trainer, t_idx, s_idx,
                               placement_matrix=None):
        num_reqs = len(t_idx)
        placements = self._get_batch_placements(
            trainer, s_idx, placement_matrix)
        masks = placements.unsqueeze(-1).expand(
            -1, -1, trainer.max_models).reshape(num_reqs, -1)
        invalid = (masks.sum(dim=1) == 0)
        if invalid.any():
            masks = masks.clone()
            masks[invalid] = 1.0
        return masks

    def get_lower_actions(self, trainer, res_lower, t_idx, s_idx,
                           t_acc, t_dl, batch_sizes):
        obs_dict = res_lower['obs']
        mf_terminals = res_lower['mean_field']
        meta = trainer.env.metadata

        data_sizes = batch_sizes * \
            meta['service_input_size'][s_idx].squeeze(-1)
        s_tasks = torch.stack([
            data_sizes, t_acc, t_dl,
            meta['service_omega'][s_idx].squeeze(-1)
        ], dim=1).float()

        cp = trainer.env.engine.placement_matrix[:, s_idx]
        s_backlogs = (obs_dict['backlog'][:, s_idx] * cp).T
        s_cpus = (obs_dict['cpu_alloc'][:, s_idx] * cp).T
        states = torch.cat([s_tasks, s_backlogs, s_cpus], dim=1)
        states[:, 0] /= trainer.config.norm_data_size
        states[:, 1] /= 100.0
        if states.shape[1] > 4:
            states[:, 4:4 + 2 * trainer.num_nodes] /= \
                trainer.config.norm_gflop

        masks = self.calculate_lower_masks(trainer, t_idx, s_idx)
        mfs = mf_terminals[t_idx]

        batch_actions, log_probs, values = \
            trainer.shared_lower_agent.choose_action_batch(
                states, mfs, masks_batch=masks,
                agent_indices=torch.arange(
                    trainer.num_terminals, device=trainer.device),
                deterministic=self.is_evaluating,
                zeta=1.0)

        a_ids = batch_actions.view(-1).long()
        return (a_ids // trainer.max_models,
                a_ids % trainer.max_models,
                masks, log_probs, values)

    # ══════════════════════════════════════════════════
    #  STORE TRANSITIONS
    # ══════════════════════════════════════════════════

    def store_lower_transitions(self, trainer, current_res, next_res,
                                 t_idx, s_idx, n_idx, m_idx,
                                 masks, log_probs, values):
        from matrix_source.trainers.train import log_transform

        def build_state(obs, tidx, sidx):
            p = trainer.env.engine.placement_matrix[:, sidx]
            b = (obs['backlog'][:, sidx] * p).T
            c = (obs['cpu_alloc'][:, sidx] * p).T
            st = torch.cat([obs['task_reqs'][tidx], b, c], dim=1)
            st[:, 0] /= trainer.config.norm_data_size
            st[:, 1] /= 100.0
            if st.shape[1] > 4:
                st[:, 4:4 + 2 * trainer.num_nodes] /= \
                    trainer.config.norm_gflop
            return st

        c_obs, n_obs = current_res['obs'], next_res['obs']
        states = build_state(c_obs, t_idx, s_idx)
        next_states = build_state(n_obs, t_idx, s_idx)

        reward = next_res['reward'] - next_res["obs"]["virtual_drift"]
        div = trainer.config.norm_lower_rw
        norm_rew = log_transform(reward / (div if div != 0 else 1.0))
        norm_rew -= 0.5 * next_res["violations"]

        avg_mf_loss = 0.0
        if not self.is_evaluating:
            done = torch.tensor(
                [next_res["new_frame"]] * len(t_idx),
                dtype=torch.float32, device=trainer.device)
            c_mf = current_res['mean_field']
            n_mf = next_res['mean_field']
            rewards = torch.full(
                (len(t_idx),), norm_rew,
                dtype=torch.float32, device=trainer.device)
            a_ids = (n_idx * trainer.max_models + m_idx).long()

            avg_mf_loss = \
                trainer.shared_lower_agent \
                    .store_transition_train_mf_batch(
                        states, c_mf[t_idx], n_mf[t_idx],
                        a_ids, rewards, next_states, done,
                        agent_ids=t_idx, log_prob=log_probs,
                        value=values, masks=masks)

        trainer.aggregator.add_lower(
            next_res, mf_loss=avg_mf_loss,
            state=states[0] if len(states) > 0 else None)

    def store_upper_transitions(self, trainer, s_all, ns_all,
                                 current_res, next_res, acts_matrix,
                                 log_probs, values, is_done):
        from matrix_source.trainers.train import log_transform

        reward = next_res['reward_global']
        div = trainer.config.norm_upper_rw
        norm_rew = log_transform(reward / (div if div != 0 else 1.0))

        avg_mf_loss = 0.0
        edge_states = (s_all[trainer.edge_node_ids]
                       if s_all is not None else None)

        if not self.is_evaluating and acts_matrix is not None:
            edge_next = ns_all[trainer.edge_node_ids]
            dones = torch.full(
                (trainer.num_edge_agents,),
                1.0 if is_done else 0.0,
                dtype=torch.float32, device=trainer.device)

            n_mf = next_res['mean_fields']
            c_mf = current_res['mean_fields']
            edge_c = c_mf[trainer.edge_node_ids]
            edge_acts = acts_matrix[trainer.edge_node_ids]

            pw2 = 2 ** torch.arange(
                trainer.num_services - 1, -1, -1,
                device=trainer.device).float()
            edge_a_ids = (edge_acts * pw2).sum(dim=1).long()

            rewards = torch.full(
                (trainer.num_edge_agents,), norm_rew,
                dtype=torch.float32, device=trainer.device)
            inst_idx = torch.tensor(
                [trainer.node_to_instance[nid]
                 for nid in trainer.edge_node_ids],
                device=trainer.device)

            avg_mf_loss = \
                trainer.shared_upper_agent \
                    .store_transition_train_mf_batch(
                        edge_states, edge_c,
                        n_mf[trainer.edge_node_ids],
                        edge_a_ids, rewards, edge_next,
                        dones, agent_ids=inst_idx,
                        log_prob=log_probs, value=values)

        agg_state = (edge_states[0]
                     if (edge_states is not None
                         and len(edge_states) > 0) else None)
        trainer.aggregator.add_upper(
            next_res, mf_loss=avg_mf_loss, state=agg_state)

        if self.is_evaluating:
            return {
                'reward': next_res['reward_global'],
                'backlog': next_res['obs']['backlog'].sum().item(),
                'energy': next_res['info'].get('energy', 0.0),
            }
        return None

    # ══════════════════════════════════════════════════
    #  STATE BUILDERS
    # ══════════════════════════════════════════════════

    def build_upper_state(self, trainer, obs_upper):
        return torch.cat(
            [obs_upper['actions'], obs_upper['phi_prob']], dim=1)

    def compute_lower_mean_fields(self, trainer, t_idx, s_idx,
                                    n_idx, m_idx):
        B = len(t_idx)
        mf_dim = trainer.num_nodes + trainer.max_models
        device = trainer.device
        two_hot = torch.zeros(B, mf_dim, device=device)
        arange_b = torch.arange(B, device=device)
        two_hot[arange_b, n_idx.long()] = 1.0
        two_hot[arange_b, trainer.num_nodes + m_idx.long()] = 1.0

        num_svc = int(s_idx.max().item()) + 1
        sizes = torch.bincount(s_idx, minlength=num_svc).float()
        sums = torch.zeros(num_svc, mf_dim, device=device)
        sums.index_add_(0, s_idx, two_hot)

        s_sums = sums[s_idx]
        s_sizes = sizes[s_idx]
        denom = (s_sizes - 1).clamp(min=1)
        local = (s_sums - two_hot) / denom.unsqueeze(1)
        local = torch.where(
            (s_sizes > 1).unsqueeze(1), local,
            torch.zeros_like(local))
        return local

    # ══════════════════════════════════════════════════
    #  EVALUATION
    # ══════════════════════════════════════════════════

    def run_evaluation(self, trainer, num_episodes=5):
        print(f"\n{'='*60}")
        print(f"  EVALUATION ({num_episodes} episodes, deterministic)")
        print(f"{'='*60}")
        self.is_evaluating = True
        max_slots = trainer.env.time_manager.max_steps

        metrics = {'rewards': [], 'backlogs': [], 'energies': []}

        for ep in range(num_episodes):
            res = trainer.env.reset()
            obs_upper = res['upper']
            prev_lower = res['lower']
            mf_dim = trainer.num_nodes + trainer.max_models
            prev_lower["mean_field"] = torch.zeros(
                (trainer.num_terminals, mf_dim), device=trainer.device)
            upper_state = self.build_upper_state(trainer, obs_upper)

            ep_r, ep_b, ep_e = 0.0, [], 0.0

            for slot in range(max_slots):
                if trainer.env.time_manager.is_new_frame():
                    u_a, _, _ = self.get_upper_actions(
                        trainer, upper_state, obs_upper)
                    trainer.env.step_upper(u_a)

                t_idx, s_idx, bs, t_acc, t_dl = \
                    trainer.workload_gen.generate_step()

                if len(t_idx) > 0:
                    n_idx, m_idx, masks, _, _ = self.get_lower_actions(
                        trainer, prev_lower, t_idx, s_idx,
                        t_acc, t_dl, bs)
                    results = trainer.env.step_lower(
                        t_idx, s_idx, bs, n_idx, m_idx, t_dl, t_acc)

                    ep_r += results['reward_global']
                    ep_b.append(
                        results['obs']['backlog'].sum().item())
                    ep_e += results.get(
                        'energy',
                        results['info'].get('energy', 0.0))

                    mf = self.compute_lower_mean_fields(
                        trainer, t_idx, s_idx, n_idx, m_idx)
                    results['mean_field'] = \
                        prev_lower['mean_field'].clone()
                    results['mean_field'][t_idx] = mf
                    prev_lower = results
                else:
                    trainer.env.time_manager.tick()

                if trainer.env.time_manager.is_new_frame():
                    ru = trainer.env.collect_upper_metrics()
                    obs_upper = ru
                    upper_state = self.build_upper_state(trainer, ru)

            metrics['rewards'].append(ep_r)
            metrics['backlogs'].append(
                np.mean(ep_b) if ep_b else 0)
            metrics['energies'].append(ep_e)
            print(f"  Ep {ep+1}: Reward={ep_r:.2f}, "
                  f"Backlog={metrics['backlogs'][-1]:.2f}")

        self.generate_report(metrics)
        self.is_evaluating = False

    def generate_report(self, metrics):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = f"evaluation_report_{ts}.md"
        avg_r = np.mean(metrics['rewards'])
        avg_b = np.mean(metrics['backlogs'])
        avg_e = np.mean(metrics['energies'])

        content = f"""# PPO Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Config: {self.max_train_steps} train steps, collect-then-train

## Summary ({len(metrics['rewards'])} episodes)
| Metric | Average |
|:---|:---|
| **Reward** | {avg_r:.4f} |
| **Backlog** | {avg_b:.4f} |
| **Energy** | {avg_e:.4f} |

## Per-Episode
"""
        for i in range(len(metrics['rewards'])):
            content += (f"- Ep {i+1}: R={metrics['rewards'][i]:.2f}, "
                        f"B={metrics['backlogs'][i]:.2f}, "
                        f"E={metrics['energies'][i]:.2f}\n")

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"\n[Eval] Report: {path}")
