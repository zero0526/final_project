import torch
from matrix_source.agents.ppo import PPOAgent
from matrix_source.agents.ppo_scaffold_v3 import PPOSCAFFOLDREPAgent

from matrix_source.trainers.strategies import AlgorithmStrategy
from matrix_source.trainers.train import log_transform
from matrix_source.utils.math_utils import to_binary
from tqdm import tqdm
import os
import numpy as np
from datetime import datetime


def compute_gae(rewards, next_values, values, dones, agent_ids, gamma, lmbda):
    """
    Generalized Advantage Estimation (GAE)
    Vectorized mask calculation to avoid CPU-GPU syncs in the loop.
    """
    device = rewards.device
    num_steps = rewards.size(0)

    deltas = rewards + gamma * next_values * (1 - dones) - values
    advantages = torch.zeros_like(deltas)

    # Pre-calculate combined masks for resets (dones OR agent boundaries)
    # boundary_mask[t] = 0 if agent_ids[t] != agent_ids[t+1] else 1
    masks = (1 - dones) * (gamma * lmbda)
    boundary_mask = torch.ones(num_steps, device=device)
    if num_steps > 1:
        boundary_mask[:-1] = (agent_ids[:-1] == agent_ids[1:]).float()

    combined_mask = masks * boundary_mask

    curr_advantage = 0
    # The loop is still needed for GAE, but we avoid indexing agent_ids and if-checks.
    # By using pre-calculated combined_mask, we minimize syncs.
    for t in reversed(range(num_steps)):
        curr_advantage = deltas[t] + curr_advantage * (combined_mask[t] if t < num_steps - 1 else 0)
        advantages[t] = curr_advantage

    return advantages


class PPOSCAFFOLDREPStrategy(AlgorithmStrategy):
    def __init__(self):
        super().__init__()
        self.lower_train_num = 0
        self.upper_train_num = 0
        self.alt_train_num = 0
        self.alt_next = 'UPPER'
        self.upper_mf_ema = None
        self.mf_ema_alpha = 0.7

        # Hyperparams from user (Strict 5-Cycle Curriculum)
        self.cycle_configs = {
            1: {'lower': 15, 'upper': 10, 'zeta': 1.0, 'det': False},
            2: {'lower': 12, 'upper': 8, 'zeta': 1.0, 'det': False},
            3: {'lower': 10, 'upper': 7, 'zeta': 1.0, 'det': False},
            4: {'lower': 8, 'upper': 5, 'zeta': 1.0, 'det': False},
            5: {'lower': 6, 'upper': 4, 'zeta': 1.0, 'det': True}
        }

        self.lower_cfg = {'min_size': 4096, 'batch': 128, 'epochs': 7}
        self.upper_cfg = {'min_size': 512, 'batch': 64, 'epochs': 5}

        self.cycle_num = 1
        self.max_cycles = 5

        # Initial settings for Cycle 1
        cfg = self.cycle_configs[self.cycle_num]
        self.lower_warmup_steps = cfg['lower']
        self.upper_warmup_steps = cfg['upper']
        self.current_zeta = cfg['zeta']

        self.phase = 'LOWER_ONLY'
        self.current_phase_updates = 0
        self.entropy_decay_rate = 0.99
        self.is_evaluating = False

        # Frame-level accumulators for Minimalist Upper State features (V_cum, V_nbr)
        # These are reset at the start of each Time Frame (is_new_frame)
        self._frame_arrivals = None   # (num_nodes,)  – total task arrivals per node
        self._frame_hw_fails = None   # (num_nodes,)  – hardware-limit failures per node
        self._frame_expired  = None   # (num_nodes,)  – queue-expired failures per node
        # Cached spatio-temporal features (computed at end-of-frame, used next frame)
        self._v_cum = None            # (num_nodes,)  – cumulative failure intensity
        self._v_nbr = None            # (num_nodes,)  – peer-average failure intensity

    def initialize_agents(self, trainer):
        # 1. Upper Agent (Minimalist State: 2*S+1,  Augmented MF: S+2)
        trainer.upper_state_dim  = trainer.num_services * 2 + 1
        trainer.upper_action_dim = trainer.num_services + 2   # augmented MF dim

        trainer.shared_upper_agent = PPOAgent(
            node_id=-2, node_type="Edge_Group",
            state_dim=trainer.upper_state_dim,
            action_dim=trainer.upper_action_dim,
            u_action_dim=trainer.upper_u_action_dim,
            mf_hidden_sizes=tuple(trainer.config.hyper_neural["MF_HIDDEN_LAYER"]),
            mf_lr=float(trainer.config.hyper_neural['MF_LR']),
            buffer_min_size=self.upper_cfg['min_size'],
            hidden_sizes=trainer.config.hyper_neural['AGENT_HIDDEN_LAYER'],
            lr=float(trainer.config.hyper_neural['UPPER_LR']),
            gamma=trainer.config.hyper_neural['DISCOUNT_FACTOR'],
            lam=trainer.config.hyper_neural.get('LAMBDA', 0.95),
            clip_eps=trainer.config.hyper_neural.get('CLIP_EPS', 0.2),
            k_epochs=self.upper_cfg['epochs'],
            batch_size=self.upper_cfg['batch'],
            num_instances=trainer.num_edge_agents,
            device=trainer.device
        )

        # 2. Lower Agent
        trainer.shared_lower_agent = PPOSCAFFOLDREPAgent(
            node_id=-1, node_type="Terminal_Group",
            state_dim=trainer.lower_state_dim,
            action_dim=trainer.lower_action_dim,
            u_action_dim=trainer.lower_u_action_dim,
            mf_hidden_sizes=tuple(trainer.config.hyper_neural["MF_HIDDEN_LAYER"]),
            mf_lr=float(trainer.config.hyper_neural['MF_LR']),
            buffer_min_size=self.lower_cfg['min_size'],
            num_groups=trainer.num_edge_agents,
            hidden_sizes=tuple(trainer.config.hyper_neural['AGENT_HIDDEN_LAYER']),
            lr=float(trainer.config.hyper_neural['LOWER_LR']),
            gamma=trainer.config.hyper_neural['DISCOUNT_FACTOR'],
            lam=trainer.config.hyper_neural.get('LAMBDA', 0.95),
            clip_eps=trainer.config.hyper_neural.get('CLIP_EPS', 0.2),
            k_epochs=self.lower_cfg['epochs'],
            batch_size=self.lower_cfg['batch'],
            num_instances=trainer.num_terminals,
            device=trainer.device
        )

        # 3. Compute Terminal-to-Group Mapping from Environment
        # terminal_to_comp_node_map corresponds to (num_terminals, num_comp_nodes)
        mapping_matrix = trainer.env.static_matrices['terminal_to_comp_node_map']
        terminal_node_indices = mapping_matrix.argmax(dim=1)

        terminal_to_group = torch.zeros(trainer.num_terminals, dtype=torch.long, device=trainer.device)
        for k in range(trainer.num_terminals):
            node_idx = int(terminal_node_indices[k])
            # Map node index to edge agent index using node_to_instance
            terminal_to_group[k] = trainer.node_to_instance.get(node_idx, 0)

        trainer.terminal_to_group = terminal_to_group

        # 4. Initialize Cluster Mapping (Edge Node -> List of Terminals)
        self.node_to_terminals = {}
        for t_idx, g_idx in enumerate(terminal_to_group.tolist()):
            if g_idx not in self.node_to_terminals:
                self.node_to_terminals[g_idx] = []
            self.node_to_terminals[g_idx].append(t_idx)

        print(f"[PPOSCAFFOLDREPStrategy] Initialized {len(self.node_to_terminals)} SCAFFOLD clusters.")

        # 5. Initialise accumulators with env dimensions
        num_nodes = trainer.num_nodes
        dev = trainer.device
        self._frame_arrivals = torch.zeros(num_nodes, device=dev)
        self._frame_hw_fails = torch.zeros(num_nodes, device=dev)
        self._frame_expired  = torch.zeros(num_nodes, device=dev)
        self._v_cum = torch.zeros(num_nodes, device=dev)
        self._v_nbr = torch.zeros(num_nodes, device=dev)

        # 6. Initial Phase Jump (if warmup is 0)
        if self.phase == 'LOWER_ONLY' and self.lower_warmup_steps == 0:
            self.phase = 'UPPER_ONLY'
            print(f"[Curriculum] Initial skip: LOWER_ONLY -> UPPER_ONLY")

        if self.phase == 'UPPER_ONLY' and self.upper_warmup_steps == 0:
            pass

    # ------------------------------------------------------------------
    # Minimalist Upper State helpers
    # ------------------------------------------------------------------
    def build_upper_state(self, trainer, obs_upper):
        """Returns (num_nodes, 2*S+1) state: [actions, phi_prob, cpu_total_util]."""
        actions  = obs_upper['actions']   # (N, S)
        phi      = obs_upper['phi_prob']  # (N, S)

        # Total CPU utilisation = sum(cpu_alloc) / cpu_max  (scalar per node)
        cpu_alloc = trainer.env.engine.cpu_alloc_matrix    # (N, S)
        cpu_max   = trainer.env.engine.resource_specs[:, 0].clamp(min=1.0)  # (N,)
        cpu_util  = cpu_alloc.sum(dim=1) / cpu_max         # (N,)
        return torch.cat([actions, phi, cpu_util.unsqueeze(-1)], dim=-1)  # (N, 2S+1)

    def _compute_upper_mf(self, trainer, obs_upper):
        """Augmented mean field: [neighbor_avg_actions, V_cum, V_nbr]."""
        # Env-provided neighbor-average placement actions (N, S)
        env_mf = obs_upper.get(
            'mean_fields',
            torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device)
        )  # (N, S)

        # Spatio-temporal failure features (N,) → unsqueeze to (N,1)
        v_cum = self._v_cum.unsqueeze(-1)  # (N, 1)
        v_nbr = self._v_nbr.unsqueeze(-1)  # (N, 1)

        return torch.cat([env_mf, v_cum, v_nbr], dim=-1)  # (N, S+2)

    def _reset_frame_accumulators(self, trainer):
        self._frame_arrivals.zero_()
        self._frame_hw_fails.zero_()
        self._frame_expired.zero_()

    def _update_frame_accumulators(self, trainer, step_info):
        """Accumulate per-step node-level metrics. Call after each step_lower."""
        # arrivals: (N, S) → sum over services → (N,)
        arrival_mat = step_info.get('arrival_matrix',
                                    torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device))
        self._frame_arrivals += arrival_mat.sum(dim=1)

        # hw failures: fail_hw is (N, S)
        fail_hw = trainer.env.engine.fail_hw  # (N, S) – already zeroed & filled inside process_arrivals
        self._frame_hw_fails += fail_hw.sum(dim=1)

        # expired in queue: violate_qos minus immediate_fails = expired count
        violate_qos = step_info.get('violate_qos',
                                    torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device))
        immediate   = trainer.env.engine.immediate_fails   # (N, S)
        expired_mat = (violate_qos - immediate).clamp(min=0)
        self._frame_expired += expired_mat.sum(dim=1)

    def _finalize_frame_features(self, trainer):
        """Compute V_cum and V_nbr at the end of a Time Frame."""
        total_fails = self._frame_hw_fails + self._frame_expired   # (N,)
        arrivals    = self._frame_arrivals.clamp(min=1.0)
        self._v_cum = total_fails / arrivals                        # (N,)

        adj         = trainer.env.engine.adj_matrix                # (N, N)
        nbr_count   = adj.sum(dim=1).clamp(min=1.0)                # (N,)
        self._v_nbr = (adj @ self._v_cum.unsqueeze(-1)).squeeze(-1) / nbr_count  # (N,)

    def get_upper_actions(self, trainer, current_upper_state, obs_upper):
        act_matrix = torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device)

        # 1. Build augmented MF and update EMA
        mf_augmented = self._compute_upper_mf(trainer, obs_upper)  # (N, S+2)
        if self.upper_mf_ema is None:
            self.upper_mf_ema = mf_augmented.clone()
        else:
            self.upper_mf_ema = (1 - self.mf_ema_alpha) * self.upper_mf_ema + self.mf_ema_alpha * mf_augmented

        edge_states = current_upper_state[trainer.edge_node_ids]
        edge_mfs    = self.upper_mf_ema[trainer.edge_node_ids]
        instance_indices = torch.tensor([trainer.node_to_instance[nid] for nid in trainer.edge_node_ids],
                                        device=trainer.device)

        # Use current curriculum config for deterministic flag and zeta
        cfg = self.cycle_configs.get(self.cycle_num, self.cycle_configs[1])
        is_det = self.is_evaluating or cfg['det']
        zeta = cfg['zeta']

        batch_a_ids, log_probs, values = trainer.shared_upper_agent.choose_action_batch(
            edge_states, edge_mfs, agent_indices=instance_indices, deterministic=is_det, zeta=zeta
        )

        for i, nid in enumerate(trainer.edge_node_ids):
            act_matrix[nid] = torch.tensor(to_binary(batch_a_ids[i], trainer.num_services), device=trainer.device)

        for nid in trainer.env.static_matrices.get("cloud_ids", []):
            act_matrix[nid] = torch.ones(trainer.num_services, device=trainer.device)
        return act_matrix, log_probs, values

    def _get_batch_placements(self, trainer, s_idx, num_reqs, placement_matrix=None):
        if placement_matrix is None:
            placement_matrix = trainer.env.engine.placement_matrix

        placements = placement_matrix[:, s_idx]  # (num_nodes, ...)
        return placements.T  # (num_reqs, num_nodes)

    def calculate_lower_masks(self, trainer, t_idx, s_idx, tasks_min_accuracy, placement_matrix=None):
        num_reqs = len(t_idx)
        current_placements = self._get_batch_placements(trainer, s_idx, num_reqs, placement_matrix)

        # Action space: (num_reqs, num_nodes, max_models)
        node_model_mask = current_placements.unsqueeze(-1).expand(-1, -1, trainer.max_models)
        masks = node_model_mask.reshape(num_reqs, -1)

        # Safety: if no node is valid, allow all to prevent NaNs in softmax
        invalid_mask_rows = (masks.sum(dim=1) == 0)
        if invalid_mask_rows.any():
            masks = masks.clone()
            masks[invalid_mask_rows] = 1.0
        return masks

    # ------------------------------------------------------------------
    # Mean-field helpers
    # ------------------------------------------------------------------
    def build_lower_state(self, trainer, obs, t_idx, s_idx):
        """Build a normalised state tensor from an obs dict for a batch of tasks.

        Args:
            obs   : observation dict returned by env (has 'task_reqs', 'backlog', 'cpu_alloc')
            t_idx : 1-D LongTensor – terminal indices of each task
            s_idx : 1-D LongTensor (same length as t_idx) – service index for each task

        Returns:
            states : (len(t_idx), state_dim) float tensor
        """
        placement = trainer.env.engine.placement_matrix[:, s_idx]  # (num_nodes, B)
        b_masked = (obs['backlog'][:, s_idx]).T  # (B, num_nodes)
        c_masked = (obs['cpu_alloc'][:, s_idx] * placement).T  # (B, num_nodes)
        st = torch.cat([obs['task_reqs'][t_idx], b_masked, c_masked], dim=1)
        st = st.clone()
        st[:, 0] /= trainer.config.norm_data_size
        st[:, 1] /= 100.0
        if st.shape[1] > 4:
            st[:, 4:4 + 2 * trainer.num_nodes] /= trainer.config.norm_gflop
        return st

    def compute_lower_mean_fields(self, trainer, t_idx, s_idx, n_idx, m_idx):
        """Compute per-task **local** mean fields using two-hot action encoding.

        For every unique service group in s_idx the tasks that share the same
        service are treated as one cooperative group. The *global* mean field
        of the group is the sum of two-hot vectors (n_idx one-hot ||  m_idx
        one-hot). Each task's *local* mean field is then the group sum minus
        its own vector, normalised by (group_size - 1).  Tasks that are alone
        in their service group receive a zero mean field.

        Args:
            t_idx : (B,) terminal indices      (not used for MF calc, forwarded)
            s_idx : (B,) service indices        – determines the groups
            n_idx : (B,) chosen node indices    – first part of two-hot
            m_idx : (B,) chosen model indices   – second part of two-hot

        Returns:
            local_mfs : (B, num_nodes + max_models) float tensor
        """
        B = len(t_idx)
        mf_dim = trainer.num_nodes + trainer.max_models
        device = trainer.device

        # cal meanfield
        two_hot = torch.zeros(B, mf_dim, device=device)
        arange_b = torch.arange(B, device=device)
        # node one-hot
        two_hot[arange_b, n_idx.long()] = 1.0
        # model one-hot
        two_hot[arange_b, trainer.num_nodes + m_idx.long()] = 1.0

        # num_services should cover all possible ids
        num_services = int(s_idx.max().item()) + 1

        # group_sizes[s] = number of tasks in service s
        group_sizes = torch.bincount(
            s_idx,
            minlength=num_services
        ).float()

        # group_sums[s] = sum of all two_hot vectors in service s
        group_sums = torch.zeros(
            num_services,
            mf_dim,
            device=device
        )

        group_sums.index_add_(0, s_idx, two_hot)

        sampled_group_sums = group_sums[s_idx]  # (B, mf_dim)
        sampled_group_sizes = group_sizes[s_idx]  # (B,)

        # local_mf_i = (group_sum - self) / (n - 1)
        denom = (sampled_group_sizes - 1).clamp(min=1)

        local_mfs = (sampled_group_sums - two_hot) / denom.unsqueeze(1)

        # groups with only 1 member -> zero vector
        local_mfs = torch.where(
            (sampled_group_sizes > 1).unsqueeze(1),
            local_mfs,
            torch.zeros_like(local_mfs)
        )

        return local_mfs

    def get_lower_actions(self, trainer, res_lower, mf, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes):
        # Dùng hàm build_lower_state đã chuẩn hóa thay vì code lại
        states = self.build_lower_state(trainer, res_lower['obs'], t_idx, s_idx)
        masks = self.calculate_lower_masks(trainer, t_idx, s_idx, tasks_min_accuracy)

        # Correctly slice MF and indices for the current batch
        mfs = mf[t_idx] if mf.shape[0] == trainer.num_terminals else mf

        # Use current curriculum config for deterministic flag and zeta
        cfg = self.cycle_configs.get(self.cycle_num, self.cycle_configs[1])
        is_det = self.is_evaluating or cfg['det']
        zeta = cfg['zeta']

        # Get terminal-to-group mapping for shared actions
        batch_group_indices = trainer.terminal_to_group[t_idx]

        batch_actions, log_probs, values = trainer.shared_lower_agent.choose_action_batch(
            states, mfs, masks_batch=masks,
            agent_indices=t_idx, group_indices=batch_group_indices,
            deterministic=is_det, zeta=zeta
        )

        a_ids = batch_actions.view(-1).long()
        n_idx = a_ids // trainer.max_models
        m_idx = a_ids % trainer.max_models

        # Trả về thêm 'states' để lưu buffer, tránh lỗi Staleness
        return n_idx, m_idx, masks, log_probs, values, states

    def store_lower_transitions(self, trainer, next_res,
                                prev_state, prev_mf, curr_state, curr_mf,
                                t_idx, n_idx, m_idx, masks, log_probs, values):
        """Store a lower-level transition and optionally train the MF predictor.

        Args:
            next_res    : env result dict after step_lower (for reward / done / metrics)
            prev_state  : (B, state_dim) – state *before* the action was taken
            prev_mf     : (B, mf_dim)   – local mean field at previous state
            curr_state  : (B, state_dim) – state *after* the action was taken
            curr_mf     : (B, mf_dim)   – local mean field at current state
            t_idx       : (B,) terminal indices
            n_idx       : (B,) chosen node indices
            m_idx       : (B,) chosen model indices
            masks       : (B, action_dim) validity mask
            log_probs   : (B,) log-probabilities from the actor
            values      : (B,) critic values
        """

        # 1. Extract metrics and rewards
        reward = next_res['reward']
        reward -= next_res["obs"]["virtual_drift"]
        rew_divisor = trainer.config.norm_lower_rw
        norm_rew = log_transform(reward / (rew_divisor if rew_divisor != 0 else 1.0))

        # 2. Handle MF training and transition storage
        avg_mf_loss = 0.0
        is_frozen = (self.phase == 'UPPER_ONLY')

        if not is_frozen and not self.is_evaluating:
            done = torch.tensor([next_res["new_frame"]] * len(t_idx), dtype=torch.float32, device=trainer.device)
            rewards = torch.full((len(t_idx),), norm_rew, dtype=torch.float32, device=trainer.device)
            a_ids = (n_idx * trainer.max_models + m_idx).long()

            # Get group_ids for MF training and storage
            t_group_ids = trainer.terminal_to_group[t_idx]

            # Explicitly train MF network if requested (returning item loss)
            avg_mf_loss = trainer.shared_lower_agent.learn_mf_batch(
                prev_state, prev_mf, curr_mf, t_group_ids
            )

            # Store transition in agent memory
            trainer.shared_lower_agent.memory.add_batch(
                prev_state, prev_mf, curr_mf, a_ids, rewards, curr_state, done,
                log_probs, values, agent_ids=t_idx, masks=masks
            )

        # 3. ALWAYS record metrics!
        trainer.aggregator.add_lower(next_res, mf_loss=avg_mf_loss,
                                     state=curr_state[0] if (curr_state is not None and len(curr_state) > 0) else None)

    def store_upper_transitions(self, trainer, s_all, ns_all, current_res, next_res, acts_matrix, log_probs, values,
                                is_done):
        # 1. Extract global metrics
        reward = next_res['reward_global']
        rew_divisor = trainer.config.norm_upper_rw
        norm_rew = log_transform(reward / (rew_divisor if rew_divisor != 0 else 1.0))

        # 2. Handle MF training and transition storage (if NOT frozen and NOT evaluating)
        avg_mf_loss = 0.0
        is_frozen = (self.phase == 'LOWER_ONLY')

        # Safely extract edge_states for metric recording
        edge_states = s_all[trainer.edge_node_ids] if s_all is not None else None

        if not is_frozen and not self.is_evaluating:
            edge_next_states = ns_all[trainer.edge_node_ids]
            dones = torch.full((trainer.num_edge_agents,), 1.0 if is_done else 0.0, dtype=torch.float32,
                               device=trainer.device)

            # Use same EMA consistency logic for storage
            next_raw_mf = next_res['mean_fields']
            if self.upper_mf_ema is None:
                self.upper_mf_ema = next_raw_mf
            next_ema = (1 - self.mf_ema_alpha) * self.upper_mf_ema + self.mf_ema_alpha * next_raw_mf

            edge_c_mfs = self.upper_mf_ema[trainer.edge_node_ids]
            edge_n_mfs = next_ema[trainer.edge_node_ids]
            edge_acts = acts_matrix[trainer.edge_node_ids]

            pw2 = 2 ** torch.arange(trainer.num_services - 1, -1, -1, device=trainer.device).float()
            edge_a_ids = (edge_acts * pw2).sum(dim=1).long()

            rewards = torch.full((trainer.num_edge_agents,), norm_rew, dtype=torch.float32, device=trainer.device)
            instance_indices = torch.tensor([trainer.node_to_instance[nid] for nid in trainer.edge_node_ids],
                                            device=trainer.device)

            avg_mf_loss = trainer.shared_upper_agent.store_transition_train_mf_batch(
                edge_states, edge_c_mfs, next_raw_mf[trainer.edge_node_ids], edge_a_ids, rewards, edge_next_states,
                dones, agent_ids=instance_indices, log_prob=log_probs, value=values
            )

        # 3. ALWAYS record metrics!
        agg_state = edge_states[0] if (edge_states is not None and len(edge_states) > 0) else None
        trainer.aggregator.add_upper(next_res, mf_loss=avg_mf_loss, state=agg_state)

        # If evaluating, return step metrics
        if self.is_evaluating:
            return {
                'reward': next_res['reward_global'],
                'backlog': next_res['obs']['backlog'].sum().item(),
                'energy': next_res['info'].get('energy', 0.0)  # Assume energy is in info
            }
        return None

    def perform_scaffold_aggregation(self, agent: PPOSCAFFOLDREPAgent):
        """
        Cluster-Based Federated Aggregation for PPO (split backbone/head).
        Synchronizes terminals connected to the same Edge Node.
        """
        with torch.no_grad():
            for group_idx, terminal_ids in self.node_to_terminals.items():
                if not terminal_ids:
                    continue

                t_ids = torch.tensor(terminal_ids, device=agent.device)

                # Step 1: Finalize local control variates using accumulated gradients
                agent.update_local_cvariates(t_ids)

                # Step 2: Average backbone weights within the cluster
                # This implements Federated Representation Learning for the backbone
                bone_params = list(agent.critic.backbone.parameters())
                for p in bone_params:
                    # Compute mean of weights for terminals in this cluster
                    cluster_mean = p.data[t_ids].mean(dim=0, keepdim=True)
                    # Broadcast average back to all cluster members
                    p.data[t_ids] = cluster_mean.expand(len(terminal_ids), *cluster_mean.shape[1:])

                # Step 3: Aggregate backbone control variates to update the global correction term
                c_b_local_slices = agent.get_c_b_local(t_ids)
                # Global variate for the cluster is the average of local variates
                c_b_new_global = [c.mean(dim=0, keepdim=True).expand(len(terminal_ids), *c.shape[1:])
                                  for c in c_b_local_slices]
                agent.set_c_b_global(t_ids, c_b_new_global)

        # Step 4: Reset round-specific accumulators (gradients and step counts)
        agent.save_base_initial()

    def run_training(self, trainer):
        max_slots = trainer.env.time_manager.max_steps
        ep = 0

        pbar = tqdm(total=self.max_cycles, desc="Sequential Refinement Progress")

        # mf_dim for lower agents = num_nodes + max_models (two-hot encoding)
        lower_mf_dim = trainer.num_nodes + trainer.max_models

        while self.cycle_num <= self.max_cycles:
            obs = trainer.env.reset()
            obs_upper = obs['upper']
            init_lower_obs = obs['lower']
            current_upper_state = self.build_upper_state(trainer, obs_upper)

            # Persistent lower-level state and mean-field across steps
            # Will be properly initialised on the first step with real tasks
            prev_lower_obs = init_lower_obs  # raw obs dict for state building
            prev_lower_state = None  # (num_terminals, state_dim) – lazily initialised
            prev_lower_mf = None  # (num_terminals, mf_dim)

            for slot in range(max_slots):
                if trainer.env.time_manager.is_new_frame():
                    # Finalize V_cum / V_nbr from the previous frame's data,
                    # then reset counters for the new frame.
                    self._finalize_frame_features(trainer)
                    self._reset_frame_accumulators(trainer)

                    u_acts_matrix, u_log_probs, u_values = self.get_upper_actions(trainer, current_upper_state,
                                                                                  obs_upper)
                    trainer.env.step_upper(u_acts_matrix)

                t_idx, s_idx, batch_sizes, tasks_min_accuracy, task_deadlines = trainer.workload_gen.generate_step()
                if len(t_idx) > 0:
                    # --- (A) Build a temporary obs-dict so get_lower_actions can use prev obs ---
                    # get_lower_actions reads mf from the obs dict we pass in
                    prev_lower_mf = (
                        prev_lower_mf if prev_lower_mf is not None
                        else torch.zeros((trainer.num_terminals, lower_mf_dim), device=trainer.device)
                    )
                    prev_lower_obs['mean_field'] = prev_lower_mf
                    n_idx, m_idx, masks, l_log_probs, l_values, fresh_prev_states = self.get_lower_actions(
                        trainer, prev_lower_obs, prev_lower_mf, t_idx, s_idx,
                        tasks_min_accuracy, task_deadlines, batch_sizes)

                    # --- (B) Compute local mean fields from the chosen actions ---
                    curr_lower_mf = self.compute_lower_mean_fields(trainer, t_idx, s_idx, n_idx, m_idx)

                    # --- (C) Step the environment ---
                    results = trainer.env.step_lower(t_idx, s_idx, batch_sizes, n_idx, m_idx, task_deadlines,
                                                     tasks_min_accuracy)

                    # --- (D) Build curr state from the environment's next obs ---
                    curr_lower_state = self.build_lower_state(trainer, results['obs'], t_idx, s_idx)

                    # --- (D1) Accumulate frame-level failure metrics for V_cum ---
                    self._update_frame_accumulators(trainer, results['info'])

                    # --- (E) Slice prev state/mf for this batch's tasks ---
                    if prev_lower_mf is None:
                        # Very first step: no history yet → zeros for both
                        _prev_mf = torch.zeros(len(t_idx), lower_mf_dim, device=trainer.device)
                    else:
                        _prev_mf = prev_lower_mf[t_idx]  # (B, mf_dim)

                    # --- (F) Store transition with correctly paired (prev, curr) pairs ---
                    self.store_lower_transitions(
                        trainer, results,
                        fresh_prev_states, _prev_mf,
                        curr_lower_state, curr_lower_mf,
                        t_idx, n_idx, m_idx, masks, l_log_probs, l_values
                    )

                    trainer.aggregator.add_step_matrices(
                        f_alloc=trainer.env.engine.cpu_alloc_matrix,
                        arrivals=results['info']['arrival_matrix'],
                        backlog=trainer.env.engine.backlog_queue.sum(dim=-1)
                    )

                    # --- (G) Roll prev ← curr: scatter back to (num_terminals, *) ---
                    def scatter_update(base, idx, val):
                        out = base.clone()
                        out[idx] = val
                        return out

                    if prev_lower_state is None:
                        prev_lower_state = torch.zeros(
                            trainer.num_terminals,
                            curr_lower_state.shape[1],
                            device=trainer.device
                        )
                    if prev_lower_mf is None:
                        prev_lower_mf = torch.zeros(
                            trainer.num_terminals,
                            lower_mf_dim,
                            device=trainer.device
                        )
                    prev_lower_state = scatter_update(
                        prev_lower_state,
                        t_idx,
                        curr_lower_state
                    )
                    prev_lower_mf = scatter_update(
                        prev_lower_mf,
                        t_idx,
                        curr_lower_mf
                    )
                    prev_lower_obs = results

                    # 1. Train Lower Level (ONLY in Phase LOWER_ONLY)
                    if self.phase == 'LOWER_ONLY':
                        # Get current zeta from config
                        cfg = self.cycle_configs.get(self.cycle_num, self.cycle_configs[1])
                        zeta = cfg['zeta']

                        loss = trainer.shared_lower_agent.learn(
                            agents_ids=torch.arange(trainer.num_terminals, device=trainer.device),
                            group_ids=trainer.terminal_to_group,
                            zeta=zeta
                        )
                        if loss is not None:
                            trainer.total_lower_steps += 1
                            self.lower_train_num += 1
                            self.current_phase_updates += 1
                            trainer.aggregator.record_td_losses(lower_losses=loss)

                            # Checkpoint
                            if self.lower_train_num % 10 == 0:
                                os.makedirs('checkpoints', exist_ok=True)
                                trainer.shared_lower_agent.save(f'checkpoints/ppo_lower_{self.lower_train_num}.pth')

                            # Phase Transition: LOWER_ONLY -> UPPER_ONLY
                            if self.current_phase_updates >= self.lower_warmup_steps:
                                self.phase = 'UPPER_ONLY'
                                self.current_phase_updates = 0
                                print(f"\n[Cycle {self.cycle_num}] LOWER Phase Complete. Switching to UPPER training.")
                else:
                    trainer.env.time_manager.tick()

                if trainer.env.time_manager.is_new_frame():
                    res_upper = trainer.env.collect_upper_metrics()
                    next_upper_state = self.build_upper_state(trainer, res_upper)
                    is_ep_done = (slot == max_slots - 1)

                    self.store_upper_transitions(trainer, current_upper_state, next_upper_state, obs_upper, res_upper,
                                                 u_acts_matrix, u_log_probs, u_values, is_ep_done)

                    # 2. Train Upper Level (ONLY in Phase UPPER_ONLY)
                    if self.phase == 'UPPER_ONLY':
                        # Get current zeta from config
                        cfg = self.cycle_configs.get(self.cycle_num, self.cycle_configs[1])
                        zeta = cfg['zeta']

                        loss = trainer.shared_upper_agent.learn(
                            torch.arange(trainer.num_edge_agents, device=trainer.device),
                            zeta=zeta
                        )
                        if loss is not None:
                            trainer.total_upper_steps += 1
                            self.upper_train_num += 1
                            self.current_phase_updates += 1
                            trainer.aggregator.record_td_losses(upper_losses=loss)

                            # Checkpoint
                            if self.upper_train_num % 10 == 0:
                                os.makedirs('checkpoints', exist_ok=True)
                                trainer.shared_upper_agent.save(f'checkpoints/ppo_upper_{self.upper_train_num}.pth')

                            # Phase Transition: UPPER_ONLY -> LOWER_ONLY (Cycle End)
                            if self.current_phase_updates >= self.upper_warmup_steps:
                                self.phase = 'LOWER_ONLY'
                                self.current_phase_updates = 0

                                # End of a full Lower-Upper pair cycle
                                pbar.update(1)
                                self.cycle_num += 1

                                # Update parameters for next cycle if exists
                                if self.cycle_num <= self.max_cycles:
                                    n_cfg = self.cycle_configs[self.cycle_num]
                                    self.lower_warmup_steps = n_cfg['lower']
                                    self.upper_warmup_steps = n_cfg['upper']
                                    print(
                                        f"\n[Curriculum] Cycle {self.cycle_num - 1} Complete. Starting Cycle {self.cycle_num}")
                                    print(
                                        f"Config: Lower={self.lower_warmup_steps}, Upper={self.upper_warmup_steps}, Zeta={n_cfg['zeta']}, Det={n_cfg['det']}")
                                else:
                                    print(f"\n[Curriculum] All {self.max_cycles} Cycles Complete.")

                    current_upper_state = next_upper_state
                    obs_upper = res_upper

            # 3. Synchronize Clusters (SCAFFOLD Aggregation)
            self.perform_scaffold_aggregation(trainer.shared_lower_agent)

            trainer.aggregator.store_history()
            trainer.aggregator.report_episode(ep)
            trainer.aggregator.reset_episode()
            print(f"--- Curriculum Status ---")
            print(
                f"Cycle: {self.cycle_num} | Phase: {self.phase} | Phase Progress: {self.current_phase_updates}/{self.lower_warmup_steps if self.phase == 'LOWER_ONLY' else self.upper_warmup_steps}")
            ep += 1
        pbar.close()

        # --- Start Post-Training Evaluation ---
        self.run_evaluation(trainer, num_episodes=5)

    def run_evaluation(self, trainer, num_episodes=5):
        print(f"\n--- Starting Post-Training Evaluation ({num_episodes} Episodes) ---")
        self.is_evaluating = True
        max_slots = trainer.env.time_manager.max_steps

        eval_metrics = {
            'rewards': [],
            'backlogs': [],
            'energies': [],
            'success_rates': []
        }

        lower_mf_dim = trainer.num_nodes + trainer.max_models

        for ep in range(num_episodes):
            res = trainer.env.reset()
            obs_upper, init_lower_obs = res['upper'], res['lower']
            current_upper_state = self.build_upper_state(trainer, obs_upper)

            prev_lower_obs = init_lower_obs
            prev_lower_state = None  # (num_terminals, state_dim)
            prev_lower_mf = None  # (num_terminals, mf_dim)

            ep_reward = 0
            ep_backlog = []
            ep_energy = 0

            for slot in range(max_slots):
                if trainer.env.time_manager.is_new_frame():
                    # Finalize and reset accumulators – same as training
                    self._finalize_frame_features(trainer)
                    self._reset_frame_accumulators(trainer)

                    u_acts_matrix, u_log_probs, u_val = self.get_upper_actions(trainer, current_upper_state, obs_upper)
                    trainer.env.step_upper(u_acts_matrix)


                t_idx, s_idx, batch_sizes, tasks_min_accuracy, task_deadlines = trainer.workload_gen.generate_step()
                if len(t_idx) > 0:
                    # --- (A) Build temporary obs-dict for evaluation ---
                    prev_lower_mf = (
                        prev_lower_mf if prev_lower_mf is not None
                        else torch.zeros((trainer.num_terminals, lower_mf_dim), device=trainer.device)
                    )
                    prev_lower_obs['mean_field'] = prev_lower_mf

                    n_idx, m_idx, masks, l_log_probs, l_vals, fresh_prev_states = self.get_lower_actions(
                        trainer, prev_lower_obs, prev_lower_mf, t_idx, s_idx,
                        tasks_min_accuracy, task_deadlines, batch_sizes)

                    # --- (B) Compute local mean fields from the chosen actions ---
                    curr_lower_mf = self.compute_lower_mean_fields(trainer, t_idx, s_idx, n_idx, m_idx)

                    # --- (C) Step the environment ---
                    results = trainer.env.step_lower(t_idx, s_idx, batch_sizes, n_idx, m_idx, task_deadlines,
                                                     tasks_min_accuracy)

                    # --- (C1) Accumulate frame-level failure metrics for V_cum ---
                    self._update_frame_accumulators(trainer, results['info'])

                    # --- (D) ALWAYS record metrics (Skip storage during eval) ---
                    self.store_lower_transitions(
                        trainer, results,
                        fresh_prev_states, prev_lower_mf[t_idx],
                        None, curr_lower_mf,  # curr_state is only for storage
                        t_idx, n_idx, m_idx, masks, l_log_probs, l_vals
                    )

                    # Record upper metrics using full results
                    self.store_upper_transitions(trainer, current_upper_state, None, obs_upper, results, u_acts_matrix,
                                                 None, None, (slot == max_slots - 1))

                    ep_reward += results['reward_global']
                    ep_backlog.append(results['obs']['backlog'].sum().item())
                    ep_energy += results.get('energy', results['info'].get('energy', 0.0))

                    # --- (E) Roll prev ← curr: scatter back to (num_terminals, *) ---
                    def scatter_update(base, idx, val):
                        out = base.clone()
                        out[idx] = val
                        return out

                    prev_lower_mf = scatter_update(prev_lower_mf, t_idx, curr_lower_mf)
                    prev_lower_obs = results
                else:
                    trainer.env.time_manager.tick()

                if trainer.env.time_manager.is_new_frame():
                    res_upper = trainer.env.collect_upper_metrics()
                    obs_upper = res_upper
                    current_upper_state = self.build_upper_state(trainer, res_upper)

            eval_metrics['rewards'].append(ep_reward)
            eval_metrics['backlogs'].append(np.mean(ep_backlog) if ep_backlog else 0)
            eval_metrics['energies'].append(ep_energy)

            print(f"Eval Episode {ep + 1}: Reward={ep_reward:.2f}, Avg Backlog={eval_metrics['backlogs'][-1]:.2f}")

        # Generate Report
        self.generate_report(eval_metrics)
        self.is_evaluating = False

    def generate_report(self, metrics):
        report_path = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        avg_reward = np.mean(metrics['rewards'])
        avg_backlog = np.mean(metrics['backlogs'])
        avg_energy = np.mean(metrics['energies'])

        content = f"""# Post-Training Evaluation Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary of Deterministic Execution (5 Episodes)
| Metric | Average Value |
| :--- | :--- |
| **Total Reward** | {avg_reward:.4f} |
| **Mean Backlog** | {avg_backlog:.4f} |
| **Total Energy** | {avg_energy:.4f} |

## Details per Episode
"""
        for i in range(len(metrics['rewards'])):
            content += f"- Episode {i + 1}: Reward={metrics['rewards'][i]:.2f}, Backlog={metrics['backlogs'][i]:.2f}, Energy={metrics['energies'][i]:.2f}\n"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"\n[Evaluation] Report generated: {report_path}")
        print("=" * 40)
        print(content)
        print("=" * 40)