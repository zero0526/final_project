import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import math

from matrix_source.agents.buffer.rollout_buffer import MultiAgentRolloutBuffer
from matrix_source.agents.residual_net import ResidualCritic, RefineActor, ProposalActor, MFNetwork
from matrix_source.agents.residual_visual import ResidualTracker, explained_variance

class ResidualRoutingAgent:
    def __init__(self, agent_id, node_type,
                 service_state_dim, mf_dim, proposal_dim,
                 action_dim, u_action_dim,
                 mf_hidden_sizes, mf_lr, buffer_min_size,
                 hidden_sizes=(128, 64), lr=3e-4,
                 gamma=0.99, alpha=0.2,
                 buffer_size=100_000, batch_size=128,
                 lam=0.95, clip_eps=0.2, k_epochs=5,
                 entropy_coef=0.05, exclude_zero=False,
                 num_instances=1, device=None):

        self.agent_id = agent_id
        self.node_type = node_type
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.num_instances = num_instances
        self.action_dim = action_dim
        self.u_action_dim = u_action_dim
        self.exclude_zero = exclude_zero

        self.M = service_state_dim // 2
        self.max_models = u_action_dim // self.M

        self.gamma = gamma
        self.lmbda = lam
        self.eps_clip = clip_eps
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.min_batch_size = buffer_min_size
        self.alpha = alpha

        self.initial_entropy_coef = entropy_coef
        self.entropy_coef = entropy_coef
        self.entropy_decay_rate = 0.99
        self.min_entropy_coef = 0.001

        TASK_DIM = 4
        GENERAL_TASK_DIM = 7

        # ── Networks ──
        self.mf_net = MFNetwork(
            input_dim=GENERAL_TASK_DIM + service_state_dim + mf_dim,
            output_dim=mf_dim, hidden_sizes=mf_hidden_sizes, num_instances=num_instances,
        ).to(self.device)

        self.proposal = ProposalActor(
            task_state=TASK_DIM, service_state=service_state_dim, mf_dim=mf_dim,
            action_dim=u_action_dim, hidden_sizes=hidden_sizes, num_instances=num_instances,
        ).to(self.device)

        self.refine = RefineActor(
            task_state=TASK_DIM, service_state=service_state_dim, mf_dim=mf_dim,
            proposal_dim=proposal_dim, action_dim=u_action_dim,
            hidden_sizes=hidden_sizes, num_instances=num_instances,
        ).to(self.device)

        # CRITIC KHÔNG NHẬN h_star (hist_dim)
        self.critic = ResidualCritic(
            general_task_states=GENERAL_TASK_DIM, service_states=service_state_dim,
            mf_dim=mf_dim, hidden_sizes=hidden_sizes, num_instances=num_instances,
        ).to(self.device)

        # ── Optimizers (FIX #6: LR ÷ 10) ──
        stable_lr = lr * 0.1
        self.optimizer_proposal = optim.Adam(self.proposal.parameters(), lr=stable_lr)
        self.optimizer_refine = optim.Adam(self.refine.parameters(), lr=stable_lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=stable_lr)
        self.mf_optimizer = optim.Adam(self.mf_net.parameters(), lr=mf_lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.memory = MultiAgentRolloutBuffer(
            num_agents=num_instances, node_type=node_type,
            max_size_per_agent=buffer_size, service_state_dim=service_state_dim,
            action_dim=action_dim, device=self.device,
        )

        self.learn_step_counter = 0
        self.proposal_load_var: float = 0.0
        self.equilibrium_load_var: float = 0.0
        
        # ── Diagnostics Tracker ──
        self.tracker = ResidualTracker(name=self.node_type)

    def update_coeff(self, step):
        entropy_coef = 0.001 + (0.05 - 0.001) * math.exp(-0.0307 * step)
        return entropy_coef
    # ----------------------------------------------------------
    # ① INFERENCE (ROLLING)
    # ----------------------------------------------------------
    def choose_action(self, state, prev_mf, mask=None, agent_idx=0, task_state=None, deterministic=False):
        idx_t = torch.tensor([agent_idx], device=self.device)
        if state.dim() == 1: state = state.unsqueeze(0)
        if prev_mf.dim() == 1: prev_mf = prev_mf.unsqueeze(0)
        if not isinstance(task_state, list): task_state = [task_state]
        if mask is not None and not isinstance(mask, list): mask = [mask]

        actions, log_probs, values, _ = self.choose_action_batch(
            service_states=state, prev_mfs=prev_mf, task_states=task_state,
            masks_batch=mask, agent_indices=idx_t, deterministic=deterministic,
        )
        return actions[0], log_probs[0], values[0]

    def tasks_to_general(self, task_states):
        if isinstance(task_states, (list, tuple)):
            return torch.stack([self._general_single(t) for t in task_states])
        return self._general_single(task_states)

    def _general_single(self, tasks):
        if tasks.shape[0] == 0: return torch.zeros(7, device=self.device)
        t = tasks.float()
        mean = t.mean(dim=0)
        std = t.std(dim=0, correction=0) if t.shape[0] > 1 else torch.zeros_like(mean)
        return torch.tensor([mean[0], mean[1], float(t.shape[0]), mean[2], std[2], mean[3], std[3]], 
                            dtype=torch.float32, device=self.device)

    def choose_action_batch(self, service_states, prev_mfs, task_states, masks_batch=None,
                            agent_indices=None, deterministic=False, phrase="Proposal_Free"):
        B = service_states.shape[0]
        device = self.device

        if agent_indices is None:
            agent_indices = torch.zeros(B, dtype=torch.long, device=device)
        else:
            agent_indices = agent_indices.to(device).view(-1)

        service_states = service_states.to(device).float()
        prev_mfs = prev_mfs.to(device).float()
        general_task = self.tasks_to_general(task_states)

        with torch.no_grad():
            # 1. Mean Field Prediction
            pred_mfs = self.mf_net(torch.cat([general_task, service_states, prev_mfs], dim=-1), indices=agent_indices)

            # 2. Flatten & Expand
            task_lens = torch.tensor([t.shape[0] for t in task_states], device=device)
            total_tasks = int(task_lens.sum().item())
            batch_idx = torch.repeat_interleave(torch.arange(B, device=device), task_lens)

            tasks_cat = torch.cat(task_states, dim=0).to(device).float()
            svc_exp, mf_exp, idx_exp = service_states[batch_idx], pred_mfs[batch_idx], agent_indices[batch_idx]

            # 3. Mask Handling (Robust)
            if masks_batch is not None:
                if isinstance(masks_batch, list):
                    if masks_batch[0].dim() == 1:
                        masks_exp = torch.stack(masks_batch).to(device)[batch_idx]
                    else:
                        masks_exp = torch.cat(masks_batch, dim=0).to(device)
                else:
                    masks_exp = masks_batch.to(device)
            else:
                masks_exp = None

            # 4. Proposal Forward
            prop_logits = self.proposal(tasks_cat, svc_exp, mf_exp, indices=idx_exp)
            self.last_prop_logits_mean = prop_logits.detach().float().abs().mean()

            # 5. Histogram & Overload (CHỈ làm input cho Refine)
            h_node, overload = self._compute_histogram_and_overload(
                prop_logits.detach(), masks_exp, batch_idx, B, total_tasks, service_states
            )
            self.proposal_load_var = h_node.var(dim=1).mean().item()

            # 6. Refinement Forward
            if phrase == "Proposal_Only":
                delta_logits = torch.zeros_like(prop_logits)
            else:
                delta_logits = self.refine(
                    tasks_cat, svc_exp, mf_exp,
                    prop_logits.detach(),  # ✅ Stop-gradient ngầm định ở inference
                    h_node[batch_idx], overload[batch_idx],
                    indices=idx_exp
                )

            self.equilibrium_load_var = h_node.var(dim=1).mean().item()

            # 7. Fusion (FREE DELTA: không nhân alpha)
            final_logits = prop_logits + delta_logits

            # 8. Mask & Sanitize
            if masks_exp is not None:
                final_logits = final_logits.masked_fill(masks_exp == 0, -1e9)
            if self.exclude_zero and self.u_action_dim > 1:
                final_logits[:, 0] = -1e9
            final_logits = self._sanitize_logits(final_logits)

            # 9. Action Selection
            if deterministic:
                actions_cat = final_logits.argmax(dim=-1)
                log_probs_cat = torch.zeros(total_tasks, device=device)
            else:
                dist = Categorical(logits=final_logits)
                actions_cat = dist.sample()
                log_probs_cat = dist.log_prob(actions_cat)

            # 10. Unflatten & Aggregate
            task_lens_list = task_lens.cpu().tolist()
            all_actions = list(actions_cat.split(task_lens_list))

            # Joint log-prob = sum(log_prob_i) → đúng cho PPO ratio
            sum_lp = torch.zeros(B, device=device).scatter_add_(0, batch_idx, log_probs_cat)
            all_log_probs = list(sum_lp.unbind())

            # 11. Critic (KHÔNG nhận h_node)
            all_values = self.critic(general_task, service_states, pred_mfs, indices=agent_indices)

        return all_actions, all_log_probs, all_values, h_node

    @staticmethod
    def _sanitize_logits(z):
        z = torch.where(torch.isnan(z), torch.tensor(-1e6, device=z.device), z)
        z = torch.where(torch.isinf(z), torch.tensor(-1e6, device=z.device), z)
        z = torch.where((z <= -1e5).all(dim=-1, keepdim=True), torch.zeros_like(z), z)
        return z

    def _compute_histogram_and_overload(self, prop_logits, masks_exp, batch_idx,
                                        B_sub, total_n, b_svc):
        """
        Tính histogram và overload từ Proposal logits (cho Refinement input).

        Returns:
            hist_exp: (total_n, M) - histogram per task
            over_exp: (total_n, M) - overload per task
        """
        # Mask và sanitize
        prop_for_hist = prop_logits.detach()
        if masks_exp is not None:
            prop_for_hist = prop_for_hist.masked_fill(masks_exp == 0, -1e9)

        # Tính xác suất
        prop_probs = F.softmax(self._sanitize_logits(prop_for_hist), dim=-1)
        prop_probs_M = prop_probs.view(total_n, self.M, self.max_models).sum(dim=2)

        # Aggregate về per-agent
        h_node = torch.zeros(B_sub, self.M, device=self.device)
        h_node.scatter_add_(0, batch_idx.unsqueeze(1).expand(-1, self.M), prop_probs_M)

        # Tính overload
        f_v = b_svc[:, :self.M]
        h_weighted_mean = (h_node * f_v).sum(dim=1, keepdim=True) / (f_v.sum(dim=1, keepdim=True) + 1e-8)
        overload = (h_node - h_weighted_mean) / (h_weighted_mean + 1e-8)

        # Expand về per-task
        hist_exp = h_node[batch_idx]
        over_exp = overload[batch_idx]

        return hist_exp, over_exp
    # ------------------    ----------------------------------------
    # ② STORE TRANSITION + TRAIN MF
    # ----------------------------------------------------------
    def store_transition_train_mf_batch(self, service_states, task_states, prev_mfs, curr_mfs,
                                        actions, rewards, next_service_states, dones,
                                        agent_ids, log_probs, values, masks=None):
        general_tasks = self.tasks_to_general(task_states)
        loss_mf = self.learn_mf_batch(general_tasks, service_states, prev_mfs, curr_mfs, agent_ids)
        self.memory.add_batch(service_states, task_states, prev_mfs, curr_mfs, actions, rewards, 
                              next_service_states, dones, log_probs, values, agent_ids, masks=masks)
        return loss_mf

    def learn_mf_batch(self, general_tasks, service_states, prev_mfs, ground_truth_mfs, agent_ids):
        gt, s, pm = torch.as_tensor(general_tasks, dtype=torch.float32, device=self.device), \
                     torch.as_tensor(service_states, dtype=torch.float32, device=self.device), \
                     torch.as_tensor(prev_mfs, dtype=torch.float32, device=self.device)
        gf = torch.as_tensor(ground_truth_mfs, dtype=torch.float32, device=self.device)
        pred_mf = self.mf_net(torch.cat([gt, s, pm], dim=-1), indices=agent_ids)
        loss = self.loss_fn(pred_mf, gf)
        self.mf_optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mf_net.parameters(), max_norm=5.0); self.mf_optimizer.step()
        return loss.item()

    # ----------------------------------------------------------
    # ③ PPO LEARN (2 PHASES MINIMALIST)
    # ----------------------------------------------------------
    def _unpack_task_batch(self, task_batch_cat, task_lens):
        task_states, offset = [], 0
        for n_i in task_lens:
            n_i = int(n_i.item())
            task_states.append(task_batch_cat[offset:offset + n_i]); offset += n_i
        return task_states

    def _prepare_batch_data(self, idx, service_states, prev_mfs, old_log_probs,
                            advantages, returns, agent_ids, general_tasks,
                            detached_mfs_all, task_lens, task_offsets, all_flat_idx,
                            task_batch_cat, actions_cat, all_masks):
        """
        Chuẩn bị dữ liệu cho một mini-batch.

        Returns:
            batch_data: dict chứa tất cả tensors cần thiết
        """
        B_sub = len(idx)

        b_svc = service_states[idx]
        b_old_lp = old_log_probs[idx]
        b_adv = advantages[idx]
        b_ret = returns[idx]
        b_aids = agent_ids[idx]
        b_gen = general_tasks[idx]
        b_mf = detached_mfs_all[idx]
        b_t_lens = task_lens[idx]

        # Flat indices for tasks
        segments = [all_flat_idx[task_offsets[i]:task_offsets[i] + task_lens[i]] for i in idx]
        flat_indices = torch.cat(segments)

        t_cat = task_batch_cat[flat_indices]
        act_cat = actions_cat[flat_indices]
        total_n = t_cat.shape[0]

        # Task → agent mapping
        batch_idx = torch.repeat_interleave(torch.arange(B_sub, device=self.device), b_t_lens)

        # Expand per-agent data to per-task
        svc_exp = b_svc[batch_idx]
        mf_exp = b_mf[batch_idx]
        aids_exp = b_aids[batch_idx]
        masks_exp = all_masks[idx][batch_idx] if all_masks is not None else None

        return {
            'B_sub': B_sub,
            'total_n': total_n,
            'b_svc': b_svc, 'b_old_lp': b_old_lp, 'b_adv': b_adv, 'b_ret': b_ret,
            'b_aids': b_aids, 'b_gen': b_gen, 'b_mf': b_mf, 'b_t_lens': b_t_lens,
            't_cat': t_cat, 'act_cat': act_cat,
            'batch_idx': batch_idx,
            'svc_exp': svc_exp, 'mf_exp': mf_exp, 'aids_exp': aids_exp, 'masks_exp': masks_exp
        }

    def _mask_and_sanitize(self, z, masks_exp):
        """
        Apply mask và sanitize logits.
        """
        if masks_exp is not None:
            z = z.masked_fill(masks_exp == 0, -1e9)
        if self.exclude_zero and self.u_action_dim > 1:
            z = z.clone()
            z[:, 0] = -1e9
        return self._sanitize_logits(z)

    def _compute_contribution_routing(self, prop_logits, delta_logits, act_cat,
                                      batch_idx, b_t_lens):
        """
        Tính tỷ lệ đóng góp (Contribution-Based Routing).

        Returns:
            alpha_prop: (B_sub,) - trọng số cho Proposal
            alpha_refine: (B_sub,) - trọng số cho Refinement
        """
        # Lấy logit tại action được chọn
        prop_at_action = prop_logits.gather(1, act_cat.unsqueeze(-1)).squeeze(-1)
        delta_at_action = delta_logits.gather(1, act_cat.unsqueeze(-1)).squeeze(-1)

        # Tính tỷ lệ đóng góp per-task
        prop_abs = torch.abs(prop_at_action)
        delta_abs = torch.abs(delta_at_action)
        alpha_prop_task = prop_abs / (prop_abs + delta_abs + 1e-8)

        # Aggregate về per-agent
        sum_alpha_p = torch.zeros(len(b_t_lens), device=self.device).scatter_add_(
            0, batch_idx, alpha_prop_task
        )
        alpha_prop = torch.clamp(sum_alpha_p / b_t_lens.float(), 0.1, 0.9).detach()
        alpha_refine = 1.0 - alpha_prop

        return alpha_prop, alpha_refine

    def _compute_ppo_loss_with_stop_gradient(self, prop_logits, delta_logits, act_cat,
                                             batch_idx, b_t_lens, b_old_lp, b_adv, masks_exp):
        """
        Tính PPO loss với Stop-Gradient chéo.

        Returns:
            loss_p: scalar - Proposal loss
            loss_r: scalar - Refinement loss
        """
        # -- Proposal Loss (Detach Refinement) --
        final_P = self._mask_and_sanitize(prop_logits + delta_logits.detach(), masks_exp)
        dist_P = Categorical(logits=final_P)
        sum_lp_P = torch.zeros(len(b_t_lens), device=self.device).scatter_add_(
            0, batch_idx, dist_P.log_prob(act_cat)
        )
        ratio_P = torch.exp((sum_lp_P / b_t_lens.float()) - b_old_lp)
        loss_P_agent = -torch.min(
            ratio_P * b_adv,
            torch.clamp(ratio_P, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv
        )

        # -- Refinement Loss (Detach Proposal) --
        final_R = self._mask_and_sanitize(prop_logits.detach() + delta_logits, masks_exp)
        dist_R = Categorical(logits=final_R)
        sum_lp_R = torch.zeros(len(b_t_lens), device=self.device).scatter_add_(
            0, batch_idx, dist_R.log_prob(act_cat)
        )
        ratio_R = torch.exp((sum_lp_R / b_t_lens.float()) - b_old_lp)
        loss_R_agent = -torch.min(
            ratio_R * b_adv,
            torch.clamp(ratio_R, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv
        )

        return loss_P_agent, loss_R_agent, ratio_P, ratio_R

    def _compute_final_constraints(self, prop_logits, delta_logits, batch_idx,
                                   b_t_lens, masks_exp):
        """
        Tính Final Entropy Floor và Regulation Norm penalties.

        Returns:
            entropy_penalty: scalar
            norm_penalty: scalar
            ent_joint_m: (B_sub,) - entropy per agent (cho logging)
            final_norm_m: (B_sub,) - norm per agent (cho logging)
        """
        # Joint logits
        final_joint = self._mask_and_sanitize(prop_logits + delta_logits, masks_exp)
        dist_joint = Categorical(logits=final_joint)

        # -- Entropy Floor (Target >= 0.2) --
        sum_ent = torch.zeros(len(b_t_lens), device=self.device).scatter_add_(
            0, batch_idx, dist_joint.entropy()
        )
        ent_joint_m = sum_ent / b_t_lens.float()
        # FIX #2: weight 0.03→0.15, threshold 0.2→0.5
        entropy_bonus = -0.01 * ent_joint_m.mean()
        entropy_penalty = 0.02 * torch.exp(3.0 * (0.2 - ent_joint_m)).mean()
        final_entropy_term = entropy_bonus + entropy_penalty

        # -- Regulation Norm (Target <= 3.5, NO MASK) --
        final_raw = prop_logits + delta_logits
        sum_norm = torch.zeros(len(b_t_lens), device=self.device).scatter_add_(
            0, batch_idx, final_raw.norm(dim=-1)
        )
        final_norm_m = sum_norm / b_t_lens.float()
        # FIX #3: weight 0.02→0.15, threshold 5.0→3.5
        norm_penalty = 0.15 * torch.relu(final_norm_m - 3.5).pow(2).mean()

        return final_entropy_term, norm_penalty, ent_joint_m, final_norm_m

    def _compute_exponential_anti_laziness(self, delta_logits, b_adv, batch_idx,
                                           b_t_lens, k_decay=1.0, lambda_lazy=1.0):
        """
        Tính Exponential Anti-Laziness Penalty.

        Formula: L = λ * ReLU(-A) * e^{-k|δ|}

        Returns:
            lazy_penalty: scalar
            delta_norm_agent: (B_sub,) - cho logging
        """
        # Tính |δ| per-task
        delta_norm_task = delta_logits.norm(dim=-1)

        # Aggregate về per-agent
        sum_delta_norm = torch.zeros(len(b_t_lens), device=self.device).scatter_add_(
            0, batch_idx, delta_norm_task
        )
        delta_norm_agent = sum_delta_norm / b_t_lens.float()

        # Selective trigger: chỉ phạt khi advantage âm
        selective_mask = torch.relu(-b_adv)

        # Exponential decay
        exp_decay = torch.exp(-k_decay * delta_norm_agent)

        # Penalty
        lazy_penalty = lambda_lazy * (selective_mask * exp_decay).mean()

        return lazy_penalty, delta_norm_agent

    def learn(self, step: int, agents_ids=None, zeta=1.0):
        from matrix_source.trainers.ppo_stategy import compute_gae
        if agents_ids is not None:
            agents_ids = agents_ids.to(self.device).view(-1)

        data = self.memory.get_all_ready(min_size=self.min_batch_size, agent_ids_pool=agents_ids)
        if data is None:
            return None

        # ═══ 1. UNPACK & PREPARE DATA ═══
        (service_states, task_batch_cat, task_lens, actions_cat, action_lens,
         prev_mfs, curr_mfs, rewards, next_service_states, dones,
         old_log_probs, old_values, masks, agent_ids) = data

        # Convert to device
        service_states = service_states.to(self.device).float()
        prev_mfs = prev_mfs.to(self.device).float()
        agent_ids = agent_ids.to(self.device).long()
        task_lens = task_lens.to(self.device).long()
        task_batch_cat = task_batch_cat.to(self.device).float()
        actions_cat = actions_cat.to(self.device).long()
        old_log_probs = old_log_probs.to(self.device).squeeze(-1)
        old_values = old_values.to(self.device).squeeze(-1)
        rewards = rewards.to(self.device).squeeze(-1)
        dones = dones.to(self.device).squeeze(-1)

        dataset_size = service_states.shape[0]
        total_tasks_flat = task_batch_cat.shape[0]

        task_states_list = self._unpack_task_batch(task_batch_cat, task_lens)
        general_tasks = self.tasks_to_general(task_states_list).to(self.device)

        # ═══ 2. PRE-COMPUTE (No Grad) ═══
        with torch.no_grad():
            mf_in = torch.cat([general_tasks, next_service_states, curr_mfs.to(self.device)], dim=-1)
            next_mf = self.mf_net(mf_in, indices=agent_ids)
            next_val = self.critic(general_tasks, next_service_states, next_mf, indices=agent_ids)

            advantages = compute_gae(rewards, next_val, old_values, dones, agent_ids, self.gamma, self.lmbda)
            returns = advantages + old_values
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            mf_in_all = torch.cat([general_tasks, service_states, prev_mfs], dim=-1)
            detached_mfs_all = self.mf_net(mf_in_all, indices=agent_ids).detach()

        # Pre-process masks
        if masks is not None and any(m is not None for m in masks):
            first_valid = next(m for m in masks if m is not None)
            if first_valid.dim() == 1:
                clean_masks = [m if m is not None else torch.zeros_like(first_valid) for m in masks]
                all_masks = torch.stack(clean_masks).to(self.device)
            else:
                all_masks = torch.cat([m for m in masks if m is not None], dim=0).to(self.device)
        else:
            all_masks = None

        # Pre-compute indices
        task_offsets = torch.zeros(dataset_size, dtype=torch.long, device=self.device)
        task_offsets[1:] = task_lens.cumsum(0)[:-1]
        all_flat_idx = torch.arange(total_tasks_flat, device=self.device)

        # ═══ 3. PPO TRAINING LOOP ═══
        epoch_metrics = {'v': 0.0, 'p': 0.0, 'r': 0.0, 'ent_f': 0.0, 'norm_f': 0.0,
                         'alpha_p': 0.0, 'alpha_r': 0.0, 'lazy': 0.0}
        total_batches = 0
        # FIX #7: WARMUP - skip Refine for first 200 learn steps
        is_warmup = step < 1

        for _ in range(self.k_epochs):
            perm = torch.randperm(dataset_size, device=self.device)

            for start in range(0, dataset_size, self.batch_size):
                idx = perm[start:start + self.batch_size]

                # ── Chuẩn bị batch data ──
                bd = self._prepare_batch_data(
                    idx, service_states, prev_mfs, old_log_probs, advantages, returns,
                    agent_ids, general_tasks, detached_mfs_all, task_lens, task_offsets,
                    all_flat_idx, task_batch_cat, actions_cat, all_masks
                )

                # ── Forward Pass: Proposal ──
                prop_logits = self.proposal(bd['t_cat'], bd['svc_exp'], bd['mf_exp'], indices=bd['aids_exp'])

                # ── Forward Pass: Refinement (WARMUP: zero delta) ──
                if is_warmup:
                    delta_logits = torch.zeros_like(prop_logits)
                else:
                    hist_exp, over_exp = self._compute_histogram_and_overload(
                        prop_logits, bd['masks_exp'], bd['batch_idx'],
                        bd['B_sub'], bd['total_n'], bd['b_svc']
                    )
                    delta_logits = self.refine(
                        bd['t_cat'], bd['svc_exp'], bd['mf_exp'],
                        prop_logits.detach(), hist_exp, over_exp, indices=bd['aids_exp']
                    )

                # ── Contribution Routing (WARMUP: fixed 0.9/0.1) ──
                if is_warmup:
                    B_sub = bd['B_sub']
                    alpha_prop = torch.ones(B_sub, device=self.device) * 0.9
                    alpha_refine = torch.ones(B_sub, device=self.device) * 0.1
                else:
                    alpha_prop, alpha_refine = self._compute_contribution_routing(
                        prop_logits, delta_logits, bd['act_cat'], bd['batch_idx'], bd['b_t_lens']
                    )

                # ── PPO Loss ──
                loss_P_agent, loss_R_agent , ratio_P, ratio_R= self._compute_ppo_loss_with_stop_gradient(
                    prop_logits, delta_logits, bd['act_cat'], bd['batch_idx'],
                    bd['b_t_lens'], bd['b_old_lp'], bd['b_adv'], bd['masks_exp']
                )
                loss_p = (alpha_prop * loss_P_agent).mean()
                loss_r = (alpha_refine * loss_R_agent).mean()

                # ── Final Constraints ──
                final_entropy_term, norm_penalty, ent_joint_m, final_norm_m = self._compute_final_constraints(
                    prop_logits, delta_logits, bd['batch_idx'], bd['b_t_lens'], bd['masks_exp']
                )

                # ── Anti-Laziness (skip during warmup) ──
                if is_warmup:
                    lazy_penalty = torch.tensor(0.0, device=self.device)
                    delta_norm_agent = torch.zeros(bd['B_sub'], device=self.device)
                else:
                    lazy_penalty, delta_norm_agent = self._compute_exponential_anti_laziness(
                        delta_logits, bd['b_adv'], bd['batch_idx'], bd['b_t_lens'],
                        k_decay=2.0, lambda_lazy=0.05
                    )

                # ── Total Loss ──
                total_penalty = final_entropy_term + norm_penalty + lazy_penalty
                if is_warmup:
                    loss_actor = loss_p + final_entropy_term + norm_penalty
                else:
                    loss_actor = loss_p + loss_r + total_penalty

                # ── Actor Update (FIX #4: clip 0.5→0.1) ──
                self.optimizer_proposal.zero_grad(set_to_none=True)
                self.optimizer_refine.zero_grad(set_to_none=True)
                loss_actor.backward()
                torch.nn.utils.clip_grad_norm_(self.proposal.parameters(), 0.1)
                torch.nn.utils.clip_grad_norm_(self.refine.parameters(), 0.1)
                self.optimizer_proposal.step()
                self.optimizer_refine.step()

                # ── Critic Update (FIX #5: clip 0.5→0.1) ──
                c_vals = self.critic(bd['b_gen'], bd['b_svc'], bd['b_mf'], indices=bd['b_aids'])
                c_loss = F.mse_loss(c_vals, bd['b_ret'])
                self.optimizer_critic.zero_grad(set_to_none=True)
                c_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.1)
                self.optimizer_critic.step()

                # ── Save diagnostic state (FIX #7 monitoring) ──
                self._last_final_norm_mean = final_norm_m.mean().item()
                self._last_entropy_mean = ent_joint_m.mean().item()
                self._last_nan_count = (prop_logits.isnan().sum() + delta_logits.isnan().sum()).item()

                # ── Metrics ──
                epoch_metrics['v'] += c_loss.item()
                epoch_metrics['p'] += loss_p.item()
                epoch_metrics['r'] += loss_r.item()
                epoch_metrics['ent_f'] += ent_joint_m.mean().item()
                epoch_metrics['norm_f'] += final_norm_m.mean().item()
                epoch_metrics['alpha_p'] += alpha_prop.mean().item()
                epoch_metrics['alpha_r'] += alpha_refine.mean().item()
                epoch_metrics['lazy'] += lazy_penalty.item()
                total_batches += 1

                # ── LOG TO RESIDUAL TRACKER ──
                # Calculate gradient norms (before clipping again or after, here before step)
                grad_norm_p = torch.nn.utils.clip_grad_norm_(self.proposal.parameters(), 1e6).item()
                grad_norm_r = torch.nn.utils.clip_grad_norm_(self.refine.parameters(), 1e6).item()
                
                # Check NaNs
                nan_p = sum(torch.isnan(p.grad).sum().item() for p in self.proposal.parameters() if p.grad is not None)
                nan_r = sum(torch.isnan(p.grad).sum().item() for p in self.refine.parameters() if p.grad is not None)

                # Explained Variance
                ev = explained_variance(c_vals, bd['b_ret']).item()

                self.tracker.log(
                    reward=rewards.mean().item(),
                    ppo_ratio=ratio_P.mean().item(), # Proxy for PPO behavior
                    value_loss=c_loss.item(),
                    expl_var=ev,
                    grad_prop=grad_norm_p,
                    grad_refine=grad_norm_r,
                    grad_ratio=grad_norm_r / (grad_norm_p + 1e-8),
                    nan_count=nan_p + nan_r,
                    grad_sim=0.0, # Networks are disjoint
                    prop_norm=prop_logits.norm(dim=-1).mean().item(),
                    delta_norm=delta_logits.norm(dim=-1).mean().item(),
                    final_norm=final_norm_m.mean().item(),
                    final_max=(prop_logits + delta_logits).norm(dim=-1).max().item(),
                    contrib_ratio=(delta_logits.norm(dim=-1) / (prop_logits.norm(dim=-1) + delta_logits.norm(dim=-1) + 1e-8)).mean().item(),
                    alpha_prop=alpha_prop.mean().item(),
                    alpha_std=alpha_prop.std().item() if alpha_prop.numel() > 1 else 0.0,
                    pct_prop_dom=(alpha_prop > 0.8).float().mean().item() * 100,
                    pct_ref_dom=(alpha_prop < 0.2).float().mean().item() * 100,
                    entropy=ent_joint_m.mean().item(),
                    pct_low_ent=(ent_joint_m < 0.2).float().mean().item() * 100,
                    entropy_pen=final_entropy_term.item(),
                    norm_pen=norm_penalty.item(),
                    lazy_pen=lazy_penalty.item(),
                    pct_bad=(bd['b_adv'] < 0).float().mean().item() * 100,
                    pct_zero_delta=(delta_norm_agent < 0.01).float().mean().item() * 100
                )

        self.learn_step_counter += 1

        # Plot dashboard every step for immediate verification (increments on training calls, not episodes)
        if step % 3 == 0:
            self.tracker.plot_dashboard()

        # ── Logging ──
        log_freq = 10 if self.node_type == "Edge_Group" else 100
        if self.learn_step_counter % log_freq == 0 and total_batches > 0:
            n = total_batches
            warmup_tag = "[WARMUP]" if is_warmup else ""
            print(
                f"[{self.node_type}]{warmup_tag} Step {self.learn_step_counter:5d} | "
                f"V: {epoch_metrics['v'] / n:.4f} | P: {epoch_metrics['p'] / n:.4f} | R: {epoch_metrics['r'] / n:.4f} | "
                f"Ent: {epoch_metrics['ent_f'] / n:.3f} | Norm: {epoch_metrics['norm_f'] / n:.2f} | "
                f"α_P: {epoch_metrics['alpha_p'] / n:.2f} | α_R: {epoch_metrics['alpha_r'] / n:.2f} | "
                f"Lazy: {epoch_metrics['lazy'] / n:.4f}"
            )

        self.memory.clear()

        # ── EARLY WARNING CHECK (every 10 learn steps) ──
        if self.learn_step_counter % 10 == 0:
            warnings_list = []
            if hasattr(self, '_last_final_norm_mean') and self._last_final_norm_mean > 6.0:
                warnings_list.append(f"⚠️  NORM EXPLOSION: ||Final|| = {self._last_final_norm_mean:.2f}")
            if hasattr(self, '_last_entropy_mean') and self._last_entropy_mean < 0.5:
                warnings_list.append(f"⚠️  ENTROPY COLLAPSE: H = {self._last_entropy_mean:.3f}")
            if hasattr(self, '_last_nan_count') and self._last_nan_count > 0:
                warnings_list.append(f"🚨 GRADIENT NaN: count = {self._last_nan_count}")
            if warnings_list:
                print(f"\n{'!'*70}")
                print(f"[{self.node_type}] STEP {step} - WARNINGS:")
                for w in warnings_list:
                    print(f"  {w}")
                print(f"{'!'*70}\n")
                if (getattr(self, '_last_final_norm_mean', 0.0) > 8.0 or
                        getattr(self, '_last_nan_count', 0) > 0):
                    print("🔧 Auto-reducing LR by 2x...")
                    self.reduce_learning_rate(factor=0.5)

        return epoch_metrics['v'] / total_batches if total_batches > 0 else 0.0

    def reduce_learning_rate(self, factor=0.5):
        """Giảm learning rate của tất cả optimizers (FIX #6)"""
        for optimizer in [self.optimizer_proposal, self.optimizer_refine,
                          self.optimizer_critic]:
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] = old_lr * factor
                print(f"  [{self.node_type}] LR giảm: {old_lr:.6f} → {param_group['lr']:.6f}")

    # def learn(self, phrase: str, step:int, agents_ids=None):
    #     from matrix_source.trainers.ppo_stategy import compute_gae
    #     if agents_ids is not None: agents_ids = agents_ids.to(self.device).view(-1)
    #     data = self.memory.get_all_ready(min_size=self.min_batch_size, agent_ids_pool=agents_ids)
    #     if data is None: return None
    #
    #     # ENTROPY CHO 2 PHA
    #     current_ent_coef = self.initial_entropy_coef if phrase == "Proposal_Free" else \
    #         self.update_coeff(step)
    #     if phrase == "Proposal_Only": self.entropy_coef = current_ent_coef
    #
    #     (service_states, task_batch_cat, task_lens, actions_cat, action_lens,
    #      prev_mfs, curr_mfs, rewards, next_service_states, dones,
    #      old_log_probs, old_values, masks, agent_ids) = data
    #
    #     service_states, prev_mfs, agent_ids = service_states.to(self.device).float(), prev_mfs.to(
    #         self.device).float(), agent_ids.to(self.device).long()
    #     task_lens = task_lens.to(self.device).long()
    #     task_batch_cat = task_batch_cat.to(self.device).float()
    #     # FIX Ở ĐÂY: Đổi tên biến cục bộ thành b_actions để không đè lên biến actions_cat gốc
    #     b_actions_cat = actions_cat.to(self.device).long()
    #
    #     old_log_probs, old_values = old_log_probs.to(self.device).squeeze(-1), old_values.to(self.device).squeeze(-1)
    #     rewards, dones = rewards.to(self.device).squeeze(-1), dones.to(self.device).squeeze(-1)
    #
    #     dataset_size, total_tasks_flat = service_states.shape[0], task_batch_cat.shape[0]
    #     general_tasks = self.tasks_to_general(self._unpack_task_batch(task_batch_cat, task_lens)).to(self.device)
    #
    #     with torch.no_grad():
    #         # CRITIC BOOTSTRAP KHÔNG CẦN h_node
    #         next_mf = self.mf_net(torch.cat([general_tasks, next_service_states, curr_mfs.to(self.device)], dim=-1),
    #                               indices=agent_ids)
    #         next_val = self.critic(general_tasks, next_service_states, next_mf, indices=agent_ids)
    #
    #         advantages = compute_gae(rewards, next_val, old_values, dones, agent_ids, self.gamma, self.lmbda)
    #         returns = advantages + old_values
    #         if advantages.numel() > 1: advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    #         detached_mfs_all = self.mf_net(torch.cat([general_tasks, service_states, prev_mfs], dim=-1),
    #                                        indices=agent_ids).detach()
    #
    #     if masks is not None and any(m is not None for m in masks):
    #         first_valid = next(m for m in masks if m is not None)
    #         all_masks = torch.stack([m if m is not None else torch.zeros_like(first_valid) for m in masks], dim=0).to(
    #             self.device) if first_valid.dim() == 1 else torch.cat([m for m in masks if m is not None], dim=0).to(
    #             self.device)
    #     else:
    #         all_masks = None
    #
    #     task_offsets = torch.zeros(dataset_size, dtype=torch.long, device=self.device)
    #     task_offsets[1:] = task_lens.cumsum(0)[:-1]
    #     all_flat_idx = torch.arange(total_tasks_flat, device=self.device)
    #
    #     epoch_metrics = {'v': 0.0, 'p': 0.0, 'r': 0.0, 'ent_f': 0.0, 'norm_f': 0.0, 'alpha_p': 0.0, 'alpha_r': 0.0}
    #     total_batches = 0
    #
    #     for _ in range(self.k_epochs):
    #         perm = torch.randperm(dataset_size, device=self.device)
    #         for start in range(0, dataset_size, self.batch_size):
    #             idx = perm[start:start + self.batch_size]
    #             B_sub = len(idx)
    #             b_svc, b_old_lp, b_adv, b_ret = service_states[idx], old_log_probs[idx], advantages[idx], returns[idx]
    #             b_aids, b_gen, b_mf, b_t_lens = agent_ids[idx], general_tasks[idx], detached_mfs_all[idx], task_lens[
    #                 idx]
    #
    #             flat_indices = torch.cat([all_flat_idx[task_offsets[i]:task_offsets[i] + task_lens[i]] for i in idx])
    #             t_cat = task_batch_cat[flat_indices]
    #             # FIX Ở ĐÂY: Dùng b_actions_cat thay vì actions_cat
    #             act_cat = b_actions_cat[flat_indices]
    #             total_n = flat_indices.shape[0]
    #
    #             batch_idx = torch.repeat_interleave(torch.arange(B_sub, device=self.device), b_t_lens)
    #             svc_exp, mf_exp, aids_exp = b_svc[batch_idx], b_mf[batch_idx], b_aids[batch_idx]
    #             masks_exp = all_masks[idx][batch_idx] if all_masks is not None else None
    #
    #             # ══════════════════════════════════════════
    #             # 1. FORWARD PASS
    #             # ══════════════════════════════════════════
    #             prop_logits = self.proposal(t_cat, svc_exp, mf_exp, indices=aids_exp)
    #             h_node, overload = self._compute_hist_and_overload(prop_logits.detach(), masks_exp, b_svc, batch_idx,
    #                                                                B_sub, total_n)
    #
    #             if phrase == "Proposal_Only":
    #                 delta_logits = torch.zeros_like(prop_logits)
    #             else:
    #                 delta_logits = self.refine(t_cat, svc_exp, mf_exp, prop_logits.detach(), h_node[batch_idx],
    #                                            overload[batch_idx], indices=aids_exp)
    #
    #             # ══════════════════════════════════════════
    #             # 2. TÍNH LOSS TÙY THEO PHASE
    #             # ══════════════════════════════════════════
    #             loss_proposal, loss_refine = 0.0, 0.0
    #
    #             def apply_mask_and_sanitize(z):
    #                 if masks_exp is not None: z = z.masked_fill(masks_exp == 0, -1e9)
    #                 if self.exclude_zero and self.u_action_dim > 1: z[:, 0] = -1e9
    #                 return self._sanitize_logits(z)
    #
    #             if phrase == "Proposal_Only":
    #                 final_logits = apply_mask_and_sanitize(prop_logits)
    #                 dist = Categorical(logits=final_logits)
    #                 sum_lp = torch.zeros(B_sub, device=self.device).scatter_add_(0, batch_idx, dist.log_prob(act_cat))
    #                 ratio = torch.exp(sum_lp - b_old_lp)
    #                 loss_proposal = -torch.min(ratio * b_adv,
    #                                            torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv).mean() \
    #                                 - current_ent_coef * dist.entropy().mean()
    #
    #             elif phrase == "Proposal_Free":
    #                 # Nhánh Proposal (Stop-gradient đối với Refine)
    #                 z_P = apply_mask_and_sanitize(prop_logits + self.alpha * delta_logits.detach())
    #                 dist_P = Categorical(logits=z_P)
    #                 sum_lp_P = torch.zeros(B_sub, device=self.device).scatter_add_(0, batch_idx,
    #                                                                                dist_P.log_prob(act_cat))
    #                 ratio_P = torch.exp(sum_lp_P - b_old_lp)
    #                 loss_proposal = -torch.min(ratio_P * b_adv, torch.clamp(ratio_P, 1 - self.eps_clip,
    #                                                                         1 + self.eps_clip) * b_adv).mean() \
    #                                 # - current_ent_coef * dist_P.entropy().mean()
    #
    #                 # Nhánh Refine (Stop-gradient đối với Proposal)
    #                 z_R = apply_mask_and_sanitize(prop_logits.detach() + self.alpha * delta_logits)
    #                 dist_R = Categorical(logits=z_R)
    #                 sum_lp_R = torch.zeros(B_sub, device=self.device).scatter_add_(0, batch_idx,
    #                                                                                dist_R.log_prob(act_cat))
    #                 ratio_R = torch.exp(sum_lp_R - b_old_lp)
    #                 loss_refine = -torch.min(ratio_R * b_adv,
    #                                          torch.clamp(ratio_R, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv).mean()
    #
    #             # ══════════════════════════════════════════
    #             # 3. CRITIC UPDATE (KHÔNG NHẬN h_node)
    #             # ══════════════════════════════════════════
    #             c_vals = self.critic(b_gen, b_svc, b_mf, indices=b_aids)
    #             c_loss = F.mse_loss(c_vals, b_ret)
    #
    #             # ══════════════════════════════════════════
    #             # 4. BACKPROPAGATION
    #             # ══════════════════════════════════════════
    #             if isinstance(loss_proposal, torch.Tensor):
    #                 self.optimizer_proposal.zero_grad(set_to_none=True)
    #                 loss_proposal.backward()
    #                 torch.nn.utils.clip_grad_norm_(self.proposal.parameters(), 0.5)
    #                 self.optimizer_proposal.step()
    #
    #             if isinstance(loss_refine, torch.Tensor):
    #                 self.optimizer_refine.zero_grad(set_to_none=True)
    #                 loss_refine.backward()
    #                 torch.nn.utils.clip_grad_norm_(self.refine.parameters(), 0.5)
    #                 self.optimizer_refine.step()
    #
    #             self.optimizer_critic.zero_grad(set_to_none=True)
    #             c_loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
    #             self.optimizer_critic.step()
    #
    #             epoch_metrics['v'] += c_loss.item()
    #             epoch_metrics['p'] += loss_proposal.item() if isinstance(loss_proposal, torch.Tensor) else 0.0
    #             epoch_metrics['r'] += loss_refine.item() if isinstance(loss_refine, torch.Tensor) else 0.0
    #             total_batches += 1
    #
    #     self.learn_step_counter += 1
    #     if self.learn_step_counter % 1 == 0 and total_batches > 0:
    #         n = total_batches
    #         print(
    #             f"[{self.node_type}][{phrase}] Step {self.learn_step_counter:5d} | V: {epoch_metrics['v'] / n:.5f} | P: {epoch_metrics['p'] / n:.5f} | R: {epoch_metrics['r'] / n:.5f}")
    #
    #     self.memory.clear()
    #     return epoch_metrics['v'] / total_batches if total_batches > 0 else 0.0



    # ----------------------------------------------------------
    # ④ CHECKPOINT
    # ----------------------------------------------------------
    def save(self, path: str):
        torch.save({
            'proposal': self.proposal.state_dict(), 'refine': self.refine.state_dict(),
            'critic': self.critic.state_dict(), 'mf_net': self.mf_net.state_dict(),
            'proposal_opt': self.optimizer_proposal.state_dict(), 'refine_opt': self.optimizer_refine.state_dict(),
            'critic_opt': self.optimizer_critic.state_dict(), 'mf_opt': self.mf_optimizer.state_dict(),
            'learn_step': self.learn_step_counter, 'entropy_coef': self.entropy_coef,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        for k in ['proposal', 'refine', 'critic', 'mf_net']: getattr(self, k).load_state_dict(ckpt[k])
        for k in ['proposal_opt', 'refine_opt', 'critic_opt', 'mf_opt']: getattr(self, f"optimizer_{k.replace('_opt','')}").load_state_dict(ckpt[k])
        self.learn_step_counter = ckpt.get('learn_step', 0)
        self.entropy_coef = ckpt.get('entropy_coef', self.initial_entropy_coef)

def compute_exponential_anti_laziness(delta_logits, advantages,
                                      batch_idx, b_t_lens, device, k=2.0, lambda_lazy=0.05):
    """
    Exponential Anti-Laziness Penalty.

    Công thức: L = λ * ReLU(-A) * e^{-k|δ|}

    Args:
        delta_logits: (total_n, action_dim) - logits từ Refinement
        advantages: (B_sub,) - advantage per agent
        batch_idx: (total_n,) - mapping task → agent
        b_t_lens: (B_sub,) - số task mỗi agent
        k: decay rate (mặc định 2.0)
        lambda_lazy: trọng số penalty (mặc định 0.05)

    Returns:
        penalty: scalar
        metrics: dict
    """
    delta_norm_task = delta_logits.norm(dim=-1)  # (total_n,)

    # ═══════════════════════════════════════════════
    # BƯỚC 2: Aggregate về per-agent
    # ═══════════════════════════════════════════════

    sum_delta_norm = torch.zeros(len(b_t_lens), device=device).scatter_add_(
        0, batch_idx, delta_norm_task
    )
    delta_norm_agent = sum_delta_norm / b_t_lens.float()  # (B_sub,)

    # ═══════════════════════════════════════════════
    # BƯỚC 3: Tính penalty
    # ═══════════════════════════════════════════════

    # Selective trigger: chỉ phạt khi advantage âm
    selective_mask = torch.relu(-advantages)  # (B_sub,)

    # Exponential decay: e^{-k|δ|}
    exp_decay = torch.exp(-k * delta_norm_agent)  # (B_sub,)

    # Penalty per-agent
    lazy_penalty_per_agent = selective_mask * exp_decay  # (B_sub,)

    # Trung bình
    lazy_penalty = lambda_lazy * lazy_penalty_per_agent.mean()

    # ═══════════════════════════════════════════════
    # BƯỚC 4: Metrics
    # ═══════════════════════════════════════════════

    metrics = {
        'lazy_penalty': lazy_penalty.item(),
        'avg_delta_norm': delta_norm_agent.mean().item(),
        'pct_bad_agents': (advantages < 0).float().mean().item() * 100,
        'avg_exp_decay': exp_decay.mean().item(),
    }

    return lazy_penalty, metrics