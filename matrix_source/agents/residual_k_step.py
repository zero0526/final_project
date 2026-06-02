"""
Residual Routing PPO Agent  (K-step Equilibrium Refinement)
============================================================
Multi-agent PPO với kiến trúc:
  Proposal → K-step Histogram-aware Refinement → Equilibrium

Architecture:
  - MFNetwork     : Predict mean field (shared, MultiInstance)
  - ProposalActor : Base routing decision P(s)  (shared, MultiInstance)
  - RefineActor   : Residual correction δ(a^k, h^k)  (shared, MultiInstance, init≈0)
  - Critic        : State value V(s)  (shared, MultiInstance)

K-step update rule:
  a^(k+1) = P(s) + α · R(a^(k), h(a^(k)))

Forward:  K iterations searching equilibrium
Backward: gradient only through final step  (DEQ-style)

Interface tương thích hoàn toàn với PPOAgent:
  choose_action_batch(states, mfs, masks_batch, agent_indices, deterministic, zeta)
  store_transition_train_mf_batch(...)
  learn(agents_ids, zeta)
  save(path) / load(path)

Tracked diagnostics: last_residual, proposal_load_var, equilibrium_load_var
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from matrix_source.agents.buffer.rollout_buffer import MultiAgentRolloutBuffer
from matrix_source.agents.residual_net import RefineActor,ProposalActor, MFNetwork, ResidualCritic

def compute_overload(h, weights):
    h_weighted_mean = (h * weights).sum() / (weights.sum() + 1e-8)
    overload = (h - h_weighted_mean) / (h_weighted_mean + 1e-8)
    return overload


class ResidualRoutingAgent:
    def __init__(self, agent_id, node_type,
                 service_state_dim, mf_dim, proposal_dim,
                 action_dim, u_action_dim,
                 mf_hidden_sizes, mf_lr, buffer_min_size,
                 hidden_sizes=(128, 64), lr=3e-4,
                 gamma=0.99, alpha=0.1,
                 buffer_size=100_000, batch_size=128,
                 lam=0.95, clip_eps=0.2, k_epochs=5,
                 entropy_coef=0.05, exclude_zero=False,
                 num_instances=1, device=None):

        self.agent_id = agent_id
        self.node_type = node_type

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.num_instances = num_instances
        self.action_dim = action_dim
        self.u_action_dim = u_action_dim
        self.exclude_zero = exclude_zero

        # ═══ THÊM: M và max_models ═══
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
            output_dim=mf_dim,
            hidden_sizes=mf_hidden_sizes,
            num_instances=num_instances,
        ).to(self.device)

        self.proposal = ProposalActor(
            task_state=TASK_DIM,
            service_state=service_state_dim,
            mf_dim=mf_dim,
            action_dim=u_action_dim,
            hidden_sizes=hidden_sizes,
            num_instances=num_instances,
        ).to(self.device)

        self.refine = RefineActor(
            task_state=TASK_DIM,
            service_state=service_state_dim,
            mf_dim=mf_dim,
            proposal_dim=proposal_dim,
            action_dim=u_action_dim,
            hidden_sizes=hidden_sizes,
            num_instances=num_instances,
        ).to(self.device)

        self.critic = ResidualCritic(
            general_task_states=GENERAL_TASK_DIM,
            service_states=service_state_dim,
            mf_dim=mf_dim,
            histo_dim=self.M,
            hidden_sizes=hidden_sizes,
            num_instances=num_instances,
        ).to(self.device)

        # ── Optimizers ──
        self.optimizer_proposal = optim.Adam(self.proposal.parameters(), lr=lr)
        self.optimizer_refine = optim.Adam(self.refine.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.mf_optimizer = optim.Adam(self.mf_net.parameters(), lr=mf_lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.memory = MultiAgentRolloutBuffer(
            num_agents=num_instances,
            node_type=node_type,
            max_size_per_agent=buffer_size,
            service_state_dim=service_state_dim,
            action_dim=action_dim,
            device=self.device,
        )

        self.learn_step_counter = 0

        # ── K-step diagnostics ──
        self.last_residual: float = 0.0
        self.proposal_load_var: float = 0.0
        self.equilibrium_load_var: float = 0.0
        self.inference_counter: int = 0

        # Traces for visualization (captured during inference)
        self.residual_trace: list[float] = []
        self.load_var_trace: list[float] = []
        self.hist_change_trace: list[float] = []

    def choose_action(self, state, prev_mf, mask=None, agent_idx=0,
                      task_state=None, deterministic=False, zeta=1.0, temp=1.0):
        """
        Args:
            state:      (2*M,) service state
            prev_mf:    (mf_dim,) previous mean field
            mask:       (u_action_dim,) hoặc None
            task_state: (N, 4) task tensors cho agent này
        Returns:
            actions:   (N,) LongTensor
            log_probs: (N,) FloatTensor
            value:     scalar
        """
        idx_t = torch.tensor([agent_idx], device=self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)
        if prev_mf.dim() == 1:
            prev_mf = prev_mf.unsqueeze(0)
        if task_state is None:
            raise ValueError("task_state required")

        if not isinstance(task_state, list):
            task_state = [task_state]
        if mask is not None and not isinstance(mask, list):
            mask = [mask]

        actions, log_probs, values = self.choose_action_batch(
            service_states=state,
            prev_mfs=prev_mf,
            task_states=task_state,
            masks_batch=mask,
            agent_indices=idx_t,
            deterministic=deterministic,
            zeta=zeta,
            temp=temp,
        )

        return actions[0], log_probs[0], values[0]

    def tasks_to_general(self, task_states):
        """
        Args:
            task_states: List[(N_i, 4)] hoặc (N_i, 4) Tensor
        Returns:
            Nếu list: (B, 7) — mỗi hàng là summary cho 1 agent
            Nếu tensor: (7,) — summary cho 1 group
        """
        # Nếu là list (batch)
        if isinstance(task_states, (list, tuple)):
            results = []
            for t in task_states:
                results.append(self._general_single(t))
            return torch.stack(results)  # (B, 7)

        # Nếu là tensor (single group)
        return self._general_single(task_states)  # (7,)

    def _general_single(self, tasks):
        """tasks: (N_i, 4) → (7,)"""
        if tasks.shape[0] == 0:
            return torch.zeros(7, device=self.device)

        t = tasks.float()
        mean = t.mean(dim=0)
        std = t.std(dim=0, correction=0) if t.shape[0] > 1 else torch.zeros_like(mean)

        return torch.tensor([
            mean[0],  # mean omega
            mean[1],  # mean data_size
            float(t.shape[0]),  # num_tasks
            mean[2],  # mean deadline
            std[2],  # std deadline
            mean[3],  # mean acc
            std[3],  # std acc
        ], dtype=torch.float32, device=self.device)

    def choose_action_batch(self, service_states, prev_mfs, task_states,
                            masks_batch=None, agent_indices=None,
                            deterministic=False, zeta=1.0, temp=1.0):
        B = service_states.shape[0]
        device = self.device

        if agent_indices is None:
            agent_indices = torch.zeros(B, dtype=torch.long, device=device)
        else:
            agent_indices = agent_indices.to(device).view(-1)

        service_states = service_states.to(device).float()
        prev_mfs = prev_mfs.to(device).float()

        # General task: (B, 7)
        general_task = self.tasks_to_general(task_states)

        with torch.no_grad():
            # ═══ 1. MF (1 pass cho toàn batch) ═══
            mf_input = torch.cat([general_task, service_states, prev_mfs], dim=-1)
            pred_mfs = self.mf_net(mf_input, indices=agent_indices)

            # ═══ 2. TẠO FLATTEN BATCH ═══
            task_lens = torch.tensor([t.shape[0] for t in task_states], device=device)
            total_tasks = int(task_lens.sum().item())

            # Map từng task về đúng index của agent: [0,0, 1,1,1, 2...]
            batch_idx = torch.repeat_interleave(torch.arange(B, device=device), task_lens)

            # Đưa toàn bộ tasks thành 1 tensor 2D duy nhất
            tasks_cat = torch.cat(task_states, dim=0).to(device).float()
            svc_exp = service_states[batch_idx]
            mf_exp = pred_mfs[batch_idx]
            idx_exp = agent_indices[batch_idx]

            if masks_batch is not None:
                # Xử lý mask list
                if masks_batch[0].dim() == 1:
                    masks_exp = torch.stack(masks_batch).to(device)[batch_idx]
                else:
                    masks_exp = torch.cat(masks_batch, dim=0).to(device)
            else:
                masks_exp = None

            # ═══ 3. PROPOSAL (1 pass cho TẤT CẢ tasks) ═══
            # FIX: Apply temperature scaling (1.5 -> 1.0) to encourage global exploration
            prop_logits = self.proposal(tasks_cat, svc_exp, mf_exp, indices=idx_exp) / temp

            # ═══ 4. K-STEP EQUILIBRIUM REFINEMENT ═══
            self.residual_trace = []
            self.load_var_trace = []
            self.hist_change_trace = []

            # Track proposal load variance (before refinement)
            h_prop, _ = self._compute_hist_and_overload(
                prop_logits, masks_exp, service_states, batch_idx, B, total_tasks,
            )
            self.proposal_load_var = h_prop.var(dim=1).mean().item()

            # All K steps — no_grad at inference
            current_logits = prop_logits
            h_node = h_prop  # initialise for the critic call below
            prev_h = h_prop

            for k in range(K_REFINE_STEPS):
                h_node, overload = self._compute_hist_and_overload(
                    current_logits, masks_exp, service_states, batch_idx, B, total_tasks,
                )

                # Trace: Load Variance at each step
                self.load_var_trace.append(h_node.var(dim=1).mean().item())
                # Trace: Histogram change from previous step
                if k > 0:
                    self.hist_change_trace.append((h_node - prev_h).norm(dim=-1).mean().item())
                prev_h = h_node

                hist_exp = h_node[batch_idx]
                over_exp = overload[batch_idx]

                delta = self.refine(
                    tasks_cat, svc_exp, mf_exp,
                    current_logits.detach(),
                    hist_exp, over_exp,
                    indices=idx_exp,
                )
                # FIX: Damped fixed-point iteration for contraction control
                # z_{k+1} = (1-alpha)*z_k + alpha*(P + R(z_k))
                target_logits = prop_logits + delta

                # Trace: Z-residual (how much logit changes this step)
                self.residual_trace.append((target_logits - current_logits).norm(dim=-1).mean().item())

                current_logits = (1 - self.alpha) * current_logits + self.alpha * target_logits

            # Final step to ensure hist_change_trace has enough points if possible
            # (Matches KStepMonitor expectations)
            if K_REFINE_STEPS > 0 and len(self.hist_change_trace) == 0:
                 # If K=1, hist_change is 0
                 self.hist_change_trace.append(0.0)

            # Compute one extra refine call to measure true fp residual
            h_fp, overload_fp = self._compute_hist_and_overload(
                current_logits, masks_exp, service_states, batch_idx, B, total_tasks,
            )
            # FIX: residual should be based on the operator F(z) = P + R(z)
            delta_fp = self.refine(
                tasks_cat, svc_exp, mf_exp,
                current_logits.detach(),
                h_fp[batch_idx], overload_fp[batch_idx],
                indices=idx_exp,
            )
            # True residual: ||F(z) - z||
            fp_residual = (prop_logits + delta_fp - current_logits).norm(dim=-1).mean()
            self.last_residual = fp_residual.item()

            # Equilibrium histogram h* and load variance
            h_star = h_node  # (B, M) — equilibrium histogram
            self.equilibrium_load_var = h_star.var(dim=1).mean().item()

            # ═══ 5. FUSION — final_logits = current equilibrium ═══
            final_logits = current_logits
            if masks_exp is not None:
                final_logits = final_logits.masked_fill(masks_exp == 0, -1e9)
            if self.exclude_zero and self.u_action_dim > 1:
                final_logits[:, 0] = -1e9
            final_logits = self._sanitize_logits(final_logits)

            # ═══ 7. DIAGNOSTICS (Periodic) ═══
            self.inference_counter += 1
            if self.inference_counter % 100 == 0:
                # To show the 'progress', we can calculate the residual after Step 1 as well
                with torch.no_grad():
                    # Residual after first step (k=0)
                    h_init, o_init = self._compute_hist_and_overload(prop_logits, masks_exp, service_states, batch_idx, B, total_tasks)
                    d_init = self.refine(tasks_cat, svc_exp, mf_exp, prop_logits, h_init[batch_idx], o_init[batch_idx], indices=idx_exp)
                    res_init = (prop_logits + d_init - prop_logits).norm(dim=-1).mean().item()

                print(
                    f"[INFER] Step {self.inference_counter:5d} | "
                    f"Refinement convergence: {res_init:.4f} → {self.last_residual:.6f} | "
                    f"LoadVar: {self.proposal_load_var:.4f} → {self.equilibrium_load_var:.4f}"
                )
            if deterministic:
                actions_cat = final_logits.argmax(dim=-1)
                log_probs_cat = torch.zeros(total_tasks, device=device)
            else:
                dist = Categorical(logits=final_logits)
                actions_cat = dist.sample()
                log_probs_cat = dist.log_prob(actions_cat)

            # Trả về format List[Tensor] gốc bằng .split() và unbind()
            task_lens_list = task_lens.cpu().tolist()
            all_actions = list(actions_cat.split(task_lens_list))

            # FIX: store JOINT log-prob (sum, not mean) — matches PPO ratio theory
            sum_lp = torch.zeros(B, device=device).scatter_add_(0, batch_idx, log_probs_cat)
            all_log_probs = list(sum_lp.unbind())  # joint: sum_i log pi_i

            # ═══ 7. CRITIC — V(s, h*) ═══
            # FIX: pass h_star so critic sees the equilibrium population state
            h_star_per_agent = h_star  # (B, M)
            all_values = self.critic(
                general_task, service_states, pred_mfs, h_star_per_agent,
                indices=agent_indices
            )

            # ═══ 8. LOG INFERENCE ═══
            if not hasattr(self, '_infer_logged'):
                self._infer_logged = 0
            if self._infer_logged < 3:
                print(f"  [INFER] h_star[0]: {[f'{x:.2f}' for x in h_star[0].tolist()]}")
                print(f"  [INFER] overload[0]: {[f'{x:.3f}' for x in overload[0].tolist()]}")
                print(f"  [INFER] fp_residual: {self.last_residual:.6f} | "
                      f"load_var proposal→equil: {self.proposal_load_var:.4f}→{self.equilibrium_load_var:.4f}")
                self._infer_logged += 1

        return all_actions, all_log_probs, all_values

    @staticmethod
    def _sanitize_logits(z):
        """Đảm bảo logits hợp lệ: không NaN, không toàn -inf."""
        z = torch.where(
            torch.isnan(z),
            torch.tensor(-1e6, device=z.device),
            z,
        )
        z = torch.where(
            torch.isinf(z),
            torch.tensor(-1e6, device=z.device),
            z,
        )
        # Nếu TẤT CẢ logits trong 1 hàng đều <= -1e5 → uniform
        all_bad = (z <= -1e5).all(dim=-1, keepdim=True)
        z = torch.where(all_bad, torch.zeros_like(z), z)
        return z

    def _compute_hist_and_overload(
        self,
        logits,
        masks_exp,
        svc_batch,
        batch_idx,
        B_batch,
        total_n,
    ):
        """Tính histogram h_node và overload từ logits hiện tại.

        Args:
            logits:    (total_n, u_action_dim) — logits của TỪNG task
            masks_exp: (total_n, u_action_dim) hoặc None
            svc_batch: (B_batch, 2*M) — service states của mini-batch
            batch_idx: (total_n,) — ánh xạ task → agent index trong mini-batch
            B_batch:   int — số agents trong mini-batch
            total_n:   int — tổng số tasks

        Returns:
            h_node:  (B_batch, M) — histogram load per model type
            overload: (B_batch, M) — normalised overload
        """
        # FIX: detach only when called from no_grad contexts (inference / warm-up);
        # the caller is responsible for passing detached logits in those contexts.
        # Here we keep whatever grad-mode the caller sets so the final step in
        # learn() can propagate gradients through h(z).
        logits_for_hist = logits
        if masks_exp is not None:
            logits_for_hist = logits_for_hist.masked_fill(masks_exp == 0, -1e9)

        probs = F.softmax(self._sanitize_logits(logits_for_hist), dim=-1)
        probs_M = probs.view(total_n, self.M, self.max_models).sum(dim=2)  # (total_n, M)

        h_node = torch.zeros(B_batch, self.M, device=self.device)
        h_node.scatter_add_(0, batch_idx.unsqueeze(1).expand(-1, self.M), probs_M)

        # FIX: Normalize histogram by task count so distribution is invariant to N
        task_counts = torch.zeros(B_batch, 1, device=self.device).scatter_add_(
            0, batch_idx.unsqueeze(1), torch.ones(total_n, 1, device=self.device)
        )
        h_node = h_node / (task_counts + 1e-8)

        f_v = svc_batch[:, :self.M]  # (B_batch, M)
        h_weighted_mean = (h_node * f_v).sum(dim=1, keepdim=True) / (f_v.sum(dim=1, keepdim=True) + 1e-8)
        overload = (h_node - h_weighted_mean) / (h_weighted_mean + 1e-8)

        return h_node, overload

    # ----------------------------------------------------------
    # ② STORE TRANSITION + TRAIN MF
    # ----------------------------------------------------------

    def store_transition_train_mf_batch(
            self, service_states, task_states, prev_mfs, curr_mfs,
            actions, rewards, next_service_states, dones,
            agent_ids, log_probs, values, masks=None,
    ):
        """
        1. Train MF (supervised) với ground truth curr_mfs
        2. Store vào buffer
        """
        # ── Train MF ──
        general_tasks = self.tasks_to_general(task_states)  # (B, 7)
        loss_mf = self.learn_mf_batch(
            general_tasks, service_states, prev_mfs, curr_mfs, agent_ids
        )

        # ── Store ──
        self.memory.add_batch(
            service_states, task_states, prev_mfs, curr_mfs,
            actions, rewards, next_service_states, dones,
            log_probs, values, agent_ids, masks=masks,
        )

        return loss_mf

    def learn_mf_batch(self, general_tasks, service_states,
                       prev_mfs, ground_truth_mfs, agent_ids):
        """
        Train MF net supervised với ground truth MF.

        Args:
            general_tasks:       (B, 7)
            service_states:      (B, 2*M)
            prev_mfs:            (B, mf_dim)
            ground_truth_mfs:    (B, mf_dim)
            agent_ids:           (B,)
        """
        gt = torch.as_tensor(general_tasks, dtype=torch.float32, device=self.device)
        s = torch.as_tensor(service_states, dtype=torch.float32, device=self.device)
        pm = torch.as_tensor(prev_mfs, dtype=torch.float32, device=self.device)
        gf = torch.as_tensor(ground_truth_mfs, dtype=torch.float32, device=self.device)

        # MF input: general_task || service_state || prev_mf
        mf_input = torch.cat([gt, s, pm], dim=-1)
        pred_mf = self.mf_net(mf_input, indices=agent_ids)
        loss = self.loss_fn(pred_mf, gf)

        self.mf_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mf_net.parameters(), max_norm=5.0)
        self.mf_optimizer.step()
        return loss.item()

    # ----------------------------------------------------------
    # ③ PPO LEARN
    # ----------------------------------------------------------
    def _unpack_task_batch(self, task_batch_cat, task_lens):
        """
        Args:
            task_batch_cat: (total_tasks, task_dim) — concatenated
            task_lens:      (dataset_size,)          — số task mỗi sample
        Returns:
            task_states:    List[(N_i, task_dim)]
        """
        task_states = []
        offset = 0
        for n_i in task_lens:
            n_i = int(n_i.item())
            task_states.append(task_batch_cat[offset:offset + n_i])
            offset += n_i
        return task_states

    def _estimate_h_star(self, general_task, service_states, mf_states,
                         task_states_list, agent_indices, steps=3):
        """No-grad helper to estimate h* for GAE bootstrapping."""
        B = service_states.shape[0]
        device = self.device

        with torch.no_grad():
            task_lens = torch.tensor([t.shape[0] for t in task_states_list], device=device)
            total_tasks = int(task_lens.sum().item())
            batch_idx = torch.repeat_interleave(torch.arange(B, device=device), task_lens)

            tasks_cat = torch.cat(task_states_list, dim=0).to(device).float()
            svc_exp = service_states[batch_idx]
            mf_exp = mf_states[batch_idx]
            idx_exp = agent_indices[batch_idx]

            prop_logits = self.proposal(tasks_cat, svc_exp, mf_exp, indices=idx_exp)

            current_logits = prop_logits
            for _ in range(steps):
                h, o = self._compute_hist_and_overload(
                    current_logits, None, service_states, batch_idx, B, total_tasks
                )
                delta = self.refine(tasks_cat, svc_exp, mf_exp, current_logits,
                                     h[batch_idx], o[batch_idx], indices=idx_exp)
                current_logits = (1 - self.alpha) * current_logits + self.alpha * (prop_logits + delta)

            h_star, _ = self._compute_hist_and_overload(
                current_logits, None, service_states, batch_idx, B, total_tasks
            )
        return h_star

    def learn(self, agents_ids=None, zeta=1.0, temp=1.0):
        from matrix_source.trainers.ppo_stategy import compute_gae

        if agents_ids is not None:
            agents_ids = agents_ids.to(self.device).view(-1)

        data = self.memory.get_all_ready(min_size=self.min_batch_size, agent_ids_pool=agents_ids)
        if data is None:
            return None

        self.entropy_coef = max(
            self.entropy_coef * self.entropy_decay_rate,
            self.min_entropy_coef,
        )

        # ═══ 1. UNPACK ═══
        (service_states, task_batch_cat, task_lens, actions_cat, action_lens,
         prev_mfs, curr_mfs, rewards, next_service_states, dones,
         old_log_probs, old_values, masks, agent_ids) = data

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

        # Reconstruct variable-length task states
        task_states_list = self._unpack_task_batch(task_batch_cat, task_lens)
        general_tasks = self.tasks_to_general(task_states_list).to(self.device)

        # ═══ 2. PRE-COMPUTE (no grad) ═══
        with torch.no_grad():
            # GAE
            mf_in = torch.cat([general_tasks, next_service_states, curr_mfs.to(self.device)], dim=-1)
            next_mf = self.mf_net(mf_in, indices=agent_ids)

            # FIX: Estimate h* for next states for proper bootstrap rather than using zeros
            h_next_star = self._estimate_h_star(
                general_tasks, next_service_states, next_mf, task_states_list, agent_ids, steps=3
            )
            next_val = self.critic(general_tasks, next_service_states, next_mf, h_next_star, indices=agent_ids)

            advantages = compute_gae(
                rewards, next_val, old_values, dones, agent_ids,
                self.gamma, self.lmbda,
            )
            returns = advantages + old_values
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # MF for entire dataset once
            mf_in_all = torch.cat([general_tasks, service_states, prev_mfs], dim=-1)
            detached_mfs_all = self.mf_net(mf_in_all, indices=agent_ids).detach()

        # Pre-process masks → single tensor or None
        if masks is not None and any(m is not None for m in masks):
            # Find first non-None to check dim
            first_valid = next(m for m in masks if m is not None)
            if first_valid.dim() == 1:
                # Replace None entries with zeros
                clean_masks = [
                    m if m is not None else torch.zeros_like(first_valid)
                    for m in masks
                ]
                all_masks = torch.stack(clean_masks).to(self.device)
            else:
                all_masks = torch.cat([m for m in masks if m is not None], dim=0).to(self.device)
        else:
            all_masks = None

        # Pre-compute flat index slicing (once, outside loop)
        task_offsets = torch.zeros(dataset_size, dtype=torch.long, device=self.device)
        task_offsets[1:] = task_lens.cumsum(0)[:-1]
        all_flat_idx = torch.arange(total_tasks_flat, device=self.device)

        # ═══ 3. PPO TRAINING LOOP (vectorized) ═══
        epoch_metrics = {
            'v': 0.0, 'p': 0.0, 'ent': 0.0, 'kl': 0.0,
            'fp_res': 0.0, 'delta': 0.0, 'shift': 0.0, 'hist_s': 0.0,
            'lv_p': 0.0, 'lv_s': 0.0, 'ov_p': 0.0, 'ov_s': 0.0
        }
        total_batches = 0

        for _ in range(self.k_epochs):
            perm = torch.randperm(dataset_size, device=self.device)

            for start in range(0, dataset_size, self.batch_size):
                idx = perm[start:start + self.batch_size]
                B_sub = len(idx)

                # ── Slice mini-batch ──
                b_svc = service_states[idx]
                b_prev = prev_mfs[idx]
                b_old_lp = old_log_probs[idx]
                b_adv = advantages[idx]
                b_ret = returns[idx]
                b_aids = agent_ids[idx]
                b_gen = general_tasks[idx]
                b_mf = detached_mfs_all[idx]
                b_t_lens = task_lens[idx]

                # Flat indices for this mini-batch
                segments = [
                    all_flat_idx[task_offsets[i]:task_offsets[i] + task_lens[i]]
                    for i in idx
                ]
                flat_indices = torch.cat(segments)

                t_cat = task_batch_cat[flat_indices]
                act_cat = actions_cat[flat_indices]
                total_n = t_cat.shape[0]

                # Task → agent mapping within mini-batch
                batch_idx = torch.repeat_interleave(
                    torch.arange(B_sub, device=self.device), b_t_lens,
                )

                svc_exp = b_svc[batch_idx]
                mf_exp = b_mf[batch_idx]
                aids_exp = b_aids[batch_idx]
                masks_exp = all_masks[idx][batch_idx] if all_masks is not None else None

                # ── PROPOSAL (1 forward pass) ──
                # FIX: apply temperature scaling
                prop_logits = self.proposal(t_cat, svc_exp, mf_exp, indices=aids_exp) / temp

                # ── K-STEP EQUILIBRIUM REFINEMENT ──
                # FIX: proposal is anchor (detached). Refine is the sole fixed-point solver.
                # Fixed-point equation: z* = P + R(z*, h(z*))
                # Warm-up: K-1 steps without gradient (find approximate z*)
                prop_logits_detached = prop_logits.detach()

                with torch.no_grad():
                    current_logits = prop_logits_detached

                    for _ in range(K_REFINE_STEPS - 1):
                        h_node, overload = self._compute_hist_and_overload(
                            current_logits, masks_exp, b_svc, batch_idx, B_sub, total_n,
                        )
                        delta = self.refine(
                            t_cat, svc_exp, mf_exp,
                            current_logits,            # no extra detach — already no_grad
                            h_node[batch_idx], overload[batch_idx],
                            indices=aids_exp,
                        )
                        # FIX: Damped fixed-point update z_{k+1} = (1-alpha)z_k + alpha*(P+R)
                        target_z = prop_logits_detached + delta
                        current_logits = (1 - self.alpha) * current_logits + self.alpha * target_z

                    # Proposal load stats
                    h_prop, over_prop = self._compute_hist_and_overload(
                        prop_logits_detached, masks_exp, b_svc, batch_idx, B_sub, total_n,
                    )
                    self.proposal_load_var = h_prop.var(dim=1).mean().item()
                    prop_over_val = over_prop.abs().mean().item()

                # ── Final step: OPEN GRADIENT through h(z) and refine ──
                # We use prop_logits (with grad) and current_logits (near equilibrium)
                # to compute the final step. This allows PropActor to see h and delta.
                h_node, overload = self._compute_hist_and_overload(
                    current_logits, masks_exp, b_svc, batch_idx, B_sub, total_n,
                )
                hist_exp = h_node[batch_idx]
                over_exp = overload[batch_idx]

                # Equilibrium load variance
                h_star = h_node  # (B_sub, M)
                self.equilibrium_load_var = h_star.var(dim=1).mean().item()

                # Gradient flows through refine.
                # FIX: remove .detach() from current_logits branch so operator learns dR/dz.
                # We still use the warm-up equilibrium as the anchor for h.
                delta_logits = self.refine(
                    t_cat, svc_exp, mf_exp,
                    current_logits,            # FIX: branch ∂R/∂z enabled
                    hist_exp, over_exp,       # FIX: branch ∂R/∂h enabled
                    indices=aids_exp,
                )

                # FIX: fixed-point residual loss must have gradient flowing into RefineActor
                # We use (P_detach + R(z, h) - z_detach)^2 to force R to find the equilibrium.
                fp_residual = (prop_logits_detached + delta_logits - current_logits.detach()).pow(2).mean()
                self.last_residual = fp_residual.item()

                # ── FUSION — final equilibrium logits ──
                # PropActor learns through P directly AND through the damped R branch.
                final_logits = (1 - self.alpha) * current_logits + self.alpha * (prop_logits + delta_logits)
                if masks_exp is not None:
                    final_logits = final_logits.masked_fill(masks_exp == 0, -1e9)
                if self.exclude_zero and self.u_action_dim > 1:
                    final_logits[:, 0] = -1e9
                final_logits = self._sanitize_logits(final_logits)

                # ── LOG PROBS & ENTROPY ──
                dist = Categorical(logits=final_logits)
                lp_all = dist.log_prob(act_cat)  # (total_n,)

                # FIX: Entropy penalty only on Proposal to separate exploration from solving
                prop_dist = Categorical(logits=prop_logits)
                prop_entropy_all = prop_dist.entropy()

                # FIX: JOINT log-prob (sum, not mean) to match PPO ratio theory
                # r = exp(sum log π_i - sum log π_i^old)
                sum_lp = torch.zeros(B_sub, device=self.device).scatter_add_(0, batch_idx, lp_all)
                new_lp = sum_lp  # joint: Σ log π_i

                sum_ent = torch.zeros(B_sub, device=self.device).scatter_add_(0, batch_idx, prop_entropy_all)
                ent_m = sum_ent / b_t_lens.float()  # mean entropy per agent (for the loss coef)

                # ── K-STEP DETAILED DIAGNOSTICS ──
                with torch.no_grad():
                    delta_norm = delta_logits.norm(dim=-1).mean().item()
                    logit_shift = (final_logits - prop_logits).norm(dim=-1).mean().item()
                    hist_shift = (h_star - h_prop).abs().mean().item()
                    star_over_val = over_exp.abs().mean().item()

                # ── PPO CLIPPING ──
                ratio = torch.exp(new_lp - b_old_lp)  # both are joint sums
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv

                # FIX: KL Monitoring for update stability
                with torch.no_grad():
                    approx_kl = (b_old_lp - new_lp).mean()

                actor_loss = (
                    -torch.min(surr1, surr2).mean()
                    - self.entropy_coef * ent_m.mean()
                    + 0.01 * fp_residual          # FIX: RefineActor now learns to minimize fixed-point error
                )

                # ── ACTOR UPDATE ──
                self.optimizer_proposal.zero_grad(set_to_none=True)
                self.optimizer_refine.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.proposal.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.refine.parameters(), 0.5)
                self.optimizer_proposal.step()
                self.optimizer_refine.step()

                # ── CRITIC UPDATE — V(s, h*) ──
                # FIX: Critic observes equilibrium histogram h* (detached for stability)
                h_star_b = h_star.detach()
                c_vals = self.critic(b_gen, b_svc, b_mf, h_star_b, indices=b_aids)
                c_loss = F.mse_loss(c_vals, b_ret)

                self.optimizer_critic.zero_grad(set_to_none=True)
                c_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

                # ── Metrics ──
                epoch_metrics['v'] += c_loss.item()
                epoch_metrics['p'] += actor_loss.item()
                epoch_metrics['ent'] += ent_m.mean().item()
                epoch_metrics['kl'] += approx_kl.item()

                # K-step specific
                epoch_metrics['fp_res'] += self.last_residual
                epoch_metrics['delta']  += delta_norm
                epoch_metrics['shift']  += logit_shift
                epoch_metrics['hist_s'] += hist_shift
                epoch_metrics['lv_p']   += self.proposal_load_var
                epoch_metrics['lv_s']   += self.equilibrium_load_var
                epoch_metrics['ov_p']   += prop_over_val
                epoch_metrics['ov_s']   += star_over_val

                total_batches += 1

        self.learn_step_counter += 1

        log_freq = 10 if self.node_type == "Edge_Group" else 100
        if self.learn_step_counter % log_freq == 0 and total_batches > 0:
            n = total_batches
            print(
                f"[{self.node_type}] Step {self.learn_step_counter:5d} | "
                f"V: {epoch_metrics['v'] / n:.5f} | "
                f"P: {epoch_metrics['p'] / n:.5f} | "
                f"Ent: {epoch_metrics['ent'] / n:.4f} | "
                f"KL: {epoch_metrics['kl'] / n:.6f}"
            )
            # Detailed K-step Equilibrium Diagnostics
            print(
                f"  > [KSTEP] res={epoch_metrics['fp_res']/n:.4f} | "
                f"delta={epoch_metrics['delta']/n:.4f} | "
                f"shift={epoch_metrics['shift']/n:.4f} | "
                f"hist={epoch_metrics['hist_s']/n:.5f} | "
                f"lv={epoch_metrics['lv_p']/n:.4f}->{epoch_metrics['lv_s']/n:.4f} | "
                f"over={epoch_metrics['ov_p']/n:.4f}->{epoch_metrics['ov_s']/n:.4f} | "
                f"Temp={temp:.2f}"
            )

        self.memory.clear()
        return epoch_metrics['v'] / total_batches if total_batches > 0 else 0.0

    # ----------------------------------------------------------
    # ④ Checkpoint
    # ----------------------------------------------------------

    def save(self, path: str):
        checkpoint = {
            'proposal': self.proposal.state_dict(),
            'refine': self.refine.state_dict(),
            'critic': self.critic.state_dict(),
            'mf_net': self.mf_net.state_dict(),
            'proposal_opt': self.optimizer_proposal.state_dict(),
            'refine_opt': self.optimizer_refine.state_dict(),
            'critic_opt': self.optimizer_critic.state_dict(),
            'mf_opt': self.mf_optimizer.state_dict(),
            'learn_step': self.learn_step_counter,
            'entropy_coef': self.entropy_coef,
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.proposal.load_state_dict(ckpt['proposal'])
        self.refine.load_state_dict(ckpt['refine'])
        self.critic.load_state_dict(ckpt['critic'])
        self.mf_net.load_state_dict(ckpt['mf_net'])
        self.optimizer_proposal.load_state_dict(ckpt['proposal_opt'])
        self.optimizer_refine.load_state_dict(ckpt['refine_opt'])
        self.optimizer_critic.load_state_dict(ckpt['critic_opt'])
        self.mf_optimizer.load_state_dict(ckpt['mf_opt'])
        self.learn_step_counter = ckpt.get('learn_step', 0)
        self.entropy_coef = ckpt.get('entropy_coef', self.initial_entropy_coef)

    def set_lr_factor(self, factor: float):
        opts = [self.optimizer_proposal, self.optimizer_refine,
                self.optimizer_critic, self.mf_optimizer]
        for opt in opts:
            for pg in opt.param_groups:
                pg['lr'] *= factor
        print(
            f"[{self.node_type}] LR scaled ×{factor}. "
            f"Proposal LR: {self.optimizer_proposal.param_groups[0]['lr']:.6f}"
        )


