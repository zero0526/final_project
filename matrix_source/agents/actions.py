import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import math

from matrix_source.agents.buffer.rollout_buffer import MultiAgentRolloutBuffer
from matrix_source.agents.residual_net import ResidualCritic, RefineActor, ProposalActor, MFNetwork

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

        # ── Optimizers ──
        self.optimizer_proposal = optim.Adam(self.proposal.parameters(), lr=lr)
        self.optimizer_refine = optim.Adam(self.refine.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
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
            pred_mfs = self.mf_net(torch.cat([general_task, service_states, prev_mfs], dim=-1), indices=agent_indices)

            task_lens = torch.tensor([t.shape[0] for t in task_states], device=device)
            total_tasks = int(task_lens.sum().item())
            batch_idx = torch.repeat_interleave(torch.arange(B, device=device), task_lens)

            tasks_cat = torch.cat(task_states, dim=0).to(device).float()
            svc_exp, mf_exp, idx_exp = service_states[batch_idx], pred_mfs[batch_idx], agent_indices[batch_idx]

            if masks_batch is not None:
                masks_exp = torch.stack(masks_batch).to(device)[batch_idx] if masks_batch[0].dim() == 1 else torch.cat(
                    masks_batch, dim=0).to(device)
            else:
                masks_exp = None

            prop_logits = self.proposal(tasks_cat, svc_exp, mf_exp, indices=idx_exp)

            # Tính h_node CHỈ ĐỂ BỎ VÀO REFINE ACTOR
            h_node, overload = self._compute_hist_and_overload(prop_logits.detach(), masks_exp, service_states,
                                                               batch_idx, B, total_tasks)
            self.proposal_load_var = h_node.var(dim=1).mean().item()

            if phrase == "Proposal_Only":
                delta_logits = torch.zeros_like(prop_logits)
            else:
                delta_logits = self.refine(tasks_cat, svc_exp, mf_exp, prop_logits.detach(), h_node[batch_idx],
                                           overload[batch_idx], indices=idx_exp)

            self.equilibrium_load_var = h_node.var(dim=1).mean().item()

            # FIX Ở ĐÂY: Gán vào biến trước khi mask và sanitize
            final_logits = prop_logits + self.alpha * delta_logits
            if masks_exp is not None:
                final_logits = final_logits.masked_fill(masks_exp == 0, -1e9)
            if self.exclude_zero and self.u_action_dim > 1:
                final_logits[:, 0] = -1e9
            final_logits = self._sanitize_logits(final_logits)

            if deterministic:
                actions_cat = final_logits.argmax(dim=-1)
                log_probs_cat = torch.zeros(total_tasks, device=device)
            else:
                dist = Categorical(logits=final_logits)
                actions_cat = dist.sample()
                log_probs_cat = dist.log_prob(actions_cat)

            task_lens_list = task_lens.cpu().tolist()
            all_actions = list(actions_cat.split(task_lens_list))
            sum_lp = torch.zeros(B, device=device).scatter_add_(0, batch_idx, log_probs_cat)
            all_log_probs = list(sum_lp.unbind())

            # CRITIC KHÔNG NHẬN h_node
            all_values = self.critic(general_task, service_states, pred_mfs, indices=agent_indices)

        return all_actions, all_log_probs, all_values, h_node

    @staticmethod
    def _sanitize_logits(z):
        z = torch.where(torch.isnan(z), torch.tensor(-1e6, device=z.device), z)
        z = torch.where(torch.isinf(z), torch.tensor(-1e6, device=z.device), z)
        z = torch.where((z <= -1e5).all(dim=-1, keepdim=True), torch.zeros_like(z), z)
        return z

    def _compute_hist_and_overload(self, logits, masks_exp, svc_batch, batch_idx, B_batch, total_n):
        logits_for_hist = logits.detach()
        if masks_exp is not None: logits_for_hist = logits_for_hist.masked_fill(masks_exp == 0, -1e9)
        probs = F.softmax(self._sanitize_logits(logits_for_hist), dim=-1)
        probs_M = probs.view(total_n, self.M, self.max_models).sum(dim=2)
        h_node = torch.zeros(B_batch, self.M, device=self.device)
        h_node.scatter_add_(0, batch_idx.unsqueeze(1).expand(-1, self.M), probs_M)
        f_v = svc_batch[:, :self.M]
        capacity_dist = f_v / (f_v.sum(dim=1, keepdim=True) + 1e-8)
        load_ratio = h_node / (capacity_dist + 1e-8)
        mean_load_ratio = (load_ratio * capacity_dist).sum(dim=1, keepdim=True)
        overload = (load_ratio - mean_load_ratio) / (mean_load_ratio + 1e-8)
        return h_node, overload

    # ----------------------------------------------------------
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

    def learn(self, phrase: str, step:int, agents_ids=None):
        from matrix_source.trainers.ppo_stategy import compute_gae
        if agents_ids is not None: agents_ids = agents_ids.to(self.device).view(-1)
        data = self.memory.get_all_ready(min_size=self.min_batch_size, agent_ids_pool=agents_ids)
        if data is None: return None

        # ENTROPY CHO 2 PHA
        current_ent_coef = self.initial_entropy_coef if phrase == "Proposal_Free" else \
            self.update_coeff(step)
        if phrase == "Proposal_Only": self.entropy_coef = current_ent_coef

        (service_states, task_batch_cat, task_lens, actions_cat, action_lens,
         prev_mfs, curr_mfs, rewards, next_service_states, dones,
         old_log_probs, old_values, masks, agent_ids) = data

        service_states, prev_mfs, agent_ids = service_states.to(self.device).float(), prev_mfs.to(
            self.device).float(), agent_ids.to(self.device).long()
        task_lens = task_lens.to(self.device).long()
        task_batch_cat = task_batch_cat.to(self.device).float()
        # FIX Ở ĐÂY: Đổi tên biến cục bộ thành b_actions để không đè lên biến actions_cat gốc
        b_actions_cat = actions_cat.to(self.device).long()

        old_log_probs, old_values = old_log_probs.to(self.device).squeeze(-1), old_values.to(self.device).squeeze(-1)
        rewards, dones = rewards.to(self.device).squeeze(-1), dones.to(self.device).squeeze(-1)

        dataset_size, total_tasks_flat = service_states.shape[0], task_batch_cat.shape[0]
        general_tasks = self.tasks_to_general(self._unpack_task_batch(task_batch_cat, task_lens)).to(self.device)

        with torch.no_grad():
            # CRITIC BOOTSTRAP KHÔNG CẦN h_node
            next_mf = self.mf_net(torch.cat([general_tasks, next_service_states, curr_mfs.to(self.device)], dim=-1),
                                  indices=agent_ids)
            next_val = self.critic(general_tasks, next_service_states, next_mf, indices=agent_ids)

            advantages = compute_gae(rewards, next_val, old_values, dones, agent_ids, self.gamma, self.lmbda)
            returns = advantages + old_values
            if advantages.numel() > 1: advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            detached_mfs_all = self.mf_net(torch.cat([general_tasks, service_states, prev_mfs], dim=-1),
                                           indices=agent_ids).detach()

        if masks is not None and any(m is not None for m in masks):
            first_valid = next(m for m in masks if m is not None)
            all_masks = torch.stack([m if m is not None else torch.zeros_like(first_valid) for m in masks], dim=0).to(
                self.device) if first_valid.dim() == 1 else torch.cat([m for m in masks if m is not None], dim=0).to(
                self.device)
        else:
            all_masks = None

        task_offsets = torch.zeros(dataset_size, dtype=torch.long, device=self.device)
        task_offsets[1:] = task_lens.cumsum(0)[:-1]
        all_flat_idx = torch.arange(total_tasks_flat, device=self.device)

        epoch_metrics = {'v': 0.0, 'p': 0.0, 'r': 0.0}
        total_batches = 0

        for _ in range(self.k_epochs):
            perm = torch.randperm(dataset_size, device=self.device)
            for start in range(0, dataset_size, self.batch_size):
                idx = perm[start:start + self.batch_size]
                B_sub = len(idx)
                b_svc, b_old_lp, b_adv, b_ret = service_states[idx], old_log_probs[idx], advantages[idx], returns[idx]
                b_aids, b_gen, b_mf, b_t_lens = agent_ids[idx], general_tasks[idx], detached_mfs_all[idx], task_lens[
                    idx]

                flat_indices = torch.cat([all_flat_idx[task_offsets[i]:task_offsets[i] + task_lens[i]] for i in idx])
                t_cat = task_batch_cat[flat_indices]
                # FIX Ở ĐÂY: Dùng b_actions_cat thay vì actions_cat
                act_cat = b_actions_cat[flat_indices]
                total_n = flat_indices.shape[0]

                batch_idx = torch.repeat_interleave(torch.arange(B_sub, device=self.device), b_t_lens)
                svc_exp, mf_exp, aids_exp = b_svc[batch_idx], b_mf[batch_idx], b_aids[batch_idx]
                masks_exp = all_masks[idx][batch_idx] if all_masks is not None else None

                # ══════════════════════════════════════════
                # 1. FORWARD PASS
                # ══════════════════════════════════════════
                prop_logits = self.proposal(t_cat, svc_exp, mf_exp, indices=aids_exp)
                h_node, overload = self._compute_hist_and_overload(prop_logits.detach(), masks_exp, b_svc, batch_idx,
                                                                   B_sub, total_n)

                if phrase == "Proposal_Only":
                    delta_logits = torch.zeros_like(prop_logits)
                else:
                    delta_logits = self.refine(t_cat, svc_exp, mf_exp, prop_logits.detach(), h_node[batch_idx],
                                               overload[batch_idx], indices=aids_exp)

                # ══════════════════════════════════════════
                # 2. TÍNH LOSS TÙY THEO PHASE
                # ══════════════════════════════════════════
                loss_proposal, loss_refine = 0.0, 0.0

                def apply_mask_and_sanitize(z):
                    if masks_exp is not None: z = z.masked_fill(masks_exp == 0, -1e9)
                    if self.exclude_zero and self.u_action_dim > 1: z[:, 0] = -1e9
                    return self._sanitize_logits(z)

                if phrase == "Proposal_Only":
                    final_logits = apply_mask_and_sanitize(prop_logits)
                    dist = Categorical(logits=final_logits)
                    sum_lp = torch.zeros(B_sub, device=self.device).scatter_add_(0, batch_idx, dist.log_prob(act_cat))
                    ratio = torch.exp(sum_lp - b_old_lp)
                    loss_proposal = -torch.min(ratio * b_adv,
                                               torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv).mean() \
                                    - current_ent_coef * dist.entropy().mean()

                elif phrase == "Proposal_Free":
                    # Nhánh Proposal (Stop-gradient đối với Refine)
                    z_P = apply_mask_and_sanitize(prop_logits + self.alpha * delta_logits.detach())
                    dist_P = Categorical(logits=z_P)
                    sum_lp_P = torch.zeros(B_sub, device=self.device).scatter_add_(0, batch_idx,
                                                                                   dist_P.log_prob(act_cat))
                    ratio_P = torch.exp(sum_lp_P - b_old_lp)
                    loss_proposal = -torch.min(ratio_P * b_adv, torch.clamp(ratio_P, 1 - self.eps_clip,
                                                                            1 + self.eps_clip) * b_adv).mean() \
                                    - current_ent_coef * dist_P.entropy().mean()

                    # Nhánh Refine (Stop-gradient đối với Proposal)
                    z_R = apply_mask_and_sanitize(prop_logits.detach() + self.alpha * delta_logits)
                    dist_R = Categorical(logits=z_R)
                    sum_lp_R = torch.zeros(B_sub, device=self.device).scatter_add_(0, batch_idx,
                                                                                   dist_R.log_prob(act_cat))
                    ratio_R = torch.exp(sum_lp_R - b_old_lp)
                    loss_refine = -torch.min(ratio_R * b_adv,
                                             torch.clamp(ratio_R, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv).mean()

                # ══════════════════════════════════════════
                # 3. CRITIC UPDATE (KHÔNG NHẬN h_node)
                # ══════════════════════════════════════════
                c_vals = self.critic(b_gen, b_svc, b_mf, indices=b_aids)
                c_loss = F.mse_loss(c_vals, b_ret)

                # ══════════════════════════════════════════
                # 4. BACKPROPAGATION
                # ══════════════════════════════════════════
                if isinstance(loss_proposal, torch.Tensor):
                    self.optimizer_proposal.zero_grad(set_to_none=True)
                    loss_proposal.backward()
                    torch.nn.utils.clip_grad_norm_(self.proposal.parameters(), 0.5)
                    self.optimizer_proposal.step()

                if isinstance(loss_refine, torch.Tensor):
                    self.optimizer_refine.zero_grad(set_to_none=True)
                    loss_refine.backward()
                    torch.nn.utils.clip_grad_norm_(self.refine.parameters(), 0.5)
                    self.optimizer_refine.step()

                self.optimizer_critic.zero_grad(set_to_none=True)
                c_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

                epoch_metrics['v'] += c_loss.item()
                epoch_metrics['p'] += loss_proposal.item() if isinstance(loss_proposal, torch.Tensor) else 0.0
                epoch_metrics['r'] += loss_refine.item() if isinstance(loss_refine, torch.Tensor) else 0.0
                total_batches += 1

        self.learn_step_counter += 1
        if self.learn_step_counter % 1 == 0 and total_batches > 0:
            n = total_batches
            print(
                f"[{self.node_type}][{phrase}] Step {self.learn_step_counter:5d} | V: {epoch_metrics['v'] / n:.5f} | P: {epoch_metrics['p'] / n:.5f} | R: {epoch_metrics['r'] / n:.5f}")

        self.memory.clear()
        return epoch_metrics['v'] / total_batches if total_batches > 0 else 0.0



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