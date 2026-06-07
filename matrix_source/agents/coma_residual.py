import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import math

from matrix_source.agents.buffer.com_buffer import MultiAgentCOMARolloutBuffer
from matrix_source.agents.COMA_Residual_net import COMAQNetwork, RefineActor, ProposalActor, MFNetwork


class COMAResidualRoutingAgent:
    def __init__(self, agent_id, node_type,
                 service_state_dim, mf_dim,
                 action_dim, u_action_dim, max_models,
                 mf_hidden_sizes=(64, 64), mf_lr=1e-3, buffer_min_size=32,
                 hidden_sizes=(128, 64), lr=3e-4,
                 gamma=0.99, alpha=1.0, lambda_coma=0.3,
                 proposal_only_cycles=400, alpha_warmup_cycles=200,
                 buffer_size=100_000, batch_size=128,
                 lam_gae=0.95, clip_eps=0.2, k_epochs=5,
                 entropy_coef_start=0.05, entropy_coef_end=0.001,
                 temperature=0.5, exclude_zero=False,
                 num_instances=1, device=None):

        self.agent_id = agent_id
        self.node_type = node_type
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.num_instances = num_instances
        self.action_dim = action_dim
        self.u_action_dim = u_action_dim
        self.exclude_zero = exclude_zero

        self.max_models = max_models
        self.M = u_action_dim // max_models  # Số lượng node (M * K = u_action_dim)

        # Hyperparameters
        self.gamma = gamma
        self.lmbda = lam_gae
        self.lambda_coma = lambda_coma
        self.initial_lambda_coma = lambda_coma  # <--- THÊM DÒNG NÀY ĐỂ LƯU GIÁ TRỊ GỐC
        self.eps_clip = clip_eps
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.min_batch_size = buffer_min_size
        
        # --- CẤU HÌNH ALPHA WARM-UP ---
        self.target_alpha = alpha             # Giá trị mục tiêu (1.0)
        self.initial_alpha = 0.1              # Giá trị khởi điểm an toàn
        self.alpha = self.initial_alpha       # Giá trị hiện tại
        self.proposal_only_cycles = proposal_only_cycles
        self.alpha_warmup_cycles = alpha_warmup_cycles
        # ------------------------------
        
        self.temperature = temperature

        # Entropy Annealing params
        self.initial_entropy_coef = entropy_coef_start
        self.entropy_coef_end = entropy_coef_end
        self.entropy_coef = entropy_coef_start

        # Dimensions
        TASK_DIM = 4
        GENERAL_TASK_DIM = 7
        self.service_state_dim = service_state_dim
        self.mf_dim = mf_dim

        # ── Networks ──
        self.mf_net = MFNetwork(
            input_dim=GENERAL_TASK_DIM + service_state_dim + mf_dim,
            output_dim=mf_dim,
            hidden_sizes=mf_hidden_sizes,
            num_instances=num_instances,
        ).to(self.device)

        self.proposal = ProposalActor(
            task_state=TASK_DIM, service_state=service_state_dim, mf_dim=mf_dim,
            action_dim=u_action_dim, hidden_sizes=hidden_sizes, num_instances=num_instances,
        ).to(self.device)

        self.refine = RefineActor(
            task_state=TASK_DIM, service_state=service_state_dim, mf_dim=mf_dim,
            action_dim=u_action_dim, hidden_sizes=hidden_sizes, num_instances=num_instances,
        ).to(self.device)

        self.critic = COMAQNetwork(
            general_task_state=TASK_DIM, service_state=service_state_dim, mf_dim=mf_dim,
            action_dim=u_action_dim, hidden_sizes=hidden_sizes, num_instances=num_instances,
        ).to(self.device)

        # ── Optimizers ──
        self.optimizer_proposal = optim.Adam(self.proposal.parameters(), lr=lr)
        self.optimizer_refine = optim.Adam(self.refine.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.mf_optimizer = optim.Adam(self.mf_net.parameters(), lr=mf_lr)
        self.loss_fn = nn.SmoothL1Loss()

        # ── Buffer ──
        self.memory = MultiAgentCOMARolloutBuffer(
            num_agents=num_instances, node_type=node_type,
            max_size_per_agent=buffer_size, service_state_dim=service_state_dim,
            action_dim=action_dim, h_dim= self.M, device=self.device,
        )
        self.learn_step_counter = 0

    def update_coeff(self, step):
        """Cập nhật entropy coef dựa vào step (giữ nguyên logic cũ của bạn)"""
        return self.entropy_coef_end + (self.initial_entropy_coef - self.entropy_coef_end) * math.exp(-0.0307 * step)

    # ----------------------------------------------------------
    # ① INFERENCE (ROLLING)
    # ----------------------------------------------------------
    def choose_action(self, state, prev_mf, mask=None, agent_idx=0, task_state=None, deterministic=False):
        idx_t = torch.tensor([agent_idx], device=self.device)
        if state.dim() == 1: state = state.unsqueeze(0)
        if prev_mf.dim() == 1: prev_mf = prev_mf.unsqueeze(0)
        if not isinstance(task_state, list): task_state = [task_state]
        if mask is not None and not isinstance(mask, list): mask = [mask]

        all_actions, all_log_probs, all_values, h_node, prop_logits_masked = self.choose_action_batch(
            service_states=state, prev_mfs=prev_mf, task_states=task_state,
            masks_batch=mask, agent_indices=idx_t, deterministic=deterministic,
        )
        return all_actions[0], all_log_probs[0], all_values[0]

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

            if total_tasks == 0:
                empty_actions = [[] for _ in range(B)]
                empty_lps = [torch.tensor(0.0, device=device) for _ in range(B)]
                empty_vals = torch.zeros(B, device=device)
                fake_h_node = torch.zeros(B, self.M, device=device)

                fake_prop_logits_list = [torch.empty(0, self.u_action_dim, device=device) for _ in range(B)]

                return empty_actions, empty_lps, list(empty_vals.unbind()), fake_h_node, fake_prop_logits_list

            batch_idx = torch.repeat_interleave(torch.arange(B, device=device), task_lens)
            tasks_cat = torch.cat(task_states, dim=0).to(device).float()
            svc_exp = service_states[batch_idx]
            mf_exp = pred_mfs[batch_idx]
            idx_exp = agent_indices[batch_idx]

            if masks_batch is not None:
                if isinstance(masks_batch, list):
                    if masks_batch[0].dim() == 1:
                        masks_exp = torch.stack(masks_batch).to(device)[batch_idx]
                    else:
                        masks_exp = torch.cat(masks_batch, dim=0).to(device)
                else:
                    masks_exp = masks_batch.to(device)[batch_idx] if masks_batch.dim() == 2 else masks_batch.to(device)
            else:
                masks_exp = None

            prop_logits = self.proposal(tasks_cat, svc_exp, mf_exp, indices=idx_exp)

            if masks_exp is not None:
                prop_logits_masked = prop_logits.masked_fill(masks_exp == 0, -1e9)
            else:
                prop_logits_masked = prop_logits

            h_node, overload = self._compute_hist_and_overload(
                prop_logits_masked, masks_exp, service_states, batch_idx, B, total_tasks
            )
            self.proposal_load_var = h_node.var(dim=1).mean().item()

            if phrase == "Proposal_Only":
                delta_logits = torch.zeros_like(prop_logits)
            else:
                delta_logits = self.refine(
                    tasks_cat, svc_exp, mf_exp, prop_logits.detach(),
                    h_node[batch_idx], overload[batch_idx], indices=idx_exp
                )

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

            # Tách thành List cho actions
            all_actions = list(actions_cat.split(task_lens_list))

            # Tổng hợp log_probs về cấp độ Group
            sum_lp = torch.zeros(B, device=device).scatter_add_(0, batch_idx, log_probs_cat)
            all_log_probs = list(sum_lp.unbind())

            # Tính Q-values và aggregate về cấp độ Group
            h_node_exp = h_node[batch_idx]
            q_vals_task = self.critic(tasks_cat, svc_exp, mf_exp, h_node_exp, indices=idx_exp)
            max_q_per_task = q_vals_task.max(dim=-1).values

            group_max_q_sum = torch.zeros(B, device=device)
            group_max_q_sum.scatter_add_(0, batch_idx, max_q_per_task)
            all_values = group_max_q_sum / task_lens.float().clamp(min=1)

            prop_logits_list = list(prop_logits_masked.split(task_lens_list))

        return all_actions, all_log_probs, list(all_values.unbind()), h_node, prop_logits_list

    @staticmethod
    def _sanitize_logits(z):
        z = torch.where(torch.isnan(z), torch.tensor(-1e6, device=z.device), z)
        z = torch.where(torch.isinf(z), torch.tensor(-1e6, device=z.device), z)
        z = torch.where((z <= -1e5).all(dim=-1, keepdim=True), torch.zeros_like(z), z)
        return z

    def _compute_hist_and_overload(self, logits, masks_exp, svc_batch, batch_idx, B_batch, total_n):
        logits_for_hist = logits.detach()
        if masks_exp is not None:
            logits_for_hist = logits_for_hist.masked_fill(masks_exp == 0, -1e9)

        # probs có shape: (total_n, u_action_dim) tức là (total_n, M * K)
        probs = F.softmax(self._sanitize_logits(logits_for_hist), dim=-1)

        probs_M = probs.view(total_n, self.M, self.max_models).sum(dim=2)  # Shape: (total_n, M)

        # 2. Khởi tạo h_node với kích thước mới: (B_batch, M)
        h_node = torch.zeros(B_batch, self.M, device=self.device)

        # 3. Scatter_add theo chiều M
        h_node.scatter_add_(0, batch_idx.unsqueeze(1).expand(-1, self.M), probs_M)

        # 4. Tính overload (dùng luôn h_node vừa tính, không cần biến h_node_M riêng nữa)
        f_v = svc_batch[:, :self.M]
        capacity_dist = f_v / (f_v.sum(dim=1, keepdim=True) + 1e-8)
        load_ratio = h_node / (capacity_dist + 1e-8)
        mean_load_ratio = (load_ratio * capacity_dist).sum(dim=1, keepdim=True)
        overload = (load_ratio - mean_load_ratio) / (mean_load_ratio + 1e-8)

        # Trả về h_node có shape (B_batch, M)
        return h_node, overload

        # ----------------------------------------------------------

    def store_transition_train_mf_batch(self, service_states, task_states, prev_mfs, curr_mfs,
                                        actions, rewards, next_service_states, dones,
                                        agent_ids, log_probs, values, masks=None,
                                        proposal_logits=None, h_nodes=None):
        general_tasks = self.tasks_to_general(task_states)
        loss_mf = self.learn_mf_batch(general_tasks, service_states, prev_mfs, curr_mfs, agent_ids)

        self.memory.add_batch(
            service_states=service_states, task_states=task_states,
            prev_mfs=prev_mfs, curr_mfs=curr_mfs, proposal_logits=proposal_logits,
            actions=actions, rewards=rewards, next_service_states=next_service_states,
            dones=dones, log_probs=log_probs, values=values, h_nodes=h_nodes,
            agent_ids=agent_ids, masks=masks
        )
        return loss_mf

    def learn_mf_batch(self, general_tasks, service_states, prev_mfs, ground_truth_mfs, agent_ids):
        gt = torch.as_tensor(ground_truth_mfs, dtype=torch.float32, device=self.device)
        s = torch.as_tensor(service_states, dtype=torch.float32, device=self.device)
        pm = torch.as_tensor(prev_mfs, dtype=torch.float32, device=self.device)
        g_tasks = torch.as_tensor(general_tasks, dtype=torch.float32, device=self.device)
        mf_input = torch.cat([g_tasks, s, pm], dim=-1)
        pred_mf = self.mf_net(mf_input, indices=agent_ids)
        loss = self.loss_fn(pred_mf, gt)
        self.mf_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mf_net.parameters(), max_norm=5.0)
        self.mf_optimizer.step()
        return loss.item()

    def _unpack_task_batch(self, task_batch_cat, task_lens):
        task_states = []
        offset = 0
        for n_i in task_lens:
            n_i = int(n_i.item())
            if n_i > 0:
                task_states.append(task_batch_cat[offset:offset + n_i])
            else:
                task_states.append(torch.empty(0, task_batch_cat.shape[-1], device=self.device))
            offset += n_i
        return task_states

    def _expand_by_lens(self, fixed_tensor, lens_tensor):
        expanded_list = []
        for i, length in enumerate(lens_tensor):
            length = int(length.item())
            if length > 0: expanded_list.append(fixed_tensor[i:i + 1].expand(length, -1))
        if expanded_list:
            return torch.cat(expanded_list, dim=0)
        else:
            return torch.empty(0, fixed_tensor.shape[1], device=self.device)

    def learn(self, phrase: str, step: int, agents_ids=None):
        from matrix_source.trainers.ppo_stategy import compute_gae
        if agents_ids is not None: agents_ids = agents_ids.to(self.device).view(-1)
        data = self.memory.get_all_ready(min_size=self.min_batch_size, agent_ids_pool=agents_ids)
        if data is None: return None

        current_ent_coef = self.initial_entropy_coef if phrase == "Proposal_Free" else self.update_coeff(step)
        if phrase == "Proposal_Only":
            self.entropy_coef = current_ent_coef
            self.lambda_coma = 0
            self.alpha = 0.0  # <--- Chắc chắn tắt Refine trong giai đoạn này
        else:
            # 1. Khôi phục lambda_coma
            self.lambda_coma = self.initial_lambda_coma
            
            # 2. ALPHA WARM-UP SCHEDULE
            # Tính số cycle đã trôi qua kể từ khi bắt đầu phase Proposal_Free
            free_phase_step = max(0, step - self.proposal_only_cycles)
            
            # Tính tiến độ warm-up (từ 0.0 đến 1.0)
            warmup_progress = min(1.0, free_phase_step / self.alpha_warmup_cycles)
            
            # Nội suy tuyến tính alpha từ 0.1 lên 1.0
            self.alpha = self.initial_alpha + (self.target_alpha - self.initial_alpha) * warmup_progress

        (service_states, task_batch_cat, task_lens, p_logits_cat, actions_cat, action_lens,
         prev_mfs, curr_mfs, rewards, next_service_states, dones,
         old_log_probs, old_values, h_node_buffer, masks, agent_ids) = data

        service_states = service_states.to(self.device).float()
        prev_mfs = prev_mfs.to(self.device).float()
        agent_ids = agent_ids.to(self.device).long()
        task_lens = task_lens.to(self.device).long()
        task_batch_cat = task_batch_cat.to(self.device).float()
        b_actions_cat = actions_cat.to(self.device).long()
        p_logits_cat = p_logits_cat.to(self.device).float()
        h_node_buffer = h_node_buffer.to(self.device).float()

        old_log_probs = old_log_probs.to(self.device).squeeze(-1)
        old_values = old_values.to(self.device).squeeze(-1)
        rewards = rewards.to(self.device).squeeze(-1)
        dones = dones.to(self.device).squeeze(-1)

        dataset_size, total_tasks_flat = service_states.shape[0], task_batch_cat.shape[0]

        general_tasks = self.tasks_to_general(self._unpack_task_batch(task_batch_cat, task_lens)).to(self.device)

        def expand_by_lens(tensor, lens):
            return torch.repeat_interleave(tensor, lens, dim=0)

        with torch.no_grad():
            # Tính MF cho bước tiếp theo
            next_mf = self.mf_net(torch.cat([general_tasks, next_service_states, curr_mfs], dim=-1), indices=agent_ids)

            # Expand các tensor cố định (per-group) thành per-task để khớp với task_batch_cat
            next_mf_exp = expand_by_lens(next_mf, task_lens)
            next_svc_exp = expand_by_lens(next_service_states, task_lens)
            next_h_node_exp = expand_by_lens(h_node_buffer, task_lens)
            next_aids_exp = expand_by_lens(agent_ids, task_lens)

            # Critic trả về Q-values cho tất cả actions: (Total_Tasks, u_action_dim)
            next_q_vals = self.critic(task_batch_cat, next_svc_exp, next_mf_exp, next_h_node_exp, indices=next_aids_exp)

            # Lấy max Q-value làm V(s') cho từng task
            next_val_task = next_q_vals.max(dim=-1).values

            # Aggregate về lại group-level để tính GAE
            next_val_grouped = torch.zeros(dataset_size, device=self.device)
            group_indices = torch.repeat_interleave(torch.arange(dataset_size, device=self.device), task_lens)
            next_val_grouped.scatter_add_(0, group_indices, next_val_task)
            next_val_grouped = next_val_grouped / task_lens.float().clamp(min=1)

            advantages = compute_gae(rewards, next_val_grouped, old_values, dones, agent_ids, self.gamma, self.lmbda)
            returns = advantages + old_values
            if advantages.numel() > 1: advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            detached_mfs_all = self.mf_net(torch.cat([general_tasks, service_states, prev_mfs], dim=-1),
                                           indices=agent_ids).detach()

        with torch.no_grad():
            # Expand các tensor hiện tại cho Hybrid Advantage calculation
            svc_exp_all = expand_by_lens(service_states, task_lens)
            mf_exp_all = expand_by_lens(detached_mfs_all, task_lens)
            h_node_exp_all = expand_by_lens(h_node_buffer, task_lens)
            aids_exp_all = expand_by_lens(agent_ids, task_lens)

            # 1. [ĐÃ KHÔI PHỤC] Lấy Q-values cho TẤT CẢ actions từ COMA Critic
            # Shape: q_vals_all = (Total_Tasks, u_action_dim)
            q_vals_all = self.critic(task_batch_cat, svc_exp_all, mf_exp_all, h_node_exp_all, indices=aids_exp_all)

            # 2. [ĐÃ KHÔI PHỤC] Calculate Q_f (Quality of Final Action)
            # b_actions_cat shape: (Total_Tasks,) -> .view(-1, 1) ép về (Total_Tasks, 1)
            actions_idx = b_actions_cat.view(-1, 1)
            q_f = q_vals_all.gather(1, actions_idx).squeeze(-1)  # Shape: (Total_Tasks,)

            # === ĐIỂM THAY ĐỔI CỦA BẠN (GIỮ NGUYÊN) ===
            # Quyết định cách lấy baseline Q_p
            use_sampling_for_baseline = (self.learn_step_counter > 100) and (phrase == "Proposal_Free")

            if use_sampling_for_baseline:
                # LẤY MẪU: Dùng khi Refine bị "chết lâm sàng"
                dist_p = Categorical(logits=p_logits_cat)
                p_actions = dist_p.sample()
            else:
                # ARGMAX: Dùng ở giai đoạn đầu (Proposal_Only) hoặc đầu Proposal_Free
                p_actions = p_logits_cat.argmax(dim=-1)

            p_actions_idx = p_actions.view(-1, 1)
            q_p = q_vals_all.gather(1, p_actions_idx).squeeze(-1)  # Shape: (Total_Tasks,)

            # 4. Calculate Baseline B (COMA Term)
            # Shape: (Total_Tasks,)
            pi_p = F.softmax(p_logits_cat, dim=-1)
            baseline_b = (pi_p * q_vals_all).sum(dim=-1)

            # 5. Hybrid Advantage: A_r = (Q_f - Q_p) + lambda * (Q_f - B)
            adv_improvement = q_f - q_p.detach()
            adv_coma = q_f - baseline_b.detach()
            hybrid_adv_task = adv_improvement + self.lambda_coma * adv_coma

            global_q_imp = adv_improvement.mean().item()
            global_hybrid_adv = hybrid_adv_task.mean().item()

            # Aggregate hybrid_adv về group-level
            hybrid_adv_grouped = torch.zeros(dataset_size, device=self.device)
            group_indices = torch.repeat_interleave(torch.arange(dataset_size, device=self.device), task_lens)
            hybrid_adv_grouped.scatter_add_(0, group_indices, hybrid_adv_task)

            if hybrid_adv_grouped.numel() > 1:
                hybrid_adv_grouped = (hybrid_adv_grouped - hybrid_adv_grouped.mean()) / (
                        hybrid_adv_grouped.std() + 1e-8)

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

        epoch_metrics = {
            'v_loss': 0.0, 'p_loss': 0.0, 'r_loss': 0.0,
            'delta_norm': 0.0, 'flip_rate': 0.0, 'kl_div': 0.0, 'refine_grad': 0.0
        }
        total_batches = 0

        for _ in range(self.k_epochs):
            perm = torch.randperm(dataset_size, device=self.device)
            for start in range(0, dataset_size, self.batch_size):
                idx = perm[start:start + self.batch_size]
                B_sub = len(idx)
                b_svc, b_old_lp, b_adv_gae, b_ret = service_states[idx], old_log_probs[idx], advantages[idx], returns[
                    idx]
                b_hybrid_adv = hybrid_adv_grouped[idx]

                # ✅ SỬA LỖI: Lấy cả general_tasks cho batch con và expand nó
                b_gen_tasks = general_tasks[idx]
                b_aids, b_mf, b_t_lens = agent_ids[idx], detached_mfs_all[idx], task_lens[idx]

                # Expand general_tasks cho batch con này
                gen_exp_sub = expand_by_lens(b_gen_tasks, b_t_lens)

                flat_indices = torch.cat([all_flat_idx[task_offsets[i]:task_offsets[i] + task_lens[i]] for i in idx])
                t_cat = task_batch_cat[flat_indices]  # Vẫn giữ t_cat nếu cần cho Actor, nhưng không dùng cho Critic
                act_cat = b_actions_cat[flat_indices]
                b_p_logits_task = p_logits_cat[flat_indices]
                total_n = flat_indices.shape[0]

                batch_idx = torch.repeat_interleave(torch.arange(B_sub, device=self.device), b_t_lens)
                svc_exp, mf_exp, aids_exp = b_svc[batch_idx], b_mf[batch_idx], b_aids[batch_idx]
                masks_exp = all_masks[idx][batch_idx] if all_masks is not None else None

                prop_logits = self.proposal(t_cat, svc_exp, mf_exp, indices=aids_exp)
                h_node_refine, overload_refine = self._compute_hist_and_overload(prop_logits.detach(), masks_exp, b_svc,
                                                                                 batch_idx, B_sub, total_n)
                h_node_refine_exp = h_node_refine[batch_idx]
                overload_refine_exp = overload_refine[batch_idx]

                if phrase == "Proposal_Only":
                    delta_logits = torch.zeros_like(prop_logits)
                else:
                    delta_logits = self.refine(t_cat, svc_exp, mf_exp, prop_logits.detach(), h_node_refine_exp,
                                               overload_refine_exp, indices=aids_exp)

                loss_proposal, loss_refine = 0.0, 0.0

                def apply_mask_and_sanitize(z):
                    if masks_exp is not None: z = z.masked_fill(masks_exp == 0, -1e9)
                    if self.exclude_zero and self.u_action_dim > 1: z[:, 0] = -1e9
                    return self._sanitize_logits(z)

                old_p_probs = F.softmax(b_p_logits_task, dim=-1)
                old_p_lp_task = old_p_probs.gather(1, act_cat.unsqueeze(1)).squeeze(1).log()
                old_p_lp_grouped = torch.zeros(B_sub, device=self.device)
                old_p_lp_grouped.scatter_add_(0, batch_idx, old_p_lp_task)

                if phrase == "Proposal_Only":
                    final_logits = apply_mask_and_sanitize(prop_logits)
                    dist = Categorical(logits=final_logits)
                    curr_p_lp_task = dist.log_prob(act_cat)
                    curr_p_lp_grouped = torch.zeros(B_sub, device=self.device)
                    curr_p_lp_grouped.scatter_add_(0, batch_idx, curr_p_lp_task)
                    ratio = torch.exp(curr_p_lp_grouped - old_p_lp_grouped)
                    loss_proposal = -torch.min(ratio * b_adv_gae, torch.clamp(ratio, 1 - self.eps_clip,
                                                                              1 + self.eps_clip) * b_adv_gae).mean() - current_ent_coef * dist.entropy().mean()

                elif phrase == "Proposal_Free":
                    z_P = apply_mask_and_sanitize(prop_logits + self.alpha * delta_logits.detach())
                    dist_P = Categorical(logits=z_P)
                    curr_p_lp_task = dist_P.log_prob(act_cat)
                    curr_p_lp_grouped = torch.zeros(B_sub, device=self.device)
                    curr_p_lp_grouped.scatter_add_(0, batch_idx, curr_p_lp_task)
                    ratio_P = torch.exp(curr_p_lp_grouped - old_p_lp_grouped)
                    loss_proposal = -torch.min(ratio_P * b_adv_gae, torch.clamp(ratio_P, 1 - self.eps_clip,
                                                                                1 + self.eps_clip) * b_adv_gae).mean() - current_ent_coef * dist_P.entropy().mean()

                    z_R = apply_mask_and_sanitize(prop_logits.detach() + self.alpha * delta_logits)
                    dist_R = Categorical(logits=z_R)
                    curr_r_lp_task = dist_R.log_prob(act_cat)
                    curr_r_lp_grouped = torch.zeros(B_sub, device=self.device)
                    curr_r_lp_grouped.scatter_add_(0, batch_idx, curr_r_lp_task)
                    old_r_lp_grouped = b_old_lp - old_p_lp_grouped
                    ratio_R = torch.exp(curr_r_lp_grouped - old_r_lp_grouped)
                    loss_refine = -torch.min(ratio_R * b_hybrid_adv, torch.clamp(ratio_R, 1 - self.eps_clip,
                                                                                 1 + self.eps_clip) * b_hybrid_adv).mean()

                    # Tracking metrics
                    epoch_metrics['delta_norm'] += delta_logits.norm(dim=-1).mean().item()
                    prop_actions = prop_logits.argmax(dim=-1)
                    final_logits_raw = prop_logits + self.alpha * delta_logits
                    final_actions = final_logits_raw.argmax(dim=-1)
                    epoch_metrics['flip_rate'] += (prop_actions != final_actions).float().mean().item()
                    prop_probs = F.softmax(apply_mask_and_sanitize(prop_logits), dim=-1).clamp(min=1e-8)
                    final_probs = F.softmax(apply_mask_and_sanitize(final_logits_raw), dim=-1).clamp(min=1e-8)
                    kl_div = (prop_probs * (prop_probs.log() - final_probs.log())).sum(dim=-1).mean().item()
                    epoch_metrics['kl_div'] += kl_div

                q_vals_batch = self.critic(t_cat, svc_exp, mf_exp, h_node_refine_exp, indices=aids_exp)
                q_f_batch = q_vals_batch.gather(1, act_cat.unsqueeze(1)).squeeze(1)
                q_f_grouped = torch.zeros(B_sub, device=self.device)
                q_f_grouped.scatter_add_(0, batch_idx, q_f_batch)
                q_f_mean = q_f_grouped / b_t_lens.float().clamp(min=1)
                c_loss = F.mse_loss(q_f_mean, b_ret)

                if isinstance(loss_proposal, torch.Tensor):
                    self.optimizer_proposal.zero_grad(set_to_none=True)
                    loss_proposal.backward()
                    torch.nn.utils.clip_grad_norm_(self.proposal.parameters(), 0.5)
                    self.optimizer_proposal.step()

                if isinstance(loss_refine, torch.Tensor):
                    self.optimizer_refine.zero_grad(set_to_none=True)
                    loss_refine.backward()
                    refine_grad_norm = torch.nn.utils.clip_grad_norm_(self.refine.parameters(), 0.5)
                    epoch_metrics['refine_grad'] += refine_grad_norm.item()
                    self.optimizer_refine.step()

                self.optimizer_critic.zero_grad(set_to_none=True)
                c_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

                epoch_metrics['v_loss'] += c_loss.item()
                epoch_metrics['p_loss'] += loss_proposal.item() if isinstance(loss_proposal, torch.Tensor) else 0.0
                epoch_metrics['r_loss'] += loss_refine.item() if isinstance(loss_refine, torch.Tensor) else 0.0
                total_batches += 1

        for k in epoch_metrics:
            epoch_metrics[k] /= max(total_batches, 1)
        epoch_metrics['q_imp'] = global_q_imp
        epoch_metrics['hybrid_adv'] = global_hybrid_adv

        self.learn_step_counter += 1
        if self.learn_step_counter % 1 == 0 and total_batches > 0:
            n = total_batches
            print(
                f"[{self.node_type}][{phrase}] Step {self.learn_step_counter:5d} | V: {epoch_metrics['v_loss']:.5f} | P: {epoch_metrics['p_loss']:.5f} | R: {epoch_metrics['r_loss']:.5f}")

        self.memory.clear()
        return epoch_metrics

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

        # Load weights cho các mạng neural
        for k in ['proposal', 'refine', 'critic', 'mf_net']:
            getattr(self, k).load_state_dict(ckpt[k])

        # Load weights cho các optimizer
        self.optimizer_proposal.load_state_dict(ckpt['proposal_opt'])
        self.optimizer_refine.load_state_dict(ckpt['refine_opt'])
        self.optimizer_critic.load_state_dict(ckpt['critic_opt'])
        self.mf_optimizer.load_state_dict(ckpt['mf_opt'])

        # Load các biến trạng thái
        self.learn_step_counter = ckpt.get('learn_step', 0)
        self.entropy_coef = ckpt.get('entropy_coef', self.initial_entropy_coef)

        print(f"✅ Successfully loaded model and optimizers from {path}")