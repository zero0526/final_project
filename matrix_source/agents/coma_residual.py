import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import math

from matrix_source.agents.buffer.com_buffer import MultiAgentCOMARolloutBuffer
from matrix_source.agents.COMA_Residual_net import CriticNetwork, RefineActor, ProposalActor, MFNetwork
from matrix_source.agents.sac_ec import Critic


class COMAResidualRoutingAgent:
    def __init__(self, agent_id, node_type,
                 service_state_dim, mf_dim,
                 action_dim, u_action_dim, max_models,
                 mf_hidden_sizes=(64, 64), mf_lr=1e-3, buffer_min_size=32,
                 hidden_sizes=(128, 64), lr=3e-4, residual_logit_scale = 2,
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
        self.proposal_only_cycles = proposal_only_cycles
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
        self.residual_logit_scale= residual_logit_scale

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

        self.critic = CriticNetwork(
            service_state=service_state_dim, mf_dim=mf_dim,
            action_dim=1, hidden_sizes=hidden_sizes, num_instances=num_instances,
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
    def choose_action(self, state, prev_mf, mask=None, agent_idx=0, task_state=None, deterministic=False, phrase="Proposal_Free"):
        idx_t = torch.tensor([agent_idx], device=self.device)
        if state.dim() == 1: state = state.unsqueeze(0)
        if prev_mf.dim() == 1: prev_mf = prev_mf.unsqueeze(0)
        if not isinstance(task_state, list): task_state = [task_state]
        if mask is not None and not isinstance(mask, list): mask = [mask]

        all_actions, all_log_probs, all_values, h_node, prop_logits_masked = self.choose_action_batch(
            service_states=state, prev_mfs=prev_mf, task_states=task_state,
            masks_batch=mask, agent_indices=idx_t, deterministic=deterministic, phrase=phrase
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
                            agent_indices=None, deterministic=False, phrase="Proposal_Free", metrics=None):
        """
        Orchestrates the action selection process using Proposal and Refine networks.
        """
        B = service_states.shape[0]
        device = self.device

        # 1. Prepare Data
        prep_data = self._prepare_inference_data(service_states, prev_mfs, task_states, agent_indices, masks_batch)
        if prep_data['total_tasks'] == 0:
            return self._handle_empty_batch(B, device)

        with torch.no_grad():
            # 2. Proposal & Confidence
            proposal_data = self._get_proposal_and_confidence(
                prep_data['tasks_cat'], prep_data['svc_exp'], prep_data['mf_exp'], 
                prep_data['idx_exp'], prep_data['masks_exp'], service_states, 
                prep_data['batch_idx'], B, prep_data['total_tasks']
            )

            # 3. Refine Logic
            final_logits = self._get_refined_logits(
                prep_data['tasks_cat'], prep_data['svc_exp'], prep_data['mf_exp'], prep_data['idx_exp'],
                proposal_data['prop_logits'],
                proposal_data['h_node'], proposal_data['h_wl'],
                prep_data['batch_idx'], prep_data['masks_exp'], phrase
            )

            # 4. Action Selection
            action_data = self._sample_actions(
                final_logits, prep_data['task_lens'], B, prep_data['batch_idx'], device, deterministic
            )

            # 5. Critic evaluation (for V-values) - Direct Baseline at Node level
            all_values = self._evaluate_critic_value(
                service_states, prep_data['pred_mfs'], proposal_data['h_node'], 
                prep_data['masks_batch'], metrics, agent_indices, B
            )

            prop_logits_list = list(proposal_data['prop_logits_masked'].split(prep_data['task_lens'].cpu().tolist()))

        return action_data['all_actions'], action_data['all_log_probs'], list(all_values.unbind()), proposal_data['h_node'], prop_logits_list

    def _prepare_inference_data(self, service_states, prev_mfs, task_states, agent_indices, masks_batch):
        device = self.device
        B = service_states.shape[0]

        if agent_indices is None:
            agent_indices = torch.zeros(B, dtype=torch.long, device=device)
        else:
            agent_indices = agent_indices.to(device).view(-1)

        service_states = service_states.to(device).float()
        prev_mfs = prev_mfs.to(device).float()
        general_task = self.tasks_to_general(task_states)
        
        pred_mfs = self.mf_net(torch.cat([general_task, service_states, prev_mfs], dim=-1), indices=agent_indices)
        task_lens = torch.tensor([t.shape[0] for t in task_states], device=device)
        total_tasks = int(task_lens.sum().item())

        if total_tasks == 0:
            return {'total_tasks': 0}

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

        return {
            'total_tasks': total_tasks, 'task_lens': task_lens, 'batch_idx': batch_idx,
            'tasks_cat': tasks_cat, 'svc_exp': svc_exp, 'mf_exp': mf_exp, 
            'idx_exp': idx_exp, 'masks_exp': masks_exp, 'pred_mfs': pred_mfs, 'masks_batch': masks_batch
        }

    def _handle_empty_batch(self, B, device):
        return ([[] for _ in range(B)], 
                [torch.tensor(0.0, device=device) for _ in range(B)], 
                list(torch.zeros(B, device=device).unbind()), 
                torch.zeros(B, self.M, device=device), 
                [torch.empty(0, self.u_action_dim, device=device) for _ in range(B)])

    def _get_proposal_and_confidence(self, tasks_cat, svc_exp, mf_exp, idx_exp, masks_exp,
                                     service_states, batch_idx, B, total_tasks):
        prop_logits = self.proposal(tasks_cat, svc_exp, mf_exp, indices=idx_exp)
        
        if masks_exp is not None:
            prop_logits_masked = prop_logits.masked_fill(masks_exp == 0, -1e9)
        else:
            prop_logits_masked = prop_logits

        probs = F.softmax(self._sanitize_logits(prop_logits_masked), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1, keepdim=True)
        top2_probs, _ = torch.topk(probs, k=2, dim=-1)
        margin = (top2_probs[:, 0] - top2_probs[:, 1]).unsqueeze(-1)
        confidence_metrics = torch.cat([entropy, margin], dim=-1)

        h_node, _ = self._compute_hist_and_overload(
            prop_logits_masked, masks_exp, service_states, batch_idx, B, total_tasks
        )
        self.proposal_load_var = h_node.var(dim=1).mean().item()

        # --- Compute quick workload estimate (mf_load for RefineActor) ---
        # P(node | task) summed over models → (total_tasks, M)
        probs_M = probs.view(total_tasks, self.M, self.max_models).sum(dim=2)
        # ds from tasks_cat[:, 0] (normalized data size)
        task_ds = tasks_cat[:, 0].view(-1, 1)
        wl_per_task = probs_M * task_ds  # (total_tasks, M) – unnormalized workload proxy
        # Aggregate per agent group → (B, M)
        h_wl = torch.zeros(B, self.M, device=self.device)
        h_wl.scatter_add_(0, batch_idx.unsqueeze(1).expand(-1, self.M), wl_per_task)

        return {
            'prop_logits': prop_logits, 
            'prop_logits_masked': prop_logits_masked,
            'confidence_metrics': confidence_metrics, 
            'h_node': h_node,
            'h_wl': h_wl,  # expected workload estimate per agent group
        }

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
        logits_for_hist = logits.detach()
        if masks_exp is not None:
            logits_for_hist = logits_for_hist.masked_fill(masks_exp == 0, -1e9)

        probs = F.softmax(self._sanitize_logits(logits_for_hist), dim=-1)
        probs_M = probs.view(total_n, self.M, self.max_models).sum(dim=2)  # (total_n, M)

        h_node = torch.zeros(B_batch, self.M, device=self.device)
        h_node.scatter_add_(0, batch_idx.unsqueeze(1).expand(-1, self.M), probs_M)

        f_v = svc_batch[:, :self.M]  # (B_batch, M)
        h_weighted_mean = (h_node * f_v).sum(dim=1, keepdim=True) / (f_v.sum(dim=1, keepdim=True) + 1e-8)
        overload = (h_node - h_weighted_mean) / (h_weighted_mean + 1e-8)

        return h_node, overload

    def _get_refined_logits(self, tasks_cat, svc_exp, mf_exp, idx_exp, prop_logits,
                            h_node, h_wl, batch_idx, masks_exp, phrase):
        if phrase == "Proposal_Only":
            delta_logits = torch.zeros_like(prop_logits)
        else:
            # h_wl: (B, M) workload estimate per group → expand to per-task
            h_wl_exp = h_wl[batch_idx]  # (total_tasks, M)
            delta_logits = self.refine(
                tasks_cat, svc_exp, mf_exp, prop_logits.detach(),
                h_node[batch_idx], h_wl_exp, indices=idx_exp
            )

        final_logits = prop_logits + self.residual_logit_scale * delta_logits
        if masks_exp is not None:
            final_logits = final_logits.masked_fill(masks_exp == 0, -1e9)
        if self.exclude_zero and self.u_action_dim > 1:
            final_logits[:, 0] = -1e9
            
        return self._sanitize_logits(final_logits)

    def _sample_actions(self, final_logits, task_lens, B, batch_idx, device, deterministic):
        probs = F.softmax(final_logits, dim=-1)
        if deterministic:
            actions_cat = final_logits.argmax(dim=-1)
            log_probs_cat = torch.zeros(final_logits.shape[0], device=device)
        else:
            dist = Categorical(probs=probs)
            actions_cat = dist.sample()
            log_probs_cat = dist.log_prob(actions_cat)

        task_lens_list = task_lens.cpu().tolist()
        all_actions = list(actions_cat.split(task_lens_list))
        sum_lp = torch.zeros(B, device=device).scatter_add_(0, batch_idx, log_probs_cat)
        
        return {'all_actions': all_actions, 'all_log_probs': list(sum_lp.unbind()), 'probs': probs}

    def _evaluate_critic_value(self, svc, mf, h_node, masks_batch, metrics, agent_indices, B):
        """Tính toán Baseline Value (V) cho toàn bộ Node/Group."""
        device = self.device
        
        # Prepare Mask Node level
        if masks_batch is not None:
            if isinstance(masks_batch, list):
                mask_node = torch.stack(masks_batch).to(device)
            else:
                mask_node = masks_batch.to(device)
            # COMA Critic often looks at model availability (mask[:, :, 0])
            mask_node = mask_node.view(B, self.M, self.max_models)[:, :, 0]
        else:
            mask_node = torch.ones(B, self.M, device=device)

        if metrics:
            workload = metrics['workload']
            ds_metrics = metrics['ds_metrics']
            deadline_metrics = metrics['deadline_metrics']
            omega = metrics['omega']
            batch_sizes = metrics['batch_size']
        else:
            workload = torch.zeros(B, self.M, device=device)
            ds_metrics = torch.zeros(B, 5, device=device)
            deadline_metrics = torch.zeros(B, 5, device=device)
            omega = torch.zeros(B, 1, device=device)
            batch_sizes = torch.ones(B, 1, device=device)

        # Call critic directly at node level (B rows)
        v_node = self.critic(
            svc, mf, h_node, workload, mask_node, 
            ds_metrics, deadline_metrics, omega, batch_sizes, 
            indices=agent_indices
        )
        return v_node

    @staticmethod
    def _expand_by_lens(tensor, lens):
        return torch.repeat_interleave(tensor, lens, dim=0)

    @staticmethod
    def _sanitize_logits(z):
        z = torch.where(torch.isnan(z), torch.tensor(-1e6, device=z.device), z)
        z = torch.where(torch.isinf(z), torch.tensor(-1e6, device=z.device), z)
        z = torch.where((z <= -1e5).all(dim=-1, keepdim=True), torch.zeros_like(z), z)
        return z

    def store_transition_train_mf_batch(self, service_states, task_states, prev_mfs, curr_mfs,
                                        actions, rewards, next_service_states, dones,
                                        agent_ids, log_probs, values, masks=None,
                                        proposal_logits=None, h_nodes=None, 
                                        workloads=None, ds_metrics=None, deadline_metrics=None,
                                        omegas=None, batch_sizes=None):
        general_tasks = self.tasks_to_general(task_states)
        loss_mf = self.learn_mf_batch(general_tasks, service_states, prev_mfs, curr_mfs, agent_ids)

        self.memory.add_batch(
            service_states=service_states, task_states=task_states,
            prev_mfs=prev_mfs, curr_mfs=curr_mfs, proposal_logits=proposal_logits,
            actions=actions, rewards=rewards, next_service_states=next_service_states,
            dones=dones, log_probs=log_probs, values=values, h_nodes=h_nodes,
            agent_ids=agent_ids, masks=masks,
            workloads=workloads, ds_metrics_list=ds_metrics, 
            deadline_metrics_list=deadline_metrics,
            omegas=omegas, batch_sizes=batch_sizes
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

    def _prepare_learn_params(self, phrase, step):
        current_ent_coef = self.initial_entropy_coef if phrase == "Proposal_Free" else self.update_coeff(step)
        if phrase == "Proposal_Only":
            self.entropy_coef = current_ent_coef
        return current_ent_coef

    def _get_buffer_data(self, agents_ids):
        if agents_ids is not None: agents_ids = agents_ids.to(self.device).view(-1)
        data = self.memory.get_all_ready(min_size=self.min_batch_size, agent_ids_pool=agents_ids)
        if data is None: return None

        (svc, t_cat, t_lens, p_log, act, a_lens, p_mf, c_mf, rew, n_svc, done, lp, val, h_n, mask_list, aid, wl, ds, dl, om, bs) = data
        
        # Pre-stack masks to avoid list attribution errors elsewhere
        if mask_list is not None and len(mask_list) > 0:
            first_m = next(m for m in mask_list if m is not None)
            masks_tensor = torch.stack([m if m is not None else torch.zeros_like(first_m) for m in mask_list], dim=0).to(self.device)
        else:
            masks_tensor = None

        return {
            'svc': svc.to(self.device).float(), 't_cat': t_cat.to(self.device).float(), 't_lens': t_lens.to(self.device).long(),
            'p_log': p_log.to(self.device).float(), 'act': act.to(self.device).long(), 'a_lens': a_lens.to(self.device).long(),
            'p_mf': p_mf.to(self.device).float(), 'c_mf': c_mf.to(self.device).float(), 'rew': rew.to(self.device).float().squeeze(-1),
            'n_svc': n_svc.to(self.device).float(), 'done': done.to(self.device).float().squeeze(-1), 'old_lp': lp.to(self.device).float().squeeze(-1),
            'old_val': val.to(self.device).float().squeeze(-1), 'h_node': h_n.to(self.device).float(), 'masks': masks_tensor, 'aids': aid.to(self.device).long(),
            'wl': wl, 'ds': ds, 'dl': dl, 'om': om, 'bs': bs
        }

    def _compute_adv_and_returns(self, b, general_tasks):
        from matrix_source.trainers.ppo_stategy import compute_gae
        B, T = b['svc'].shape[0], b['t_cat'].shape[0]
        group_idx = torch.repeat_interleave(torch.arange(B, device=self.device), b['t_lens'])
        
        with torch.no_grad():
            n_mf = self.mf_net(torch.cat([general_tasks, b['n_svc'], b['c_mf']], dim=-1), indices=b['aids'])
            n_mf_exp, n_svc_exp, n_aid_exp = n_mf[group_idx], b['n_svc'][group_idx], b['aids'][group_idx]
            
            n_prop = self.proposal(b['t_cat'], n_svc_exp, n_mf_exp, indices=n_aid_exp)
            n_probs = F.softmax(self._sanitize_logits(n_prop), dim=-1).view(T, self.M, self.max_models).sum(dim=2)
            n_h_node_g = torch.zeros(B, self.M, device=self.device).scatter_add_(0, group_idx.unsqueeze(1).expand(-1, self.M), n_probs)
            
            n_mask_node = b['masks'].view(B, self.M, self.max_models)[:, :, 0].to(self.device)
            
            n_val_g = self.critic(
                b['n_svc'], n_mf, n_h_node_g, b['wl'], n_mask_node, 
                b['ds'], b['dl'], b['om'], b['bs'], indices=b['aids']
            )
            adv = compute_gae(b['rew'], n_val_g, b['old_val'], b['done'], b['aids'], self.gamma, self.lmbda)
            ret = adv + b['old_val']
            if adv.numel() > 1: adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            
            curr_mf_det = self.mf_net(torch.cat([general_tasks, b['svc'], b['p_mf']], dim=-1), indices=b['aids']).detach()
            
        return adv, ret, curr_mf_det, group_idx, n_mask_node

    def _compute_hybrid_advantages(self, b, adv_data):
        """Tính toán Hybrid Advantage trực tiếp ở cấp độ Agent để tránh broadcasting lỗi."""
        adv, ret, curr_mf_det, group_idx, n_mask_node = adv_data
        B = b['svc'].shape[0]
        
        with torch.no_grad():
            mask_node = b['masks'].view(B, self.M, self.max_models)[:, :, 0].to(self.device)
            
            # 1. Tính giá trị Critic hiện tại cho toàn bộ các Agent trong batch (B rows)
            v_curr = self.critic(
                b['svc'], curr_mf_det, b['h_node'], b['wl'], mask_node, 
                b['ds'], b['dl'], b['om'], b['bs'], indices=b['aids']
            )
            
            # 2. Tính Hybrid Advantage trực tiếp: R - V(s)
            # Cả ret và v_curr đều là cấp độ Agent (B,) nên phép trừ an toàn
            hyb_adv_g = ret - v_curr.detach()
            
            # 3. Chuẩn hóa (Normalization) để ổn định training
            if hyb_adv_g.numel() > 1:
                hyb_adv_g = (hyb_adv_g - hyb_adv_g.mean()) / (hyb_adv_g.std() + 1e-8)
                
        return hyb_adv_g, curr_mf_det, group_idx

    def _run_optimization_epochs(self, b, adv_ret_data, phrase, ent_coef):
        B = b['svc'].shape[0]
        epoch_metrics = {k: 0.0 for k in ['v_loss', 'p_loss', 'r_loss', 'delta_norm', 'flip_rate', 'kl_div', 'refine_grad', 'q_imp']}
        total_batches = 0
        
        t_offsets = torch.zeros(B, dtype=torch.long, device=self.device)
        t_offsets[1:] = b['t_lens'].cumsum(0)[:-1]
        all_f_idx = torch.arange(b['t_cat'].shape[0], device=self.device)
        all_m = b['masks']  # Already processed as tensor in _get_buffer_data

        for _ in range(self.k_epochs):
            perm = torch.randperm(B, device=self.device)
            for s in range(0, B, self.batch_size):
                idx = perm[s:s + self.batch_size]
                metrics = self._optimize_minibatch(idx, b, adv_ret_data, all_m, t_offsets, all_f_idx, phrase, ent_coef)
                for k in epoch_metrics: epoch_metrics[k] += metrics[k]
                total_batches += 1
        
        for k in epoch_metrics: epoch_metrics[k] /= max(total_batches, 1)
        return epoch_metrics, total_batches

    def _optimize_minibatch(self, idx, b, adv_ret_data, all_m, t_offsets, all_f_idx, phrase, ent_coef):
        adv, ret, curr_mf_det, hybrid_adv_g = adv_ret_data
        B_sub = len(idx)
        
        # 1. Prepare batch data
        b_t_lens = b['t_lens'][idx]
        flat_idx = torch.cat([all_f_idx[t_offsets[i]:t_offsets[i] + b_t_lens[i]] for i in range(len(idx))])
        t_cat, act_cat, p_log_b = b['t_cat'][flat_idx], b['act'][flat_idx], b['p_log'][flat_idx]
        group_idx = torch.repeat_interleave(torch.arange(B_sub, device=self.device), b_t_lens)
        svc_e, mf_e, aid_e = b['svc'][idx][group_idx], curr_mf_det[idx][group_idx], b['aids'][idx][group_idx]
        masks_e = all_m[idx][group_idx] if all_m is not None else None

        # 2. Actor Losses
        prop_logits = self.proposal(t_cat, svc_e, mf_e, indices=aid_e)
        h_node_r, _ = self._compute_hist_and_overload(prop_logits.detach(), masks_e, b['svc'][idx], group_idx, B_sub, flat_idx.shape[0])
        
        # Refine delta
        def apply_m_s(z):
            if masks_e is not None: z = z.masked_fill(masks_e == 0, -1e9)
            if self.exclude_zero and self.u_action_dim > 1: z[:, 0] = -1e9
            return self._sanitize_logits(z)

        if phrase == "Proposal_Only": delta_logits = torch.zeros_like(prop_logits)
        else:
            # mf_load = buffered workload from this group (stored in b['wl'])
            wl_e = b['wl'][idx][group_idx]  # (total_tasks_sub, M)
            delta_logits = self.refine(t_cat, svc_e, mf_e, prop_logits.detach(), h_node_r[group_idx], wl_e, indices=aid_e)

        # Compute losses
        loss_p, loss_r, m = self._compute_actor_losses(
            prop_logits, delta_logits, act_cat, group_idx, B_sub, 
            adv[idx], hybrid_adv_g[idx], b['old_lp'][idx], 
            phrase, ent_coef, masks_e
        )
        
        # 3. Critic Loss (using expanded metrics from buffer)
        b_mask_node = all_m[idx].view(B_sub, self.M, self.max_models)[:, :, 0]
        v_grouped = self.critic(
            b['svc'][idx], curr_mf_det[idx], h_node_r, b['wl'][idx], b_mask_node, 
            b['ds'][idx], b['dl'][idx], b['om'][idx], b['bs'][idx], indices=b['aids'][idx]
        )
        loss_c = F.mse_loss(v_grouped, ret[idx])

        # Step optimizers
        self._step_optimizers(loss_p, loss_r, loss_c)
        
        m.update({
            'v_loss': loss_c.item(), 
            'refine_grad': 0.0,  # Simplified grad norm
            'q_imp': (hybrid_adv_g[idx] - adv[idx]).mean().item()
        })
        return m

    def _compute_actor_losses(self, prop_logits, delta_logits, act_cat, group_idx, B_sub, adv_gae, hyb_adv, old_joint_lp, phrase, ent_coef, masks_e):
        def apply_m_s(z):
            if masks_e is not None: z = z.masked_fill(masks_e == 0, -1e9)
            if self.exclude_zero and self.u_action_dim > 1: z[:, 0] = -1e9
            return self._sanitize_logits(z)

        if phrase == "Proposal_Only":
            logits = apply_m_s(prop_logits)
            dist = Categorical(logits=logits)
            curr_lp = torch.zeros(B_sub, device=self.device).scatter_add_(0, group_idx, dist.log_prob(act_cat))
            
            ratio = torch.exp(curr_lp - old_joint_lp)
            
            # Aggregate entropy per node to maintain consistent scale with policy gradient (B_sub level)
            ent_per_node = torch.zeros(B_sub, device=self.device).scatter_add_(0, group_idx, dist.entropy())
            loss_p = -torch.min(ratio * adv_gae, torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * adv_gae).mean() - ent_coef * ent_per_node.mean()
            loss_r = torch.tensor(0.0, device=self.device)
        else:
            # Proposal Update in Free stage (Dùng phân phối Hybrid hiện tại so với old_lp)
            final_logits_P = apply_m_s(prop_logits + self.residual_logit_scale * delta_logits.detach())
            dist_P = Categorical(logits=final_logits_P)
            curr_p_lp = torch.zeros(B_sub, device=self.device).scatter_add_(0, group_idx, dist_P.log_prob(act_cat))
            
            ratio_P = torch.exp(curr_p_lp - old_joint_lp)
            
            # Aggregate entropy per node to maintain consistent scale with policy gradient (B_sub level)
            ent_per_node_P = torch.zeros(B_sub, device=self.device).scatter_add_(0, group_idx, dist_P.entropy())
            loss_p = -torch.min(ratio_P * adv_gae, torch.clamp(ratio_P, 1-self.eps_clip, 1+self.eps_clip) * adv_gae).mean() - ent_coef * ent_per_node_P.mean()
            
            # Refine Update (Cũng dùng phân phối Hybrid hiện tại)
            final_logits_R = apply_m_s(prop_logits.detach() + self.residual_logit_scale * delta_logits)
            dist_R = Categorical(logits=final_logits_R)
            curr_r_lp = torch.zeros(B_sub, device=self.device).scatter_add_(0, group_idx, dist_R.log_prob(act_cat))
            
            ratio_R = torch.exp(curr_r_lp - old_joint_lp)
            loss_r = -torch.min(ratio_R * hyb_adv, torch.clamp(ratio_R, 1-self.eps_clip, 1+self.eps_clip) * hyb_adv).mean()

        # Tracking (Simplified)
        return loss_p, loss_r, {
            'p_loss': loss_p.item(), 
            'r_loss': loss_r.item(), 
            'delta_norm': delta_logits.norm().item(), 
            'flip_rate': 0.0, 
            'kl_div': 0.0
        }

    def _step_optimizers(self, loss_p, loss_r, loss_c):
        if isinstance(loss_p, torch.Tensor) and loss_p.requires_grad:
            self.optimizer_proposal.zero_grad(set_to_none=True)
            loss_p.backward()
            torch.nn.utils.clip_grad_norm_(self.proposal.parameters(), 0.5)
            self.optimizer_proposal.step()
        if isinstance(loss_r, torch.Tensor) and loss_r.requires_grad:
            self.optimizer_refine.zero_grad(set_to_none=True)
            loss_r.backward()
            torch.nn.utils.clip_grad_norm_(self.refine.parameters(), 0.5)
            self.optimizer_refine.step()
        self.optimizer_critic.zero_grad(set_to_none=True)
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer_critic.step()

    def learn(self, phrase: str, step: int, agents_ids=None):
        """
        Orchestrates the training process for Proposal, Refine, and Critic networks.
        """
        # 1. Initialization and Hyperparameter updates
        ent_coef = self._prepare_learn_params(phrase, step)
        
        # 2. Get Data from Memory
        batch_dict = self._get_buffer_data(agents_ids)
        if batch_dict is None: return None

        # 3. Pre-compute necessary metrics
        general_tasks = self.tasks_to_general(self._unpack_task_batch(batch_dict['t_cat'], batch_dict['t_lens'])).to(self.device)

        # 4. Compute Advantages and Returns
        adv, ret, curr_mf_det, group_idx, n_mask_node = self._compute_adv_and_returns(batch_dict, general_tasks)
        
        # 5. Compute Hybrid Advantages for Actor
        hybrid_adv_g, _, _ = self._compute_hybrid_advantages(batch_dict, (adv, ret, curr_mf_det, group_idx, n_mask_node))

        # 6. Run Optimization Epochs
        adv_ret_data = (adv, ret, curr_mf_det, hybrid_adv_g)
        epoch_metrics, total_batches = self._run_optimization_epochs(batch_dict, adv_ret_data, phrase, ent_coef)

        # Finalize
        epoch_metrics['hybrid_adv'] = hybrid_adv_g.mean().item()
        self.learn_step_counter += 1
        
        if self.learn_step_counter % 1 == 0 and total_batches > 0:
            print(f"[{self.node_type}][{phrase}] Step {self.learn_step_counter:5d} | V: {epoch_metrics['v_loss']:.5f} | P: {epoch_metrics['p_loss']:.5f} | R: {epoch_metrics['r_loss']:.5f}")

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