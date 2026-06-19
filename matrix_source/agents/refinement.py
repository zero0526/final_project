import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, Optional, Any, Tuple
from matrix_source.agents.buffer.refinement_buffer import MultiAgentRolloutBuffer
from matrix_source.agents.residual_net import ResidualCritic, RefineActor2, ProposalActor, MFNetwork
from matrix_source.agents.phrase import ProposalFreePhase, ProposalOnlyPhase, BasePhase
from matrix_source.trainers.ppo_stategy import compute_gae

class ResidualRoutingAgent:
    """
    Agent orchestrator - chỉ có 1 hàm learn() duy nhất
    Delegate logic cho phase hiện tại
    """

    def __init__(self,
                 model_workload: torch.Tensor,
                 agent_id: int,
                 node_type: str,
                 service_state_dim: int,
                 mf_dim: int,
                 proposal_dim: int,
                 action_dim: int,
                 u_action_dim: int,
                 mf_hidden_sizes: Tuple[int, ...],
                 mf_lr: float,
                 buffer_min_size: int,
                 hidden_sizes: Tuple[int, int] = (128, 64),
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 buffer_size: int = 100_000,
                 batch_size: int = 128,
                 lam: float = 0.95,
                 clip_eps: float = 0.2,
                 k_epochs: int = 5,
                 exclude_zero: bool = False,
                 num_instances: int = 1,
                 device: Optional[str] = None,
                 min_steps:int = 300,
                 reward_stable_threshold:float = 0.1,
                 reward_stable_window:int = 10,
                 p_entropy_coef_start:float = 0.05,
                 p_entropy_coef_end:float = 0.001,
                 p_entropy_decay_rate:float = 0.995,
                 alpha:float = 1.0,  # proposal (frozen)
                 beta:float = 1.0,  # refine
                 entropy_coef_start:float = 0.05,
                 entropy_coef_end:float =0.001,  #
                 entropy_decay_rate:float = 0.995,  # Decay
                 initial_phase: str = "ProposalOnly"):

        # ══════════════════════════════════════════
        # 1. BASIC SETUP
        # ══════════════════════════════════════════
        self.model_workload= model_workload
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

        self.M = service_state_dim // 2
        self.max_models = u_action_dim // self.M

        self.gamma = gamma
        self.lmbda = lam
        self.eps_clip = clip_eps
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.min_batch_size= buffer_min_size
        # ══════════════════════════════════════════
        # 2. NETWORKS
        # ══════════════════════════════════════════
        TASK_DIM = 4
        GENERAL_TASK_DIM = 10

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

        self.refine = RefineActor2(
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
            hidden_sizes=hidden_sizes,
            num_instances=num_instances,
        ).to(self.device)

        # ══════════════════════════════════════════
        # 3. OPTIMIZERS (default LR, phase sẽ override)
        # ══════════════════════════════════════════
        self.optimizer_proposal = optim.Adam(self.proposal.parameters(), lr=lr)
        self.optimizer_refine = optim.Adam(self.refine.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.mf_optimizer = optim.Adam(self.mf_net.parameters(), lr=mf_lr)
        self.loss_fn = nn.SmoothL1Loss()

        # ══════════════════════════════════════════
        # 4. BUFFER
        # ══════════════════════════════════════════
        self.memory = MultiAgentRolloutBuffer(
            num_agents=num_instances,
            node_type=node_type,
            max_size_per_agent=buffer_size,
            service_state_dim=service_state_dim,
            action_dim=action_dim,
            device=self.device,
        )

        # 5. PHASE REGISTRY
        self.phases: Dict[str, BasePhase] = {
            "ProposalOnly": ProposalOnlyPhase({
                "min_steps": min_steps,
                "reward_stable_threshold": reward_stable_threshold,
                "reward_stable_window": reward_stable_window,
                "entropy_coef_start": p_entropy_coef_start,
                "entropy_coef_end": p_entropy_coef_end,
                "entropy_decay_rate": p_entropy_decay_rate
            }),
            "ProposalFree": ProposalFreePhase({
                'alpha': alpha,  # proposal (frozen)
                'beta': beta,  # refine
                'entropy_coef_start': entropy_coef_start,
                'entropy_coef_end':entropy_coef_end,  #
                'entropy_decay_rate': entropy_decay_rate,  # Decay
            }),
        }

        self.phase_order = ["ProposalOnly", "ProposalFree"]

        # Current phase
        assert initial_phase in self.phases, \
            f"Unknown phase: {initial_phase}. Available: {list(self.phases.keys())}"
        self.current_phase_name = initial_phase
        self.current_phase = self.phases[initial_phase]

        # Auto-transition flag
        self.auto_transition_enabled = True

        # 6. TRAINING STATE
        self.learn_step_counter = 0
        self.proposal_load_var: float = 0.0

        # Apply initial phase parameters
        self._apply_phase_parameters()

    # PHASE MANAGEMENT
    def set_phase(self, phase_name: str):
        """Force phase"""
        if phase_name not in self.phases:
            raise ValueError(
                f"Unknown phase: {phase_name}. "
                f"Available: {list(self.phases.keys())}"
            )

        if phase_name == self.current_phase_name:
            return  # Already in this phase

        # Exit current phase
        self.current_phase.on_exit(self)

        # Enter new phase
        self.current_phase_name = phase_name
        self.current_phase = self.phases[phase_name]
        self.current_phase.on_enter(self)

        # Apply phase hyperparameters
        self._apply_phase_parameters()

    def _apply_phase_parameters(self):
        """Áp dụng hyperparameters từ phase hiện tại (LR, freeze, etc.)"""
        params = self.current_phase.get_parameters()

        # Update learning rates
        if params.lr_proposal is not None:
            for pg in self.optimizer_proposal.param_groups:
                pg['lr'] = params.lr_proposal

        if params.lr_refine is not None:
            for pg in self.optimizer_refine.param_groups:
                pg['lr'] = params.lr_refine

        # Freeze/unfreeze networks based on phase
        for p in self.proposal.parameters():
            p.requires_grad = params.train_proposal and not params.freeze_proposal

        for p in self.refine.parameters():
            p.requires_grad = params.train_refine and not params.freeze_refine

        # Set train/eval mode
        self.proposal.train(params.train_proposal and not params.freeze_proposal)
        self.refine.train(params.train_refine and not params.freeze_refine)

    def _try_auto_transition(self, metrics: Dict[str, float]):
        """Thử tự động chuyển phase nếu đủ điều kiện"""
        if not self.auto_transition_enabled:
            return

        next_phase_name = self.current_phase.should_transition(metrics)
        if next_phase_name is not None:
            self.set_phase(next_phase_name)

    def get_phase_info(self) -> Dict[str, Any]:
        """Lấy thông tin phase hiện tại"""
        return {
            'current_phase': self.current_phase_name,
            'steps_in_phase': self.current_phase.step_counter,
            'phase_hp': self.current_phase.hp.copy(),
            'auto_transition': self.auto_transition_enabled,
        }

    def choose_action(self, state, prev_mf, mask=None, agent_idx=0,
                      task_state=None, task_batch_size=None, service_idx=0, deterministic=False):
        """Wrapper cho single agent"""
        idx_t = torch.tensor([agent_idx], device=self.device)
        svc_idx_t = torch.tensor([service_idx], device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if prev_mf.dim() == 1:
            prev_mf = prev_mf.unsqueeze(0)
        if not isinstance(task_state, list):
            task_state = [task_state]
        if not isinstance(task_batch_size, list) and task_batch_size is not None:
            task_batch_size = [task_batch_size]
        if mask is not None and not isinstance(mask, list):
            mask = [mask]

        actions, log_probs, values, _ = self.choose_action_batch(
            service_states=state,
            prev_mfs=prev_mf,
            task_states=task_state,
            task_batch_sizes=task_batch_size,
            service_indices=svc_idx_t,
            masks_batch=mask,
            agent_indices=idx_t,
            deterministic=deterministic,
        )
        return actions[0], log_probs[0], values[0]

    def choose_action_batch(self, service_states, prev_mfs, task_states,
                            task_batch_sizes=None, service_indices=None, masks_batch=None, 
                            agent_indices=None, deterministic=False):
        """
        Batch inference - dùng alpha/beta từ phase hiện tại để fusion

        Returns:
            all_actions, all_log_probs, all_values, h_node
        """
        B = service_states.shape[0]
        device = self.device

        # Get phase parameters (alpha, beta)
        phase_params = self.current_phase.get_parameters()
        alpha = phase_params.alpha
        beta = phase_params.beta

        if agent_indices is None:
            agent_indices = torch.zeros(B, dtype=torch.long, device=device)
        else:
            agent_indices = agent_indices.to(device).view(-1)
        
        service_indices = service_indices.to(device).view(-1)

        service_states = service_states.to(device).float()
        prev_mfs = prev_mfs.to(device).float()
        general_task = self.tasks_to_general(task_states)

        with torch.no_grad():
            # 1. Mean Field Prediction
            pred_mfs = self.mf_net(
                torch.cat([general_task, service_states, prev_mfs], dim=-1),
                indices=agent_indices
            )

            # 2. Flatten & Expand
            task_lens = torch.tensor([t.shape[0] for t in task_states], device=device)
            total_tasks = int(task_lens.sum().item())
            batch_idx = torch.repeat_interleave(
                torch.arange(B, device=device), task_lens
            )

            tasks_cat = torch.cat(task_states, dim=0).to(device).float()
            
            # Handle task batch sizes
            if task_batch_sizes is not None:
                bs_cat = torch.cat(task_batch_sizes, dim=0).to(device).float()
            else:
                bs_cat = torch.ones(total_tasks, device=device)

            svc_exp = service_states[batch_idx]
            mf_exp = pred_mfs[batch_idx]
            idx_exp = agent_indices[batch_idx]
            s_idx_exp = service_indices[batch_idx]
            
            # Workload for each task (total_n, max_models)
            workload_cat = self.model_workload[s_idx_exp].to(device)

            # 3. Mask Handling
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

            # 5. Compute histogram & overload (input cho Refine)
            h_node = self._compute_hist(
                prop_logits.detach(), masks_exp, service_states,
                batch_idx, B, total_tasks, bs_cat=bs_cat, workload_cat=workload_cat
            )
            self.proposal_load_var = h_node.var(dim=1).mean().item()

            if beta > 0:# 6. Refinement Forward (chỉ khi beta > 0)
                delta_logits = self.refine(
                    tasks_cat, svc_exp, mf_exp,
                    prop_logits.detach(),
                    h_node[batch_idx],
                    indices=idx_exp
                )
            else:# Phase 1
                delta_logits = torch.zeros_like(prop_logits)

            # 7. Fusion: final = alpha * prop + beta * delta
            final_logits = alpha * prop_logits + beta * delta_logits

            # 8. Mask & Sanitize
            final_logits = self.mask_and_sanitize(final_logits, masks_exp)

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

            sum_lp = torch.zeros(B, device=device).scatter_add_(
                0, batch_idx, log_probs_cat
            )
            all_log_probs = list(sum_lp.unbind())

            # 11. Critic (KHÔNG nhận h_node)
            all_values = self.critic(
                general_task, service_states, pred_mfs, indices=agent_indices
            )

        return all_actions, all_log_probs, all_values, h_node

    # ② STORE TRANSITION + TRAIN MF
    def store_transition_train_mf_batch(self, service_states, task_states,
                                        prev_mfs, curr_mfs,
                                        actions, rewards, next_service_states,
                                        dones, agent_ids,
                                        log_probs, values, service_indices=None, 
                                        masks=None, task_batch_sizes=None):
        """
        Store transitions to buffer AND train mean-field network.

        Returns:
            loss_mf: Mean-field prediction loss
        """
        general_tasks = self.tasks_to_general(task_states)
        loss_mf = self.learn_mf_batch(
            general_tasks, service_states, prev_mfs, curr_mfs, agent_ids
        )

        self.memory.add_batch(
            service_states, task_states, prev_mfs, curr_mfs,
            actions, rewards, next_service_states, dones,
            log_probs, values, agent_ids, service_indices=service_indices, 
            masks=masks, task_batch_sizes=task_batch_sizes
        )

        return loss_mf

    def learn_mf_batch(self, general_tasks, service_states, prev_mfs,
                       ground_truth_mfs, agent_ids):
        """Train mean-field network to predict next-step mean field"""
        gt = torch.as_tensor(general_tasks, dtype=torch.float32, device=self.device)
        s = torch.as_tensor(service_states, dtype=torch.float32, device=self.device)
        pm = torch.as_tensor(prev_mfs, dtype=torch.float32, device=self.device)
        gf = torch.as_tensor(ground_truth_mfs, dtype=torch.float32, device=self.device)

        pred_mf = self.mf_net(
            torch.cat([gt, s, pm], dim=-1), indices=agent_ids
        )
        loss = self.loss_fn(pred_mf, gf)

        self.mf_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mf_net.parameters(), max_norm=5.0)
        self.mf_optimizer.step()

        return loss.item()

    # ══════════════════════════════════════════════════════════
    # ③ PPO LEARN (Single function - delegate to phase)
    # ══════════════════════════════════════════════════════════

    def learn(self, step: int, agents_ids=None, force_phase: str = None):
        """
        Single learning function - delegates to current phase.

        Args:
            step: Current training step
            agents_ids: Agent IDs to train
            force_phase: Force specific phase (optional)

        Returns:
            Average value loss, or None if no data
        """
        # 1. FORCE PHASE (nếu có)
        if force_phase is not None and force_phase != self.current_phase_name:
            self.set_phase(force_phase)

        # ══════════════════════════════════════════
        # 2. GET PARAMETERS FROM PHASE
        # ══════════════════════════════════════════
        phase_params = self.current_phase.get_parameters()

        # 3. LOAD DATA
        if agents_ids is not None:
            agents_ids = agents_ids.to(self.device).view(-1)

        data = self.memory.get_all_ready(
            min_size=self.min_batch_size,
            agent_ids_pool=agents_ids
        )
        if data is None:
            return None

        # Unpack data
        (service_states, task_batch_cat, task_lens, actions_cat, action_lens,
         prev_mfs, curr_mfs, rewards, next_service_states, dones,
         old_log_probs, old_values, masks, agent_ids, 
         batch_sizes_cat, service_indices_cat) = data

        # Move to device
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

        # 4. PRE-COMPUTE (No Grad)
        with torch.no_grad():
            # Critic bootstrap
            next_mf = self.mf_net(
                torch.cat([general_tasks, next_service_states,
                           curr_mfs.to(self.device)], dim=-1),
                indices=agent_ids
            )
            next_val = self.critic(
                general_tasks, next_service_states, next_mf, indices=agent_ids
            )

            # GAE
            advantages = compute_gae(
                rewards, next_val, old_values, dones, agent_ids,
                self.gamma, self.lmbda
            )
            returns = advantages + old_values
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # MF for entire dataset
            detached_mfs_all = self.mf_net(
                torch.cat([general_tasks, service_states, prev_mfs], dim=-1),
                indices=agent_ids
            ).detach()

        # Pre-process masks
        if masks is not None and any(m is not None for m in masks):
            first_valid = next(m for m in masks if m is not None)
            if first_valid.dim() == 1:
                clean_masks = [
                    m if m is not None else torch.zeros_like(first_valid)
                    for m in masks
                ]
                all_masks = torch.stack(clean_masks).to(self.device)
            else:
                all_masks = torch.cat(
                    [m for m in masks if m is not None], dim=0
                ).to(self.device)
        else:
            all_masks = None

        # Pre-compute indices
        task_offsets = torch.zeros(dataset_size, dtype=torch.long, device=self.device)
        task_offsets[1:] = task_lens.cumsum(0)[:-1]
        all_flat_idx = torch.arange(total_tasks_flat, device=self.device)

        # ══════════════════════════════════════════
        # 5. TRAINING LOOP
        # ══════════════════════════════════════════
        epoch_metrics = {'v': 0.0, 'p': 0.0, 'r': 0.0}
        total_batches = 0

        for _ in range(self.k_epochs):
            perm = torch.randperm(dataset_size, device=self.device)

            for start in range(0, dataset_size, self.batch_size):
                idx = perm[start:start + self.batch_size]

                # Prepare batch data
                batch_data = self._prepare_batch(
                    idx, service_states, prev_mfs, old_log_probs,
                    advantages, returns, agent_ids, general_tasks,
                    detached_mfs_all, task_lens, task_offsets, all_flat_idx,
                    task_batch_cat, actions_cat, all_masks, 
                    batch_sizes_cat, service_indices_cat
                )

                # ── FORWARD PASS ──
                prop_logits = self.proposal(
                    batch_data['t_cat'], batch_data['svc_exp'],
                    batch_data['mf_exp'], indices=batch_data['aids_exp']
                )

                # Compute histogram & overload (input cho Refine)
                h_node = self._compute_hist(
                    prop_logits.detach(), batch_data['masks_exp'],
                    batch_data['b_svc'], batch_data['batch_idx'],
                    batch_data['B_sub'], batch_data['total_n'],
                    bs_cat=batch_data['bs_cat'], 
                    workload_cat=batch_data['workload_cat']
                )

                # Refine forward (chỉ khi beta > 0)
                if phase_params.beta > 0 and phase_params.train_refine:
                    delta_logits = self.refine(
                        batch_data['t_cat'], batch_data['svc_exp'],
                        batch_data['mf_exp'],
                        prop_logits.detach(),
                        h_node[batch_data['batch_idx']],
                        indices=batch_data['aids_exp']
                    )
                else:
                    delta_logits = torch.zeros_like(prop_logits)

                # Add to batch_data for phase
                batch_data['prop_logits'] = prop_logits
                batch_data['delta_logits'] = delta_logits
                batch_data['h_node'] = h_node

                # ── DELEGATE TO PHASE ──
                losses = self.current_phase.compute_losses(self, batch_data)

                loss_proposal = losses['loss_proposal']
                loss_refine = losses['loss_refine']

                # ── CRITIC UPDATE ──
                c_vals = self.critic(
                    batch_data['b_gen'], batch_data['b_svc'],
                    batch_data['b_mf'], indices=batch_data['b_aids']
                )
                c_loss = F.mse_loss(c_vals, batch_data['returns'])

                # ── BACKPROP (theo phase parameters) ──
                if phase_params.train_proposal and loss_proposal is not None:
                    self.optimizer_proposal.zero_grad(set_to_none=True)
                    loss_proposal.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.proposal.parameters(),
                        phase_params.grad_clip_proposal
                    )
                    self.optimizer_proposal.step()

                if phase_params.train_refine and loss_refine is not None:
                    self.optimizer_refine.zero_grad(set_to_none=True)
                    loss_refine.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.refine.parameters(),
                        phase_params.grad_clip_refine
                    )
                    self.optimizer_refine.step()

                # Critic update
                self.optimizer_critic.zero_grad(set_to_none=True)
                c_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

                # Metrics
                epoch_metrics['v'] += c_loss.item()
                epoch_metrics['p'] += (
                    loss_proposal.item() if loss_proposal is not None else 0.0
                )
                epoch_metrics['r'] += (
                    loss_refine.item() if loss_refine is not None else 0.0
                )
                total_batches += 1

        # ══════════════════════════════════════════
        # 6. POST-LEARNING
        # ══════════════════════════════════════════
        self.learn_step_counter += 1
        self.current_phase.step()

        # Auto-transition check
        avg_metrics = {
            k: v / max(total_batches, 1)
            for k, v in epoch_metrics.items()
        }
        self._try_auto_transition(avg_metrics)

        # Logging
        if self.learn_step_counter % 10 == 0 and total_batches > 0:
            n = total_batches
            print(
                f"[{self.node_type}][{self.current_phase_name}] "
                f"Step {self.learn_step_counter:5d} | "
                f"V: {epoch_metrics['v'] / n:.5f} | "
                f"P: {epoch_metrics['p'] / n:.5f} | "
                f"R: {epoch_metrics['r'] / n:.5f}"
            )

        self.memory.clear()
        return epoch_metrics['v'] / max(total_batches, 1)


    # ④ CHECKPOINT (Save Agent + All Phases)
    def save(self, path: str):
        """Lưu đầy đủ state của agent VÀ tất cả phases"""
        torch.save({
            # Networks
            'proposal': self.proposal.state_dict(),
            'refine': self.refine.state_dict(),
            'critic': self.critic.state_dict(),
            'mf_net': self.mf_net.state_dict(),

            # Optimizers
            'optimizer_proposal': self.optimizer_proposal.state_dict(),
            'optimizer_refine': self.optimizer_refine.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict(),
            'mf_optimizer': self.mf_optimizer.state_dict(),

            # Phase states (QUAN TRỌNG)
            'current_phase_name': self.current_phase_name,
            'phases_state': {
                name: phase.save_state()
                for name, phase in self.phases.items()
            },

            # Agent state
            'learn_step_counter': self.learn_step_counter,
            'auto_transition_enabled': self.auto_transition_enabled,
        }, path)

    def load(self, path: str):
        """Load state của agent VÀ restore phase states"""
        ckpt = torch.load(path, map_location=self.device)

        # Networks
        self.proposal.load_state_dict(ckpt['proposal'])
        self.refine.load_state_dict(ckpt['refine'])
        self.critic.load_state_dict(ckpt['critic'])
        self.mf_net.load_state_dict(ckpt['mf_net'])

        # Optimizers
        self.optimizer_proposal.load_state_dict(ckpt['optimizer_proposal'])
        self.optimizer_refine.load_state_dict(ckpt['optimizer_refine'])
        self.optimizer_critic.load_state_dict(ckpt['optimizer_critic'])
        self.mf_optimizer.load_state_dict(ckpt['mf_optimizer'])

        # Phase states (restore hyperparameters)
        for name, phase_state in ckpt['phases_state'].items():
            self.phases[name].load_state(phase_state)

        # Set current phase
        self.current_phase_name = ckpt['current_phase_name']
        self.current_phase = self.phases[self.current_phase_name]

        # Agent state
        self.learn_step_counter = ckpt['learn_step_counter']
        self.auto_transition_enabled = ckpt['auto_transition_enabled']

        # Apply current phase parameters
        self._apply_phase_parameters()

        print(f"✅ Loaded checkpoint at step {self.learn_step_counter}")
        print(f"   Current phase: {self.current_phase_name}")
        print(f"   Phase HPs: {self.current_phase.hp}")

    # HELPer (Orchestrator)
    def _prepare_batch(self, idx, service_states, prev_mfs, old_log_probs,
                       advantages, returns, agent_ids, general_tasks,
                       detached_mfs_all, task_lens, task_offsets, all_flat_idx,
                       task_batch_cat, actions_cat, all_masks, 
                       batch_sizes_cat, service_indices_cat):
        """Chuẩn bị data cho một mini-batch"""
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
        segments = [
            all_flat_idx[task_offsets[i]:task_offsets[i] + task_lens[i]]
            for i in idx
        ]
        flat_indices = torch.cat(segments)

        t_cat = task_batch_cat[flat_indices]
        act_cat = actions_cat[flat_indices]
        total_n = t_cat.shape[0]

        # Task → agent mapping
        batch_idx = torch.repeat_interleave(
            torch.arange(B_sub, device=self.device), b_t_lens
        )

        # Expand per-agent data to per-task
        svc_exp = b_svc[batch_idx]
        mf_exp = b_mf[batch_idx]
        aids_exp = b_aids[batch_idx]
        masks_exp = all_masks[idx][batch_idx] if all_masks is not None else None
        bs_cat = batch_sizes_cat[flat_indices]
        svc_idx_exp = service_indices_cat[idx][batch_idx]
        workload_cat = self.model_workload[svc_idx_exp].to(self.device)

        return {
            'B_sub': B_sub,
            'total_n': total_n,
            'b_svc': b_svc,
            'b_old_lp': b_old_lp,
            'b_adv': b_adv,
            'b_ret': b_ret,
            'b_aids': b_aids,
            'b_gen': b_gen,
            'b_mf': b_mf,
            'b_t_lens': b_t_lens,
            't_cat': t_cat,
            'act_cat': act_cat,
            'batch_idx': batch_idx,
            'svc_exp': svc_exp,
            'mf_exp': mf_exp,
            'aids_exp': aids_exp,
            'masks_exp': masks_exp,
            'bs_cat': bs_cat,
            'workload_cat': workload_cat,
            'returns': b_ret,  # For critic
        }

    def tasks_to_general(self, task_states):
        """Convert list of task tensors to general task representation"""
        if isinstance(task_states, (list, tuple)):
            return torch.stack([self._general_single(t) for t in task_states])
        return self._general_single(task_states)

    def _general_single(self, tasks):
        """Compute general features for a single agent's tasks"""
        if tasks.shape[0] == 0:
            return torch.zeros(10, device=self.device)
        t = tasks.float()
        mean = t.mean(dim=0)
        std = t.std(dim=0, correction=0) if t.shape[0] > 1 else torch.zeros_like(mean)
        
        # Tứ phân vị cho chiều thời gian (index 1)
        time_dim = t[:, 1]
        q = torch.quantile(time_dim, torch.tensor([0.25, 0.5, 0.75], device=self.device))
        
        return torch.tensor([
            float(t.shape[0]), mean[0], mean[1], std[1], 
            mean[2], std[2], q[0], q[1], q[2], mean[3]
        ], dtype=torch.float32, device=self.device)

    def _unpack_task_batch(self, task_batch_cat, task_lens):
        """Unpack flattened task batch into list of tensors"""
        task_states, offset = [], 0
        for n_i in task_lens:
            n_i = int(n_i.item())
            task_states.append(task_batch_cat[offset:offset + n_i])
            offset += n_i
        return task_states

    @staticmethod
    def _sanitize_logits(z):
        """Replace NaN/Inf with safe values"""
        z = torch.where(torch.isnan(z), torch.tensor(-1e6, device=z.device), z)
        z = torch.where(torch.isinf(z), torch.tensor(-1e6, device=z.device), z)
        z = torch.where(
            (z <= -1e5).all(dim=-1, keepdim=True),
            torch.zeros_like(z), z
        )
        return z

    def mask_and_sanitize(self, z, masks_exp):
        """Apply mask and sanitize logits"""
        if masks_exp is not None:
            z = z.masked_fill(masks_exp == 0, -1e9)
        if self.exclude_zero and self.u_action_dim > 1:
            z = z.clone()
            z[:, 0] = -1e9
        return self._sanitize_logits(z)

    def _compute_hist(self, logits, masks_exp, svc_batch,
                                   batch_idx, B_batch, total_n, bs_cat=None, workload_cat=None):
        """
        Compute histogram (h_node)
        """
        logits_for_hist = logits.detach()
        if masks_exp is not None:
            logits_for_hist = logits_for_hist.masked_fill(masks_exp == 0, -1e9)

        probs = F.softmax(self._sanitize_logits(logits_for_hist), dim=-1)
        
        # workload_cat is (total_n, max_models)
        # probs is (total_n, M * max_models)
        # We need (total_n, M) weighted workload
        probs_reshaped = probs.view(total_n, self.M, self.max_models)
        
        probs_M = (probs_reshaped * workload_cat.unsqueeze(1)).sum(dim=2)

        # Apply weighting by task batch size
        probs_M = probs_M * bs_cat.view(-1, 1)

        h_node = torch.zeros(B_batch, self.M, device=self.device)
        h_node.scatter_add_(
            0, batch_idx.unsqueeze(1).expand(-1, self.M), probs_M
        )
        return h_node/500.0