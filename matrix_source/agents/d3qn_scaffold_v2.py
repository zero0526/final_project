import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matrix_source.agents.buffer.ReplayBuffer import MultiAgentReplayBuffer
from matrix_source.agents.buffer.PrioritizedReplayBuffer import MultiAgentPrioritizedReplayBuffer
import random

# ─────────────────────────────────────────────────────────────────────────────
# Shared Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

class MultiInstanceLinear(nn.Module):
    """Parallel linear layer for N independent agent models."""
    def __init__(self, num_instances, in_features, out_features, bias=True):
        super().__init__()
        self.num_instances = num_instances
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(num_instances, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_instances, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_instances):
            nn.init.orthogonal_(self.weight[i], gain=1.0)
            if self.bias is not None:
                nn.init.zeros_(self.bias[i])

    def forward(self, x, indices=None):
        if indices is None:
            indices = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        w = self.weight[indices]
        out = torch.bmm(x.unsqueeze(1), w).squeeze(1)
        if self.bias is not None:
            out += self.bias[indices]
        return out


class MultiInstanceRMSNorm(nn.Module):
    """Instance-specific RMS Layer Normalization."""
    def __init__(self, num_instances, normalized_shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_instances, normalized_shape))

    def forward(self, x, indices):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight[indices]


# ─────────────────────────────────────────────────────────────────────────────
# Split Backbone / Head Network
# ─────────────────────────────────────────────────────────────────────────────

class DuelingBackbone(nn.Module):
    """
    Shared representation network (backbone).
    All terminals in a cluster will sync these weights via SCAFFOLD.
    Shape: (num_instances, state+mf) -> (num_instances, h2)
    """
    def __init__(self, state_dim, mf_dim, hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        self.num_instances = num_instances
        self.l1 = MultiInstanceLinear(num_instances, state_dim + mf_dim, h1)
        self.norm1 = MultiInstanceRMSNorm(num_instances, h1)
        self.l2 = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2 = MultiInstanceRMSNorm(num_instances, h2)

    def forward(self, state, pred_mf, indices=None):
        x = torch.cat([state, pred_mf], dim=-1)
        x = F.silu(self.norm1(self.l1(x, indices), indices))
        x = F.silu(self.norm2(self.l2(x, indices), indices))
        return x

    def get_params(self):
        return list(self.parameters())


class DuelingHead(nn.Module):
    """
    Personalized dueling head for each agent (never aggregated).
    Takes backbone output -> Value and Advantage streams.
    """
    def __init__(self, h2, action_dim, num_instances=1):
        super().__init__()
        self.num_instances = num_instances
        # Value stream
        self.v1 = MultiInstanceLinear(num_instances, h2, h2)
        self.v2 = MultiInstanceLinear(num_instances, h2, 1)
        # Advantage stream
        self.a1 = MultiInstanceLinear(num_instances, h2, h2)
        self.a2 = MultiInstanceLinear(num_instances, h2, action_dim)

    def forward(self, x, indices=None):
        V = self.v2(F.silu(self.v1(x, indices)), indices)
        A = self.a2(F.silu(self.a1(x, indices)), indices)
        return V + (A - A.mean(dim=-1, keepdim=True))

    def get_params(self):
        return list(self.parameters())


class DuelingSplitNet(nn.Module):
    """Combined backbone + head. Target net uses this class too."""
    def __init__(self, state_dim, mf_dim, action_dim, hidden_sizes, num_instances=1):
        super().__init__()
        _, h2 = hidden_sizes
        self.backbone = DuelingBackbone(state_dim, mf_dim, hidden_sizes, num_instances)
        self.head = DuelingHead(h2, action_dim, num_instances)

    def forward(self, state, pred_mf, indices=None):
        feat = self.backbone(state, pred_mf, indices)
        return self.head(feat, indices)


class MF(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, num_instances=1):
        super().__init__()
        h = hidden_sizes[0]
        self.l1 = MultiInstanceLinear(num_instances, input_size, h)
        self.norm = MultiInstanceRMSNorm(num_instances, h)
        self.l2 = MultiInstanceLinear(num_instances, h, output_size)

    def forward(self, x, indices=None):
        x = F.silu(self.norm(self.l1(x, indices), indices))
        return torch.sigmoid(self.l2(x, indices))


# ─────────────────────────────────────────────────────────────────────────────
# D3QN Agent V2 — Split Backbone/Head Phased-SCAFFOLD
# ─────────────────────────────────────────────────────────────────────────────

class D3QNAgentV2:
    """
    Multi-instance D3QN with split backbone/head SCAFFOLD.

    Training phases (controlled externally by passing `round_idx` to learn()):
      Phase 1  [0, T1):   backbone lr=lr_high,  lambda_scaffold=1.0
      Phase 2  [T1, T2):  backbone lr=lr_low,   lambda_scaffold=1.0
      Phase 3  [T2, T):   backbone frozen,       lambda_scaffold decays 1→0
    """

    def __init__(
        self,
        node_id, node_type,
        state_dim, action_dim, u_action_dim,
        mf_hidden_sizes, mf_lr, buffer_min_size,
        hidden_sizes=(128, 64),
        lr=1e-4, lr_bone_low=5e-5,
        gamma=0.99, alpha=0.005,
        buffer_size=100000, batch_size=64,
        exclude_zero=False, num_instances=1,
        device=None, use_per=False, n_step=1,
        logs_q=False, use_scaffold=False,
        # Phase schedule (in units of federated rounds)
        total_rounds=100, phase1_frac=0.45, phase2_frac=0.65,
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.action_dim = action_dim
        self.u_action_dim = u_action_dim
        self.gamma = gamma
        self.alpha = float(alpha)
        self.batch_size = batch_size
        self.exclude_zero = exclude_zero
        self.logs_q = logs_q
        self.num_instances = num_instances
        self.min_batch_size = buffer_min_size
        self.use_scaffold = use_scaffold

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # ── Networks ──────────────────────────────────────────────────────────
        self.eval_net = DuelingSplitNet(state_dim, action_dim, u_action_dim, hidden_sizes, num_instances).to(self.device)
        self.target_net = DuelingSplitNet(state_dim, action_dim, u_action_dim, hidden_sizes, num_instances).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.mf_net = MF(state_dim + action_dim, action_dim, mf_hidden_sizes, num_instances).to(self.device)

        # ── Separate optimizers for backbone and head ─────────────────────────
        self.lr_bone_high = lr
        self.lr_bone_low  = lr_bone_low
        self.lr_head      = lr

        self.bone_optimizer = optim.Adam(self.eval_net.backbone.parameters(), lr=lr)
        self.head_optimizer  = optim.Adam(self.eval_net.head.parameters(), lr=lr)
        self.mf_optimizer    = optim.Adam(self.mf_net.parameters(), lr=mf_lr)

        self.loss_fn = nn.SmoothL1Loss(reduction='none')

        # ── Replay Buffer ─────────────────────────────────────────────────────
        self.memory = MultiAgentReplayBuffer(
            num_instances, node_type, buffer_size,
            state_dim, action_dim, u_action_dim, self.device
        )

        # ── Phase schedule ────────────────────────────────────────────────────
        self.total_rounds = total_rounds
        self.T1 = int(total_rounds * phase1_frac)
        self.T2 = int(total_rounds * phase2_frac)

        # ── SCAFFOLD Control Variates ─────────────────────────────────────────
        if self.use_scaffold:
            bone_params = list(self.eval_net.backbone.parameters())
            head_params = list(self.eval_net.head.parameters())

            # Per-param, per-instance tensors
            self.c_b_local  = [torch.zeros_like(p) for p in bone_params]
            self.c_b_global = [torch.zeros_like(p) for p in bone_params]

            # Raw gradient accumulators (for multi-step c update)
            self.grad_b_sum = [torch.zeros_like(p) for p in bone_params]
            self.steps_in_round = torch.zeros(num_instances, dtype=torch.long, device=self.device)

        # ── Misc ──────────────────────────────────────────────────────────────
        self.learn_step_counter = 0

    # ── Phase helpers ─────────────────────────────────────────────────────────

    def _get_phase(self, round_idx):
        if round_idx < self.T1:
            return 1
        elif round_idx < self.T2:
            return 2
        else:
            return 3

    def _get_lambda(self, round_idx):
        """No longer used as Head SCAFFOLD is disabled."""
        return 0.0

    def _set_backbone_lr(self, phase):
        lr = self.lr_bone_high if phase == 1 else self.lr_bone_low
        for pg in self.bone_optimizer.param_groups:
            pg['lr'] = lr

    def _freeze_backbone(self, freeze: bool):
        for p in self.eval_net.backbone.parameters():
            p.requires_grad = not freeze

    # ── SCAFFOLD round lifecycle ───────────────────────────────────────────────

    def save_base_initial(self):
        """Reset accumulators at the start of each federated round."""
        if not self.use_scaffold:
            return
        with torch.no_grad():
            for g in self.grad_b_sum:
                g.zero_()
            self.steps_in_round.zero_()

    # ── Action selection (unchanged from v1) ──────────────────────────────────

    def choose_action(self, state, prev_mf, epsilon, zeta, mask=None, agent_idx=0, deterministic=False):
        idx_tensor = torch.tensor([agent_idx], device=self.device)
        actions = self.choose_action_batch(
            state.unsqueeze(0) if not torch.is_tensor(state) else state.detach().unsqueeze(0),
            prev_mf.unsqueeze(0) if not torch.is_tensor(prev_mf) else prev_mf.detach().unsqueeze(0),
            epsilon, zeta,
            masks_batch=mask.unsqueeze(0) if mask is not None else None,
            agent_indices=idx_tensor,
            deterministic=deterministic
        )
        return int(actions[0])

    def choose_action_batch(self, states_batch, prev_mfs_batch, epsilon, zeta, masks_batch=None, agent_indices=None, deterministic=False):
        batch_size = states_batch.shape[0]
        if agent_indices is None:
            agent_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        else:
            agent_indices = agent_indices.to(self.device).view(-1)

        if masks_batch is not None:
            masks_batch = masks_batch.to(self.device)

        states_batch  = states_batch.to(self.device).float()
        prev_mfs_batch = prev_mfs_batch.to(self.device).float()

        is_policy_agent = torch.tensor([
            (self.memory.get_len(aid.item()) >= self.min_batch_size) or deterministic
            for aid in agent_indices
        ], device=self.device)

        final_actions = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Cold-start: random exploration
        cold_mask = ~is_policy_agent
        if cold_mask.any():
            indices = cold_mask.nonzero(as_tuple=True)[0]
            if masks_batch is not None:
                m = masks_batch[indices]
                probs = m / m.sum(dim=1, keepdim=True).clamp(min=1e-8)
                final_actions[indices] = torch.multinomial(probs, 1).squeeze(1)
            else:
                final_actions[indices] = torch.randint(0, self.u_action_dim, (len(indices),), device=self.device)

        # Policy agents
        policy_mask = is_policy_agent
        if policy_mask.any():
            indices = policy_mask.nonzero(as_tuple=True)[0]
            s_subset   = states_batch[indices]
            mf_subset  = prev_mfs_batch[indices]
            aid_subset = agent_indices[indices]

            with torch.no_grad():
                pred_mf = self.mf_net(torch.cat([s_subset, mf_subset], dim=-1), indices=aid_subset)
                q_values = self.eval_net(s_subset, pred_mf, indices=aid_subset)

                if masks_batch is not None:
                    m = masks_batch[indices]
                    q_values = q_values + (m - 1.0) * 1e10

                if self.exclude_zero and self.u_action_dim > 1:
                    q_values[:, 0] -= 1e10

                # ── Sửa lỗi: Lấy argmax trên q_values CHƯA CLAMP để không làm mất tác dụng của masks ──
                q_values_unclamped = q_values.clone()

                # Guard: clamp q_values to prevent softmax overflow
                q_values = torch.nan_to_num(q_values, nan=0.0, posinf=50.0, neginf=-50.0)
                q_values = q_values.clamp(-50.0, 50.0)

                probs = torch.softmax(q_values, dim=1)

                # Guard: repair NaN rows (e.g. if all logits are identical after clamping)
                bad_rows = torch.isnan(probs).any(dim=1) | torch.isinf(probs).any(dim=1)
                if bad_rows.any():
                    if masks_batch is not None:
                        probs[bad_rows] = (masks_batch[indices][bad_rows].float() + 1e-8)
                    probs[bad_rows] = probs[bad_rows] / probs[bad_rows].sum(dim=1, keepdim=True)

                if random.random() < epsilon and not deterministic:
                    # epsilon-greedy random action
                    random_probs = torch.ones_like(q_values)
                    if masks_batch is not None:
                        m = masks_batch[indices]
                        random_probs = m / m.sum(dim=1, keepdim=True).clamp(min=1e-8)
                    else:
                        random_probs = random_probs / self.u_action_dim
                    final_actions[indices] = torch.multinomial(random_probs, 1).squeeze(1)
                else:
                    if deterministic:
                        final_actions[indices] = q_values_unclamped.argmax(dim=1)
                    else:
                        final_actions[indices] = torch.multinomial(probs, 1).squeeze(1)

        return final_actions.tolist()

    # ── MF training ───────────────────────────────────────────────────────────

    def store_transition_train_mf_batch(self, states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, agent_ids, masks=None, next_masks=None):
        self.memory.add_batch(states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, agent_ids, masks=masks, next_masks=next_masks)
        return self.learn_mf_batch(states, prev_mfs, curr_mfs, agent_ids)

    def learn_mf_batch(self, states_batch, prev_mf_batch, ground_truth_mf_batch, agent_ids):
        s    = states_batch.to(self.device).float()
        pmf  = prev_mf_batch.to(self.device).float()
        gt   = ground_truth_mf_batch.to(self.device).float()

        pred_mf = self.mf_net(torch.cat([s, pmf], dim=-1), indices=agent_ids)
        loss = self.loss_fn(pred_mf, gt).mean()

        self.mf_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mf_net.parameters(), max_norm=5.0)
        self.mf_optimizer.step()
        return loss.item()

    # ── Core learning step ────────────────────────────────────────────────────

    def learn(self, agents_ids: torch.Tensor = None, round_idx: int = 0):
        """
        One gradient step.
        round_idx: current federated round index, used to determine training phase.
        """
        ready_pool = (self.memory.buffer_sizes >= self.min_batch_size).nonzero(as_tuple=True)[0]

        if agents_ids is not None:
            agents_ids = agents_ids.to(self.device).view(-1)
            mask = torch.isin(agents_ids, ready_pool)
            target_agents = agents_ids[mask]
        else:
            target_agents = ready_pool

        if len(target_agents) == 0:
            return None

        states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, agent_ids, masks, next_masks = \
            self.memory.sample(self.batch_size, agent_ids=target_agents)

        self.learn_mf_batch(states, prev_mfs, curr_mfs, agent_ids)

        # ── Determine phase ───────────────────────────────────────────────────
        phase     = self._get_phase(round_idx)
        lambda_t  = self._get_lambda(round_idx)

        # Phase 3: freeze backbone (no graph needed, saves VRAM)
        self._freeze_backbone(phase == 3)

        # ── Forward pass ──────────────────────────────────────────────────────
        pred_curr_mfs = self.mf_net(torch.cat([states, prev_mfs], dim=-1), indices=agent_ids)
        feat          = self.eval_net.backbone(states, pred_curr_mfs.detach(), indices=agent_ids)
        q_eval        = self.eval_net.head(feat, indices=agent_ids).gather(1, actions)

        with torch.no_grad():
            next_pred_mfs = self.mf_net(torch.cat([next_states, curr_mfs], dim=-1), indices=agent_ids)
            next_feat     = self.eval_net.backbone(next_states, next_pred_mfs, indices=agent_ids)
            q_next_all    = self.eval_net.head(next_feat, indices=agent_ids)
            if next_masks is not None:
                q_next_all = q_next_all + (next_masks - 1.0) * 1e10
            next_actions  = q_next_all.argmax(dim=1, keepdim=True)
            next_feat_tgt = self.target_net.backbone(next_states, next_pred_mfs, indices=agent_ids)
            q_next        = self.target_net.head(next_feat_tgt, indices=agent_ids).gather(1, next_actions)
            q_target      = rewards + self.gamma * q_next * (1 - dones)

        # Fix 2: Clamp before loss to prevent Q-value explosion corrupting gradients
        q_eval   = q_eval.clamp(-50.0, 50.0)
        q_target = q_target.clamp(-50.0, 50.0)

        loss = self.loss_fn(q_eval, q_target).mean()

        # ── Backbone update (Phase 1 & 2 only) ────────────────────────────────
        self.bone_optimizer.zero_grad()
        self.head_optimizer.zero_grad()
        loss.backward()  # Fix 5: no retain_graph — single backward pass covers all leaf params

        if self.use_scaffold:
            with torch.no_grad():
                unique_ids = torch.unique(agent_ids)

                # Fix 3: Direct-index correction — no m_mask broadcasting.
                # PyTorch zeroes gradients for non-participating instances automatically.

                # ── Apply SCAFFOLD correction to backbone gradients ────────────
                if phase < 3:
                    bone_params = list(self.eval_net.backbone.parameters())
                    for i, p in enumerate(bone_params):
                        if p.grad is not None:
                            raw_g = p.grad.data.clone()
                            # g_b_corrected = g_b - c_b_local + c_b_global  (only active ids)
                            p.grad.data[unique_ids] = raw_g[unique_ids] + (
                                self.c_b_global[i][unique_ids] - self.c_b_local[i][unique_ids]
                            )
                            self.grad_b_sum[i][unique_ids] += raw_g[unique_ids]

                    self.steps_in_round[unique_ids] += 1

        # Clip and step
        if phase < 3:
            self._set_backbone_lr(phase)
            torch.nn.utils.clip_grad_norm_(self.eval_net.backbone.parameters(), max_norm=1.0)
            self.bone_optimizer.step()

        torch.nn.utils.clip_grad_norm_(self.eval_net.head.parameters(), max_norm=1.0)
        self.head_optimizer.step()

        self.learn_step_counter += 1
        # ✅ THÊM: Soft-update MỖI 3 bước local training
        if self.learn_step_counter % 3 == 0:
            self._soft_update()

        if self.logs_q:
            return {
                "loss":   loss.detach(),
                "q_min":  q_eval.min().detach(),
                "q_max":  q_eval.max().detach(),
                "q_mean": q_eval.mean().detach()
            }
        return loss.detach()

    # ── Aggregation helpers (called by strategy server) ───────────────────────

    def get_backbone_state(self, instance_ids: torch.Tensor):
        """Return backbone param slices for the given instances (for aggregation)."""
        return [p.data[instance_ids].clone() for p in self.eval_net.backbone.parameters()]

    def set_backbone_state(self, instance_ids: torch.Tensor, param_list):
        """Overwrite backbone params for given instances (after cluster averaging)."""
        with torch.no_grad():
            for p, new_val in zip(self.eval_net.backbone.parameters(), param_list):
                p.data[instance_ids] = new_val

    def get_c_b_local(self, instance_ids):
        return [c[instance_ids].clone() for c in self.c_b_local]

    def set_c_b_global(self, instance_ids, c_b_global_list):
        with torch.no_grad():
            for c_g, new_val in zip(self.c_b_global, c_b_global_list):
                c_g[instance_ids] = new_val



    def update_local_cvariates(self, instance_ids: torch.Tensor):
        """
        Call at the END of a round (before aggregation), for each instance.
        SCAFFOLD Option I (Karimireddy et al. 2020):
          c_i+ = c_i - c_global + (1/K) * sum_k grad_k

        Fix 1: Only update agents that actually trained this round (steps > 0).
        """
        if not self.use_scaffold:
            return
        with torch.no_grad():
            # Only process agents that actually ran gradient steps this round
            active_mask = self.steps_in_round[instance_ids] > 0
            if not active_mask.any():
                return
            active_ids = instance_ids[active_mask]
            steps = self.steps_in_round[active_ids].float()  # always > 0, no clamp needed

            for i in range(len(self.c_b_local)):
                K = steps.view(-1, *([1] * (self.grad_b_sum[i].dim() - 1)))
                
                # Tính delta_c
                delta_c = self.grad_b_sum[i][active_ids] / K
                
                # ✅ CLIP 1: Clip delta_c để tránh giá trị cực đoan
                delta_c = torch.clamp(delta_c, min=-1.0, max=1.0)
                
                # Update local control variate
                self.c_b_local[i][active_ids] = (
                    self.c_b_local[i][active_ids]
                    - self.c_b_global[i][active_ids]
                    + delta_c
                )
                
                # ✅ CLIP 2: Clip final c_b_local để tránh blow-up
                self.c_b_local[i][active_ids] = torch.clamp(
                    self.c_b_local[i][active_ids], 
                    min=-5.0, 
                    max=5.0
                )

    # ── Soft update & IO ──────────────────────────────────────────────────────

    def _soft_update(self):
        with torch.no_grad():
            for tp, ep in zip(self.target_net.parameters(), self.eval_net.parameters()):
                tp.data.copy_(self.alpha * ep.data + (1.0 - self.alpha) * tp.data)

    def save(self, path):
        torch.save({
            'eval_net':    self.eval_net.state_dict(),
            'target_net':  self.target_net.state_dict(),
            'mf_net':      self.mf_net.state_dict(),
            'bone_opt':    self.bone_optimizer.state_dict(),
            'head_opt':    self.head_optimizer.state_dict(),
            'mf_opt':      self.mf_optimizer.state_dict(),
            'learn_step':  self.learn_step_counter,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.eval_net.load_state_dict(ckpt['eval_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.mf_net.load_state_dict(ckpt['mf_net'])
        self.bone_optimizer.load_state_dict(ckpt['bone_opt'])
        self.head_optimizer.load_state_dict(ckpt['head_opt'])
        self.mf_optimizer.load_state_dict(ckpt['mf_opt'])
        self.learn_step_counter = ckpt.get('learn_step', 0)
