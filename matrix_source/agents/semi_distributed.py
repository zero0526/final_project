import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class MultiInstanceLinear(nn.Module):
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
    def __init__(self, num_instances, normalized_shape, eps=1e-6):
        super().__init__()
        self.num_instances = num_instances
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_instances, normalized_shape))

    def forward(self, x, indices):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        gamma = self.weight[indices]
        return x_norm * gamma


class MultiInstanceGRUCell(nn.Module):
    def __init__(self, num_instances, input_size, hidden_size):
        super().__init__()
        self.num_instances = num_instances
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(num_instances, input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.Tensor(num_instances, hidden_size, 3 * hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(num_instances, 3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(num_instances, 3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_instances):
            nn.init.orthogonal_(self.weight_ih[i])
            nn.init.orthogonal_(self.weight_hh[i])
            nn.init.zeros_(self.bias_ih[i])
            nn.init.zeros_(self.bias_hh[i])

    def forward(self, x, hx, indices=None):
        if indices is None:
            indices = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        w_ih = self.weight_ih[indices]
        w_hh = self.weight_hh[indices]
        b_ih = self.bias_ih[indices]
        b_hh = self.bias_hh[indices]
        gi = torch.bmm(x.unsqueeze(1), w_ih).squeeze(1) + b_ih
        gh = torch.bmm(hx.unsqueeze(1), w_hh).squeeze(1) + b_hh
        i_r, i_i, i_n = gi.chunk(3, dim=-1)
        h_r, h_i, h_n = gh.chunk(3, dim=-1)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = new_gate + input_gate * (hx - new_gate)
        return hy


# 2. ACTOR & CRITIC
class SequentialGRUActor(nn.Module):
    def __init__(self, state_dim, mf_dim, action_dim, hidden_dim, num_instances=1):
        super().__init__()
        self.num_instances = num_instances
        self.fc1 = MultiInstanceLinear(num_instances, state_dim + mf_dim, hidden_dim)
        self.norm1 = MultiInstanceRMSNorm(num_instances, hidden_dim)
        self.gru_cell = MultiInstanceGRUCell(num_instances, hidden_dim, hidden_dim)
        self.actor_logits = MultiInstanceLinear(num_instances, hidden_dim, action_dim)

    def forward_step(self, state, mf, hidden_state, indices=None):
        x = torch.cat([state, mf], dim=-1)
        x = F.silu(self.norm1(self.fc1(x, indices), indices))
        new_hidden = self.gru_cell(x, hidden_state, indices)
        logits = self.actor_logits(new_hidden, indices)
        return logits, new_hidden

    def evaluate(self, state, mf, hidden_state, action, masks=None, indices=None):
        logits, new_hidden = self.forward_step(state, mf, hidden_state, indices)
        if masks is not None:
            logits = logits.masked_fill(masks == 0, -1e9)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, new_hidden


class GlobalAggregatedCritic(nn.Module):
    def __init__(self, global_state_dim, hidden_sizes, num_instances):
        super().__init__()
        self.fc1 = MultiInstanceLinear(num_instances, global_state_dim, hidden_sizes[0])
        self.norm1 = MultiInstanceRMSNorm(num_instances, hidden_sizes[0])
        self.fc2 = MultiInstanceLinear(num_instances, hidden_sizes[0], hidden_sizes[1])
        self.norm2 = MultiInstanceRMSNorm(num_instances, hidden_sizes[1])
        self.value_head = MultiInstanceLinear(num_instances, hidden_sizes[1], 1)

    def forward(self, global_state, indices=None):
        x = F.silu(self.norm1(self.fc1(global_state, indices), indices))
        x = F.silu(self.norm2(self.fc2(x, indices), indices))
        return self.value_head(x, indices)


class MFNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes, num_instances):
        super().__init__()
        self.fc1 = MultiInstanceLinear(num_instances, input_dim, hidden_sizes[0])
        self.fc2 = MultiInstanceLinear(num_instances, hidden_sizes[0], output_dim)

    def forward(self, x, indices=None):
        x = F.relu(self.fc1(x, indices))
        return self.fc2(x, indices)



# 3. BUFFER — Thêm get_flat_list + pre-stack
class SequentialMultiAgentBuffer:
    def __init__(self, num_instances):
        self.num_instances = num_instances
        self.completed_trajectories = []
        self.current_trajectory = {}
        self.reset_current()

    def reset_current(self):
        self.current_trajectory = {i: {} for i in range(self.num_instances)}

    def start_episode(self):
        self.reset_current()

    def add_step(self, agent_id, state, hidden, action, log_prob, mf):
        aid = int(agent_id)
        if 'states' not in self.current_trajectory[aid]:
            self.current_trajectory[aid] = {
                'states': [], 'hiddens': [], 'actions': [],
                'log_probs': [], 'mfs': []
            }
        self.current_trajectory[aid]['states'].append(state)
        self.current_trajectory[aid]['hiddens'].append(hidden)
        self.current_trajectory[aid]['actions'].append(action)
        self.current_trajectory[aid]['log_probs'].append(log_prob)
        self.current_trajectory[aid]['mfs'].append(mf)

    def end_episode(self, agent_id, mf, mask, global_state, reward):
        aid = int(agent_id)
        self.current_trajectory[aid]['mf'] = mf
        self.current_trajectory[aid]['mask'] = mask
        self.current_trajectory[aid]['global_state'] = global_state
        self.current_trajectory[aid]['reward'] = reward

    def finalize_episode(self):
        has_data = any(
            len(v.get('states', [])) > 0
            for v in self.current_trajectory.values()
        )
        if has_data:
            self.completed_trajectories.append(self.current_trajectory)
        self.reset_current()

    def get_batch(self, batch_size):
        if len(self.completed_trajectories) < batch_size:
            return None
        return random.sample(self.completed_trajectories, batch_size)

    def clear(self):
        self.completed_trajectories = []
        self.reset_current()

    def __len__(self):
        return len(self.completed_trajectories)



# 4. PPO AGENT
class SequentialGRU_PPOAgent:
    def __init__(self, agent_id, node_type, actor_state_dim, critic_global_dim,
                 mf_action_dim, mf_hidden_sizes, mf_lr, action_dim=32,
                 hidden_dim=128, critic_hidden=(256, 128), lr=3e-4,
                 clip_eps=0.2, k_epochs=5, entropy_coef=0.01,
                 num_instances=1, zeta= 0.6, zeta_decay_rate = 0.99, max_zeta = 5, device=None):

        self.agent_id = agent_id
        self.node_type = node_type
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.num_instances = num_instances
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.learn_step_counter = 0

        self.actor = SequentialGRUActor(
            actor_state_dim, mf_action_dim, action_dim, hidden_dim, num_instances
        ).to(self.device)
        self.critic = GlobalAggregatedCritic(
            critic_global_dim, critic_hidden, num_instances
        ).to(self.device)
        self.mf_net = MFNetwork(
            actor_state_dim + mf_action_dim, mf_action_dim,
            mf_hidden_sizes, num_instances
        ).to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.mf_optimizer = optim.Adam(self.mf_net.parameters(), lr=mf_lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.memory = SequentialMultiAgentBuffer(num_instances)
        self.initial_entropy_coef = entropy_coef  # Lưu lại giá trị ban đầu (ví dụ 0.05)
        self.entropy_coef = entropy_coef          # Giá trị đang dùng hiện tại
        self.entropy_decay_rate = 0.99          # Tốc độ giảm sau mỗi lần learn (thử 0.999 - 0.9999)
        self.min_entropy_coef = 0.001             # Giá trị nhỏ nhất cho phép (không để nó bằng 0 hoàn toàn)
        self.zeta= zeta
        self.zeta_decay_rate = zeta_decay_rate
        self.max_zeta = max_zeta
    # ──────────────────────────────────────────────────
    # INFERENCE (GIỮ NGUYÊN)
    # ──────────────────────────────────────────────────
    def choose_action(self, state, mf, hidden, mask, agent_idx=0):
        res = self.choose_action_batch(
            torch.as_tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(mf, device=self.device, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(hidden, device=self.device, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(mask, device=self.device, dtype=torch.float32).unsqueeze(0),
            torch.tensor([agent_idx], device=self.device)
        )
        return res[0][0].item(), res[1][0].item(), res[2][0].cpu().numpy()

    def choose_action_batch(self, states, mfs, hiddens, masks, agent_indices):
        with torch.no_grad():
            logits, new_hiddens = self.actor.forward_step(
                states, mfs, hiddens, agent_indices
            )
            logits = logits.masked_fill(masks == 0, -1e9)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
        return actions, log_probs, new_hiddens

    # ──────────────────────────────────────────────────
    # MF LEARNING (GIỮ NGUYÊN)
    # ──────────────────────────────────────────────────
    def learn_mf(self, state, prev_mf, ground_truth_mf, agent_ids):
        s = torch.as_tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        pmf = torch.as_tensor(prev_mf, device=self.device, dtype=torch.float32).unsqueeze(0)
        gt = torch.as_tensor(ground_truth_mf, device=self.device, dtype=torch.float32).unsqueeze(0)
        idx = torch.tensor([agent_ids], device=self.device)
        pred = self.mf_net(torch.cat([s, pmf], dim=-1), indices=idx)
        loss = self.loss_fn(pred, gt)
        self.mf_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.mf_optimizer.step()

    def learn(self, batch_size=128, k_epochs=4):
        data_list = self.memory.get_batch(batch_size)
        if data_list is None:
            return None

        # ── Alpha decay ──
        self.entropy_coef = max(
            self.entropy_coef * self.entropy_decay_rate,
            self.min_entropy_coef
        )

        # ═══════════════════════════════════════════════════
        # FIX-1: LIST thay vì dict → không ghi đè data
        # ═══════════════════════════════════════════════════
        all_entries = []
        for trajectory_data in data_list:
            for aid, data in trajectory_data.items():
                if len(data.get('states', [])) == 0:
                    continue
                idx = torch.tensor([aid], device=self.device)
                gs = data['global_state'].to(self.device).unsqueeze(0)
                rew = torch.tensor(
                    [data['reward']], dtype=torch.float32, device=self.device
                )
                with torch.no_grad():
                    V = self.critic(gs, indices=idx).squeeze(-1)
                    adv = rew - V

                all_entries.append({
                    'aid': aid,
                    'idx': idx,
                    'gs': gs, 'rew': rew, 'adv': adv.detach(),
                    'mask': data['mask'].to(self.device).unsqueeze(0),
                    'seq_len': len(data['states']),
                    'states': torch.stack([
                        s.to(self.device) if s.device != self.device else s
                        for s in data['states']
                    ]),
                    'mfs': torch.stack([
                        m.to(self.device) if m.device != self.device else m
                        for m in data['mfs']
                    ]),
                    'actions': torch.tensor(
                        data['actions'], dtype=torch.long, device=self.device
                    ),
                    'old_lps': torch.tensor(
                        data['log_probs'], dtype=torch.float32, device=self.device
                    ),
                })

        if not all_entries:
            return 0

        h_dim = self.actor.gru_cell.hidden_size
        epoch_v_loss = 0.0
        total_entropy = 0.0
        num_entries = len(all_entries)

        for _ in range(k_epochs):
            for entry in all_entries:
                idx = entry['idx']
                T = entry['seq_len']

                # ═══════════════════════════════════
                # FIX-2: Per-agent step (Adam tương thích)
                # ═══════════════════════════════════

                # ── Actor ──
                self.optimizer_actor.zero_grad(set_to_none=True)

                h_gru = torch.zeros(1, h_dim, device=self.device)
                agent_actor_loss = torch.tensor(0.0, device=self.device)
                agent_entropy = torch.tensor(0.0, device=self.device)

                for k in range(T):
                    log_prob, entropy, h_gru = self.actor.evaluate(
                        entry['states'][k].unsqueeze(0),
                        entry['mfs'][k].unsqueeze(0),
                        h_gru,
                        entry['actions'][k].unsqueeze(0),
                        masks=entry['mask'],
                        indices=idx
                    )
                    ratio = torch.exp(log_prob - entry['old_lps'][k])
                    surr1 = ratio * entry['adv']
                    surr2 = torch.clamp(
                        ratio, 1 - self.clip_eps, 1 + self.clip_eps
                    ) * entry['adv']
                    agent_actor_loss += -torch.min(surr1, surr2).mean()
                    agent_entropy += entropy.mean()

                agent_actor_loss /= T
                agent_entropy /= T

                (agent_actor_loss - self.entropy_coef * agent_entropy).backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.optimizer_actor.step()

                # ── Critic ──
                self.optimizer_critic.zero_grad(set_to_none=True)

                V_pred = self.critic(entry['gs'], indices=idx).squeeze(-1)
                critic_loss = F.mse_loss(V_pred, entry['rew'])

                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.optimizer_critic.step()

                epoch_v_loss += critic_loss.item()
                total_entropy += agent_entropy.item()

        return epoch_v_loss / (num_entries * k_epochs) if num_entries > 0 else 0

    # ──────────────────────────────────────────────────
    # SAVE / LOAD (GIỮ NGUYÊN)
    # ──────────────────────────────────────────────────
    def save(self, checkpoint_path):
        directory = os.path.dirname(checkpoint_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'mf_net_state_dict': self.mf_net.state_dict(),
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict(),
            'mf_optimizer': self.mf_optimizer.state_dict()
        }, checkpoint_path)

    def load(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor_state_dict'])
        self.critic.load_state_dict(ckpt['critic_state_dict'])
        self.mf_net.load_state_dict(ckpt['mf_net_state_dict'])
        self.optimizer_actor.load_state_dict(ckpt['optimizer_actor'])
        self.optimizer_critic.load_state_dict(ckpt['optimizer_critic'])
        self.mf_optimizer.load_state_dict(ckpt['mf_optimizer'])
