import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple
from src.agents.ffn import FFN
from src.agents.ReplayBuffer import ReplayBuffer

from configs.configs import cfg

class RunningNorm:
    def __init__(self, shape):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = 1e-4

    def update(self, x):
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = x.size(0)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / (torch.sqrt(self.var) + 1e-8)

class DuelingNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...]):
        super(DuelingNetwork, self).__init__()
        # Shared Base Feature Extractor (FedRep Base)
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU()
        )
        # Personalized Heads (FedRep Heads)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_dim)
        )

    def forward(self, state, pred_mf):
        x = torch.cat([state, pred_mf], dim=-1)
        features = self.base(x)
        V = self.value_stream(features)
        A = self.advantage_stream(features)

        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q

    def get_base_params(self):
        return list(self.base.parameters())

# ---D3QN AGENT ---
class D3QNAgent:
    def __init__(self, node_id:str, node_type: str, state_dim, action_dim, u_action_dim: int, mf_hidden_sizes: Tuple[int, ...],mf_lr:float, hidden_sizes=(128, 64),
                 lr=1e-4, gamma=0.99, alpha=0.005, buffer_size=cfg.hyper_neural["MEMORY_SIZE"], buffer_min_size=cfg.hyper_neural["BUFFER_MIN_SIZE"], batch_size=64, exclude_zero=False):
        self.action_dim = action_dim
        self.u_action_dim = u_action_dim # Store u_action_dim
        self.exclude_zero = exclude_zero
        self.gamma = gamma
        self.alpha = float(alpha)  # Ensure it is a scalar float
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.node_id= node_id
        self.node_type = node_type
        self.min_batch_size= buffer_min_size[0] if node_type!="terminal" else buffer_min_size[1]
        # Evaluation Network, Target Network
        input_dim = state_dim + action_dim
        self.eval_net = DuelingNetwork(input_dim, u_action_dim, hidden_sizes).to(self.device)
        self.target_net = DuelingNetwork(input_dim, u_action_dim, hidden_sizes).to(self.device)

        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.mf_net = FFN(input_size=state_dim + action_dim, output_size=action_dim, hidden_sizes=mf_hidden_sizes).to(self.device)
        self.mf_optimizer = optim.Adam(self.mf_net.parameters(), lr=mf_lr)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayBuffer(buffer_size, state_dim, action_dim, self.device)
        
        # SCAFFOLD Control Variates
        self.c_i = [torch.zeros_like(p).to(self.device) for p in self.eval_net.get_base_params()]
        self.c_edge = [torch.zeros_like(p).to(self.device) for p in self.eval_net.get_base_params()]
        self.grad_sum = [torch.zeros_like(p).to(self.device) for p in self.eval_net.get_base_params()]
        self.initial_base_params = []
        self.steps_in_round = 0
        self.save_base_initial()
        
        # Logging state
        self.prev_loss = 0.0
        self.learn_step_counter = 0

    def save_base_initial(self):
        """Save base weights at the start of a local round for SCAFFOLD."""
        self.initial_base_params = [p.data.clone() for p in self.eval_net.get_base_params()]
        for g in self.grad_sum:
            g.zero_()
        self.steps_in_round = 0

    def choose_action(self, state, prev_mf, epsilon, zeta, mask:np.ndarray= None, assigned_node_id=None):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mf_tensor = torch.FloatTensor(prev_mf).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mf_input = torch.cat([state_tensor, mf_tensor], dim=-1)
            pred_mf = self.mf_net(mf_input)
            
            # Predict Q-values
            q_values = self.eval_net(state_tensor, pred_mf)

            # Apply masking mechanism
            if mask is not None and np.any(mask):
                mask_tensor = torch.FloatTensor(mask).to(self.device).unsqueeze(0)
                # Apply large penalty to masked actions (-1e18 for numerical stability in softmax)
                q_values = q_values + (mask_tensor - 1.0) * 1e18
            
            if self.exclude_zero and self.u_action_dim > 1:
                q_values[:, 0] -= 1e18
            
            # Boltzmann Selection (Softmax with temperature parameter zeta)
            # P(a) = exp(zeta * Q_a) / sum(exp(zeta * Q_i))
            # We use torch.softmax on (zeta * Q) for numerical stability.
            scaled_q = q_values * zeta
            probs = torch.softmax(scaled_q, dim=1).cpu().numpy().squeeze()
            
            # Weighted random selection based on Boltzmann probabilities
            action = np.random.choice(len(probs), p=probs)
            
        return int(action)

    def learn_mf(self, state, prev_mf, ground_truth_mf):
        # We now train MF network via Minibatch in learn() to avoid noisy Batch=1 updates
        pass

    def store_transition(self, state, prev_mf, curr_mf, action, reward, next_state, done):
        self.memory.add(state, prev_mf, curr_mf, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.min_batch_size:
            return None  # dont enough data

        # Sample data
        states, prev_mfs, curr_mfs, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # ---------------- TRAIN MF NETWORK ----------------
        mf_input = torch.cat([states, prev_mfs], dim=-1)
        pred_curr_mfs = self.mf_net(mf_input)
        mf_loss = self.loss_fn(pred_curr_mfs, curr_mfs)
        
        self.mf_optimizer.zero_grad()
        mf_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mf_net.parameters(), max_norm=1.0)
        self.mf_optimizer.step()

        # ---------------- DOUBLE DQN LOGIC ----------------
        pred_mfs_detached = pred_curr_mfs.detach()
        q_eval = self.eval_net(states, pred_mfs_detached).gather(1, actions)

        with torch.no_grad():
            next_mf_input = torch.cat([next_states, curr_mfs], dim=-1)
            next_pred_mfs = self.mf_net(next_mf_input)

            next_actions = self.eval_net(next_states, next_pred_mfs).argmax(dim=1, keepdim=True)

            # Bước 2: Đánh giá action đó bằng Target Network (Phương trình 47)
            q_next = self.target_net(next_states, next_pred_mfs).gather(1, next_actions)

            # y = r + gamma * Q_target(s', argmax Q_eval(s', a'))
            q_target = rewards + self.gamma * q_next * (1 - dones)

        # ---------------- LOSS & BACKPROP ----------------
        # Phương trình (48): L = MSE(Q_eval, y)
        loss = self.loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        
        # SCAFFOLD Gradient Correction ONLY on Base Params
        with torch.no_grad():
            for p, g_sum, cp_i, cp_edge in zip(self.eval_net.get_base_params(), self.grad_sum, self.c_i, self.c_edge):
                if p.grad is not None:
                    raw_g = p.grad.data.clone()
                    p.grad.data = raw_g - cp_i + cp_edge
                    g_sum += raw_g

        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=1.0)
        self.steps_in_round += 1
        self.optimizer.step()
        
        # ---------------- LOGGING ----------------
        loss_val = loss.item()
        avg_q = q_eval.mean().item()
        loss_change = loss_val - self.prev_loss
        self.prev_loss = loss_val
        self.learn_step_counter += 1

        # Log every 100 learning steps to avoid console flooding
        if self.learn_step_counter % 100 == 0:
            print(f"[Agent {self.node_id} ({self.node_type})] Update {self.learn_step_counter:5d} | "
                  f"Avg Q: {avg_q:8.3f} | TD Loss: {loss_val:8.5f} | ΔLoss: {loss_change:9.5f}")

        # ---------------- SOFT UPDATE ----------------
        self._soft_update()

        return loss.item()

    def _soft_update(self):
        """
        theta_target = alpha * theta_eval + (1 - alpha) * theta_target
        """
        with torch.no_grad():
            for target_param, eval_param in zip(self.target_net.parameters(), self.eval_net.parameters()):
                target_param.data.copy_(
                    self.alpha * eval_param.data + (1.0 - self.alpha) * target_param.data
                )