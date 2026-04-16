import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple

from torch.distributions import Categorical

from src.agents.ffn import FFN
from src.agents.ReplayBuffer import ReplayBuffer

class DuelingNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...]):
        super(DuelingNetwork, self).__init__()

        self.value_stream = FFN(input_size=input_dim, output_size=1, hidden_sizes=hidden_sizes)
        self.advantage_stream = FFN(input_size=input_dim, output_size=action_dim, hidden_sizes=hidden_sizes)

    def forward(self, state, pred_mf):
        x = torch.cat([state, pred_mf], dim=-1)
        V = self.value_stream(x)
        A = self.advantage_stream(x)

        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q

# ---D3QN AGENT ---
class D3QNAgent:
    def __init__(self, state_dim, action_dim, u_action_dim: int, mf_hidden_sizes: Tuple[int, ...],mf_lr:float, hidden_sizes=(128, 64),
                 lr=1e-4, gamma=0.99, alpha=0.005, buffer_size=100000, batch_size=64):
        self.action_dim = action_dim
        self.gamma = gamma
        self.alpha = float(alpha)  # Ensure it is a scalar float
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def choose_action(self, state, prev_mf, epsilon, zeta, mask:np.ndarray= None):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mf_tensor = torch.FloatTensor(prev_mf).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mf_input = torch.cat([state_tensor, mf_tensor], dim=-1)
            pred_mf = self.mf_net(mf_input)

            if np.random.rand() < epsilon:
                if mask is not None and np.any(mask):
                    valid_indices = np.where(mask == 1)[0]
                    action = np.random.choice(valid_indices)
                else:
                    action = np.random.randint(self.action_dim)
                return int(action)

            q_values = self.eval_net(state_tensor, pred_mf)

            scaled_q_values = zeta * q_values
            if mask is not None and np.any(mask):
                mask_tensor = torch.FloatTensor(mask).to(self.device).unsqueeze(0)
                scaled_q_values = scaled_q_values + (mask_tensor - 1.0) * 1e9
            action_probs = F.softmax(scaled_q_values, dim=1)
            dist = Categorical(action_probs)
            action = dist.sample()

        return action.item()

    def learn_mf(self, state, prev_mf, ground_truth_mf):
        # online training mf_net
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        prev_mf_tensor = torch.FloatTensor(prev_mf).unsqueeze(0).to(self.device)
        gt_mf_tensor = torch.FloatTensor(ground_truth_mf).unsqueeze(0).to(self.device)

        mf_input = torch.cat([state_tensor, prev_mf_tensor], dim=-1)
        pred_mf = self.mf_net(mf_input)

        loss = self.loss_fn(pred_mf, gt_mf_tensor)

        self.mf_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.mf_net.parameters(), max_norm=1.0)
        self.mf_optimizer.step()

    def store_transition(self, state, prev_mf, curr_mf, action, reward, next_state, done):
        self.memory.add(state, prev_mf, curr_mf, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None  # dont enough data

        # Sample data
        states, prev_mfs, curr_mfs, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # ---------------- DOUBLE DQN LOGIC ----------------
        mf_input = torch.cat([states, prev_mfs], dim=-1)
        pred_mfs = self.mf_net(mf_input).detach()
        q_eval = self.eval_net(states, pred_mfs).gather(1, actions)

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
        # torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=1.0)
        self.optimizer.step()

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