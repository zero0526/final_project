import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, device="cpu"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device

        # Pre-allocate with np.ndarray 0
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.prev_mf = np.zeros((max_size, action_dim), dtype=np.float32)
        self.curr_mf = np.zeros((max_size, action_dim), dtype=np.float32)
        self.action = np.zeros((max_size, 1), dtype=np.int64)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, prev_mf, curr_mf, action, reward, next_state, done):
        self.state[self.ptr] = state

        self.prev_mf[self.ptr] = prev_mf
        self.curr_mf[self.ptr] = curr_mf
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # replace=False
        if batch_size < self.size:
            ind= np.random.randint(0, self.size, size=batch_size)
        else: ind= np.random.randint(0, self.size, size=self.size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.prev_mf[ind]).to(self.device),
            torch.FloatTensor(self.curr_mf[ind]).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )

    def __len__(self):
        return self.size