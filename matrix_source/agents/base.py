import torch
from torch import nn

class MultiInstanceLinear(nn.Module):
    """
    Parallel linear layer for multiple independent agent models.
    Supports batched inference where each sample in the batch can 
    use a specific agent's weights.
    """
    def __init__(self, num_instances, in_features, out_features, bias=True):
        super().__init__()
        self.num_instances = num_instances
        self.in_features = in_features
        self.out_features = out_features
        
        # Weights: (num_instances, in, out)
        self.weight = nn.Parameter(torch.Tensor(num_instances, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_instances, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Orthogonal initialization per instance
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
    """
    Instance-specific Root Mean Square Layer Normalization.
    """
    def __init__(self, num_instances, normalized_shape, eps=1e-6):
        super().__init__()
        self.num_instances = num_instances
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_instances, normalized_shape))

    def forward(self, x, indices):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        gamma = self.weight[indices]
        return x_norm * gamma

class MultiInstanceNoisyLinear(nn.Module):
    def __init__(self, num_instances, in_features, out_features, std_init=0.5):
        super().__init__()
        self.num_instances = num_instances
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(num_instances, in_features, out_features))
        self.weight_sigma = nn.Parameter(torch.empty(num_instances, in_features, out_features))
        self.register_buffer("weight_epsilon", torch.empty(num_instances, in_features, out_features))

        self.bias_mu = nn.Parameter(torch.empty(num_instances, out_features))
        self.bias_sigma = nn.Parameter(torch.empty(num_instances, out_features))
        self.register_buffer("bias_epsilon", torch.empty(num_instances, out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / (self.in_features ** 0.5)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / (self.in_features ** 0.5))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / (self.out_features ** 0.5))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self._scale_noise((self.num_instances, self.in_features))
        epsilon_out = self._scale_noise((self.num_instances, self.out_features))
        
        # Outer product for each instance: (num_instances, in, out)
        # epsilon_out.unsqueeze(1): (num_instances, 1, out)
        # epsilon_in.unsqueeze(2): (num_instances, in, 1)
        self.weight_epsilon.copy_(torch.bmm(epsilon_in.unsqueeze(2), epsilon_out.unsqueeze(1)))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x, indices=None):
        if indices is None:
            indices = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        if self.training:
            weight = self.weight_mu[indices] + self.weight_sigma[indices] * self.weight_epsilon[indices]
            bias = self.bias_mu[indices] + self.bias_sigma[indices] * self.bias_epsilon[indices]
        else:
            weight = self.weight_mu[indices]
            bias = self.bias_mu[indices]

        # x: (Batch, In) -> (Batch, 1, In)
        # weight: (Batch, In, Out)
        out = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
        return out

class MF(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_sizes, num_instances=1):
        super().__init__()
        h = hidden_sizes[0]
        self.num_instances = num_instances
        
        self.l1 = MultiInstanceLinear(num_instances, in_dim, h)
        self.norm = MultiInstanceRMSNorm(num_instances, h)
        self.l2 = MultiInstanceLinear(num_instances, h, out_dim)

    def forward(self, x, indices=None):
        x = torch.nn.functional.silu(self.norm(self.l1(x, indices), indices))
        return torch.sigmoid(self.l2(x, indices))

class MultiInstanceGRUCell(nn.Module):
    """Instance-specific GRU Cell."""

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