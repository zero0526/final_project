import torch.nn as nn
from typing import Tuple


class FFN(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_sizes: Tuple[int, ...],
    ):
        """
        Args:
            input_size (int):
            output_size (int):
            hidden_sizes (tuple): e.g: (64, 128, 32).
        """
        super(FFN, self).__init__()
        self.input_size = input_size

        layers = []
        in_features = input_size

        if len(hidden_sizes) == 0:
            layers.append(nn.Linear(in_features, output_size))
        else:
            for h_dim in hidden_sizes:
                layers.append(nn.Linear(in_features, h_dim))
                layers.append(nn.ReLU())
                in_features = h_dim
            layers.append(nn.Linear(in_features, output_size))

        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)