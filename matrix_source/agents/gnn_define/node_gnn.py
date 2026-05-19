import torch.nn.functional as F
import torch.nn as nn
import torch
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class NodeGNN(nn.Module):
    """
    A multi-layer Graph Attention Network (GAT) that encodes each computing
    node's state into a rich embedding, capturing both local features and
    neighbour context via attention-weighted message passing.

    Parameters
    ----------
    num_services : int
        |S| - number of services in the system.
    hidden_dim : int
        Hidden feature dimension per layer.
    output_dim : int
        Embedding dimension output for each node.
    num_heads : int
        Number of attention heads per GAT layer.
    num_layers : int
        Depth of the GAT stack. Must be >= 1.
    edge_dim : int
        Dimension of edge features (default 2).
    dropout : float
        Dropout rate applied to intermediate representations.
    """

    NODE_STATIC_DIM = 6   # [type_onehot(3) + f_max(1) + storage(2)]
    DYNAMIC_BASE_DIM = 1  # f_avail(1)
    DYNAMIC_PER_SVC = 3   # placement(1) + backlog(1) + alloc(1) per service

    def __init__(
        self,
        num_services: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        edge_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        in_dim = self.NODE_STATIC_DIM + self.DYNAMIC_BASE_DIM + self.DYNAMIC_PER_SVC * num_services

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_dim
            out_ch = hidden_dim // num_heads
            is_last = (i == num_layers - 1)
            # Last layer: single-head for stable output
            heads = 1 if is_last else num_heads
            concat = False if is_last else True
            self.gat_layers.append(
                GATConv(
                    in_channels=in_ch,
                    out_channels=out_ch if not is_last else output_dim,
                    heads=heads,
                    concat=concat,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    add_self_loops=False,
                )
            )

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim if i < num_layers - 1 else output_dim)
            for i in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim

    # ------------------------------------------------------------------

    def forward(self, data: Data) -> torch.Tensor:
        """
        Parameters
        ----------
        data : PyG Data
            .x         (N, in_dim)
            .edge_index (2, E)
            .edge_attr  (E, edge_dim)

        Returns
        -------
        node_embeddings : (N, output_dim)
        """
        x = self.input_proj(data.x)

        for i, (gat, ln) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x_new = gat(x, data.edge_index, edge_attr=data.edge_attr)
            x_new = ln(x_new)
            x_new = F.elu(x_new)
            x_new = self.dropout(x_new)
            # Residual connection when shapes match
            if x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new

        return x   # (N, output_dim)