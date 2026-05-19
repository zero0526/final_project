"""
GNN (Graph Attention Network) for Cloud-Edge-Network State Encoding.

This module provides:
1. `GraphBuilder`: Constructs a virtual fully-connected graph from the current
   environment state. Nodes = Cloud + Edge + Network computing nodes (no replay/terminal).
2. `NodeGNN`: A multi-layer GAT that encodes node-level features via message-passing
   over the virtual graph, returning rich node embeddings for downstream agents.

Node Feature Dimensions (per node):
    - Static / semi-static (6 dim):
        [is_edge, is_network, is_cloud]          : 3  (one-hot type)
        f_v_max (normalized)                     : 1  (total compute capacity)
        C_v_VRAM / C_v_HDD (normalized)           : 2  (storage capacity)
    - Dynamic per-timeslot (1 + |S| + |S| + |S| dim):
        f_v_avail (normalized)                   : 1  (available compute)
        x_v (binary placement)                   : |S|
        Q_v_s (log-transformed backlog)          : |S|
        f_v_s (allocated resource, normalized)   : |S|
    Total: 7 + 3*|S|

Edge Feature Dimensions (per directed edge u->v):
    - hop_weight = 1 / (1 + hop_count)          : 1
    - link_rate_normalized                       : 1
    Total: 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
_TORCH_GEOMETRIC_AVAILABLE = True



# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

class GraphBuilder:
    """
    Builds PyTorch Geometric Data objects from the environment's dynamic state.

    The virtual graph is FULLY CONNECTED over all computing nodes
    (edge + network + cloud). Replay/terminal nodes are excluded.

    Parameters
    ----------
    static_matrices : dict
        Output of `init_static_matrices`. Must contain:
          - 'resource_matrix'   : (N, >=3) [cpu, ram, hdd, ...]
          - 'adj_matrix'        : (N, N) binary adjacency (used for hop-count)
          - 'comp_node_id_to_idx' : {node_id -> row_index}
        Extra expected from init (we derive internally):
          - 'edge_ids'          : list[int] of comp-node row indices for edge nodes
          - 'cloud_ids'         : list[int] of comp-node row indices for cloud nodes
    config : object
        Project config; used for services and hyper_neural parameters.
    hop_matrix : torch.Tensor (N, N)
        Pre-computed all-pairs hop distances (see `GraphBuilder.build_hop_matrix`).
    max_link_rate : float
        Maximum link transmission rate (Mbps) used for normalization.
    device : str | torch.device
    """

    def __init__(self, static_matrices, config, hop_matrix, max_link_rate=None, device="cpu"):
        self.device = device
        self.num_services = len(config.services)
        self.max_link_rate = max_link_rate or 1.0

        resource_matrix = static_matrices['resource_matrix']   # (N, >=3)
        edge_ids = static_matrices['edge_ids']                  # list[int]
        cloud_ids = static_matrices['cloud_ids']                # list[int]
        N = resource_matrix.shape[0]

        # ---- Node type one-hot -----------------------------------------------
        # Remaining indices (neither edge nor cloud) are treated as "network"
        edge_set = set(edge_ids)
        cloud_set = set(cloud_ids)
        type_onehot = torch.zeros(N, 3, device=device)  # [is_edge, is_network, is_cloud]
        for i in range(N):
            if i in edge_set:
                type_onehot[i, 0] = 1.0
            elif i in cloud_set:
                type_onehot[i, 2] = 1.0
            else:
                type_onehot[i, 1] = 1.0

        # ---- Capacity normalization ------------------------------------------
        f_max = resource_matrix[:, 0]                           # (N,) - CPU/GFLOPS
        max_cpu = f_max.max().clamp(min=1.0)
        f_max_norm = (f_max / max_cpu).unsqueeze(1)             # (N, 1)

        ram_total = resource_matrix[:, 1]
        hdd_total = resource_matrix[:, 2]
        max_ram = ram_total.max().clamp(min=1.0)
        max_hdd = hdd_total.max().clamp(min=1.0)
        storage_norm = torch.stack([
            ram_total / max_ram,
            hdd_total / max_hdd
        ], dim=1)                                               # (N, 2)

        # Store immutable static features
        self.static_node_feat = torch.cat([type_onehot, f_max_norm, storage_norm], dim=1)  # (N, 6)
        self.f_max = f_max.to(device)                           # for avail-compute norm
        self.N = N

        # ---- Edge index: fully-connected (no self-loops) ---------------------
        rows, cols = [], []
        for i in range(N):
            for j in range(N):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        self.edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)  # (2, N*(N-1))

        # ---- Pre-compute edge features (hop weight + rate) -------------------
        self.hop_matrix = hop_matrix.to(device)                 # (N, N)
        self.hop_weight_matrix = 1.0 / (1.0 + self.hop_matrix) # fuzzy hop weighting

    # ------------------------------------------------------------------

    @staticmethod
    def build_hop_matrix(G, comp_node_id_to_idx, device="cpu"):
        """
        Compute all-pairs shortest-path hop-counts for computing nodes.

        Parameters
        ----------
        G : networkx.Graph   (the same topology used in init_matrices)
        comp_node_id_to_idx : dict  {node_id -> row_index}

        Returns
        -------
        hop_matrix : torch.FloatTensor (N, N)
        """
        import networkx as nx

        N = len(comp_node_id_to_idx)
        hop_matrix = torch.zeros(N, N, device=device)
        idx_to_id = {v: k for k, v in comp_node_id_to_idx.items()}

        for i in range(N):
            src = idx_to_id[i]
            lengths = nx.single_source_shortest_path_length(G, src)
            for j in range(N):
                dst = idx_to_id[j]
                hop_matrix[i, j] = float(lengths.get(dst, float('inf')))

        return hop_matrix

    # ------------------------------------------------------------------

    def build_rate_matrix(self, G, comp_node_id_to_idx):
        """
        Extract average link transmission rate for each pair of nodes.
        Used to build normalized edge feature for link rate.

        Returns
        -------
        rate_matrix : torch.FloatTensor (N, N)  normalized by max_link_rate
        """

        N = len(comp_node_id_to_idx)
        rate_matrix = torch.zeros(N, N, device=self.device)
        idx_to_id = {v: k for k, v in comp_node_id_to_idx.items()}

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                try:
                    path = nx.shortest_path(G, idx_to_id[i], idx_to_id[j])
                    rates = [G[path[k]][path[k+1]].get('rate', self.max_link_rate)
                             for k in range(len(path) - 1)]
                    avg_rate = sum(rates) / len(rates) if rates else self.max_link_rate
                    rate_matrix[i, j] = avg_rate / self.max_link_rate
                except Exception:
                    rate_matrix[i, j] = 0.0

        self.rate_matrix = rate_matrix
        return rate_matrix

    # ------------------------------------------------------------------

    def build_graph(
        self,
        f_avail,           # (N,)   available compute capacity (current timeslot)
        placement_matrix,  # (N, S) binary service placement
        backlog_counts,    # (N, S) queue backlog counts
        cpu_alloc_matrix,  # (N, S) currently allocated CPU per service
        rate_matrix=None,  # (N, N) optional; uses stored if available
    ) -> Data:
        """
        Construct a PyG Data object for the current timeslot.

        Parameters
        ----------
        f_avail : Tensor (N,)
        placement_matrix : Tensor (N, S)
        backlog_counts : Tensor (N, S)
        cpu_alloc_matrix : Tensor (N, S)
        rate_matrix : Tensor (N, N) | None

        Returns
        -------
        PyG Data object with:
            x          : (N, 7 + 3*S)  node features
            edge_index : (2, N*(N-1))
            edge_attr  : (N*(N-1), 2)
        """
        device = self.device
        if rate_matrix is None:
            rate_matrix = getattr(self, 'rate_matrix', torch.zeros_like(self.hop_matrix))

        # ---- Dynamic node features ------------------------------------------
        f_max_safe = self.f_max.clamp(min=1.0)
        f_avail_norm = (f_avail.to(device) / f_max_safe).unsqueeze(1)          # (N, 1)

        x_v = placement_matrix.float().to(device)                              # (N, S)
        q_log = torch.log1p(backlog_counts.float().to(device))                 # (N, S)
        f_vs_norm = (cpu_alloc_matrix.float().to(device) /
                     f_max_safe.unsqueeze(1).clamp(min=1.0))                   # (N, S)

        node_feat = torch.cat([
            self.static_node_feat,    # (N, 6)
            f_avail_norm,             # (N, 1)
            x_v,                      # (N, S)
            q_log,                    # (N, S)
            f_vs_norm,                # (N, S)
        ], dim=1)                                                               # (N, 7+3S)

        # ---- Edge features --------------------------------------------------
        src, dst = self.edge_index[0], self.edge_index[1]
        hop_weights = self.hop_weight_matrix[src, dst].unsqueeze(1)            # (E, 1)
        rates = rate_matrix[src, dst].unsqueeze(1)                             # (E, 1)
        edge_attr = torch.cat([hop_weights, rates], dim=1)                     # (E, 2)

        return Data(x=node_feat, edge_index=self.edge_index, edge_attr=edge_attr)


# ---------------------------------------------------------------------------
# GNN Model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_gnn_and_graph_builder(
    config,
    static_matrices,
    G,                  # networkx.Graph for the full topology
    device="cpu",
    hidden_dim=128,
    output_dim=64,
    num_heads=4,
    num_layers=2,
    dropout=0.1,
):
    """
    Convenience factory that creates matching GraphBuilder + NodeGNN instances.

    Parameters
    ----------
    config : project config object
    static_matrices : dict (from init_static_matrices)
    G : networkx.Graph  (from init_static_matrices' internal graph)
    device : str | torch.device
    hidden_dim, output_dim, num_heads, num_layers, dropout : GNN hyper-params

    Returns
    -------
    graph_builder : GraphBuilder
    gnn_model     : NodeGNN
    """
    import networkx as nx

    # Build hop matrix
    comp_node_id_to_idx = static_matrices['comp_node_id_to_idx']
    hop_matrix = GraphBuilder.build_hop_matrix(G, comp_node_id_to_idx, device=device)

    # Infer max link rate from G
    rates = [d.get('rate', 1.0) for _, _, d in G.edges(data=True)]
    max_link_rate = max(rates) if rates else 1.0

    graph_builder = GraphBuilder(
        static_matrices=static_matrices,
        config=config,
        hop_matrix=hop_matrix,
        max_link_rate=max_link_rate,
        device=device,
    )

    # Pre-compute rate matrix
    graph_builder.build_rate_matrix(G, comp_node_id_to_idx)

    num_services = len(config.services)
    gnn_model = NodeGNN(
        num_services=num_services,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    return graph_builder, gnn_model


# ---------------------------------------------------------------------------
# Task Group Encoder
# ---------------------------------------------------------------------------






