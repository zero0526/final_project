import torch
import torch.nn as nn

from matrix_source.utils.neural import _make_branch

class TaskGroupEncoder(nn.Module):
    """
    Encodes a *group* of tasks from one timeslot into a single fixed-size
    embedding, summarising the batch for a centralised planner / upper agent.

    Inputs (from MatrixWorkloadGenerator.generate_step):
      task_batch_sizes   (N,)     – input data sizes (proxy for network load)
      tasks_min_accuracy (N,)     – min accuracy required → mapped to workload
      task_deadlines     (N,)     – per-task latency deadline
      svc_indices        (N,)     – long, which service each task belongs to
      svc_omega          (|S|,)   – omega flag (0 → latency-sensitive, 1 → throughput)
      svc_workloads      (|S|, M) – workload LUT: workload per model per service

    Feature groups and statistics computed (each has 5 dims: mean,std,min,max,sum):
      Branch 1 – Workload requirement  : derived as min_workload × min_accuracy
      Branch 2 – Deadline
      Branch 3 – Data size

    Context vector (3 dims):
      log(1 + N)        – number of tasks (log-normalised)
      omega_frac_0      – fraction of tasks with omega=0 (latency-sensitive)
      omega_frac_1      – fraction of tasks with omega=1 (throughput-oriented)

    Total fusion input = 3 × branch_dim + 3
    Output = output_dim-dimensional task-group embedding.

    Parameters
    ----------
    branch_dim  : int   output dim of each branch (default 32)
    output_dim  : int   final embedding size       (default 64)
    dropout     : float
    """

    STATS_DIM = 5  # mean, std, min, max, sum

    def __init__(
        self,
        branch_dim: int = 32,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Three specialised branches, each receiving 5-dim statistics
        self.branch_workload = _make_branch(self.STATS_DIM, branch_dim, branch_dim * 2)
        self.branch_deadline = _make_branch(self.STATS_DIM, branch_dim, branch_dim * 2)
        self.branch_datasize = _make_branch(self.STATS_DIM, branch_dim, branch_dim * 2)

        # Context: num_tasks(1) + omega_frac_0(1) + omega_frac_1(1) = 3 dims
        context_dim = 3
        fusion_in = branch_dim * 3 + context_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ELU(),
        )
        self.output_dim = output_dim

    # ------------------------------------------------------------------
    @staticmethod
    def _workload_reqs(
        svc_indices: torch.Tensor,          # (N,)  long
        tasks_min_accuracy: torch.Tensor,   # (N,)  float – [0, 1] scale
        svc_workloads: torch.Tensor,        # (|S|, M)  float – GFLOPS
    ) -> torch.Tensor:
        """
        For each task, estimate the compute requirement as:
            min_workload_of_service × tasks_min_accuracy

        This approximates "the cheapest model that meets the accuracy bar".

        Returns : (N,) float
        """
        wl_table = svc_workloads[svc_indices]               # (N, M)
        # Minimum non-zero workload per task (lightest valid model)
        wl_table_safe = wl_table.clone()
        wl_table_safe[wl_table_safe <= 0] = float('inf')
        min_wl = wl_table_safe.min(dim=1).values            # (N,)
        min_wl = torch.where(min_wl == float('inf'),
                             torch.zeros_like(min_wl),
                             min_wl)
        return min_wl * tasks_min_accuracy.clamp(min=0.0)   # (N,)

    # ------------------------------------------------------------------

    def forward(
        self,
        task_batch_sizes: torch.Tensor,     # (N,)
        tasks_min_accuracy: torch.Tensor,   # (N,)
        task_deadlines: torch.Tensor,       # (N,)
        svc_indices: torch.Tensor,          # (N,)  long
        svc_omega: torch.Tensor,            # (|S|,)
        svc_workloads: torch.Tensor,        # (|S|, M)
    ) -> torch.Tensor:
        """
        Returns
        -------
        embedding : (output_dim,)  – one vector representing the whole task group.
        """
        N = float(task_batch_sizes.shape[0])
        device = task_batch_sizes.device

        # ---- 1. Derived workload signal ----------------------------------
        wl_req = self._workload_reqs(svc_indices, tasks_min_accuracy, svc_workloads)

        # ---- 2. Compute 5-dim statistics for each group -----------------
        stats_wl = _stats5(wl_req)
        stats_dl = _stats5(task_deadlines)
        stats_ds = _stats5(task_batch_sizes)

        # ---- 3. Context vector ------------------------------------------
        num_tasks_feat = torch.log1p(torch.tensor(N, device=device, dtype=torch.float32))
        omega_flags    = svc_omega[svc_indices]         # (N,)  0 or 1
        omega_frac_1   = omega_flags.mean()
        omega_frac_0   = 1.0 - omega_frac_1
        context = torch.stack([num_tasks_feat, omega_frac_0, omega_frac_1])  # (3,)

        # ---- 4. Branch encodings ----------------------------------------
        e_wl = self.branch_workload(stats_wl)   # (branch_dim,)
        e_dl = self.branch_deadline(stats_dl)   # (branch_dim,)
        e_ds = self.branch_datasize(stats_ds)   # (branch_dim,)

        # ---- 5. Fusion --------------------------------------------------
        fused = torch.cat([e_wl, e_dl, e_ds, context], dim=0)
        return self.fusion_mlp(fused)            # (output_dim,)
