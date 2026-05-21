import torch

class KKTSolverADMM:
    def __init__(self, f_max_node, rho=1.0, max_iter=100, tol=1e-4):
        self.f_max_node = f_max_node
        self.rho_base = rho
        self.max_iter = max_iter
        self.tol = tol

    def project_simplex(self, v, budgets):
        """
        Projection of multiple vectors onto their respective simplexes: sum(v_i) <= budget_i, v_i >= 0.
        v: (Batch, N)
        budgets: (Batch, 1)
        """
        v_clipped = torch.clamp(v, min=0.0)
        sums = v_clipped.sum(dim=1, keepdim=True)

        # Nodes that already satisfy the budget constraint
        within_budget = sums <= budgets
        if within_budget.all():
            return v_clipped

        # For nodes exceeding budget, project onto the sum(z) = budget plane
        # Algorithm: Duchi et al. (2008) "Efficient Projections onto the L1-Ball for Learning in High Dimensions"
        mu, _ = torch.sort(v, dim=1, descending=True)
        cum_sum = torch.cumsum(mu, dim=1)

        # (Batch, N)
        idx = torch.arange(1, v.shape[1] + 1, device=v.device).float()

        # Condition: mu_i - (cum_sum_i - budget) / i > 0
        theta = (cum_sum - budgets) / idx
        valid = mu > theta

        # Get the largest index i that satisfies the condition
        # Clamp to 0 to avoid index -1 when budgets are 0 or precision causes no matches
        rho_idx = (torch.sum(valid.float(), dim=1).long() - 1).clamp(min=0)

        # Selected thresholds
        chosen_theta = torch.gather(theta, 1, rho_idx.unsqueeze(1))

        return torch.clamp(v - chosen_theta, min=0.0)

    def solve(self, G, Z, f_min, f_max, debug=False):
        """
        Vectorized ADMM solver for all nodes simultaneously.
        G: (M, S) - Backlog weights
        Z: (M, S) - Energy weights
        f_min, f_max: (M, S) - Box constraints
        """
        M, S = G.shape
        device = G.device

        # Initialize variables
        f = torch.zeros((M, S), device=device)
        z = torch.zeros((M, S), device=device)
        u = torch.zeros((M, S), device=device)

        # Adaptive rho initialization
        rho = 2 * Z.mean(dim=1, keepdim=True).clamp(min=1e-4) # (M, 1)

        # Pre-calculate budgets (Constant within solve call)
        budgets = (self.f_max_node - f_min.sum(dim=1, keepdim=True)).clamp(min=0.0)

        # Pre-calculate denominator for f-update
        denom = rho + 2 * Z
        
        checkpoints = [15, 30, 50, 70, 100] if debug else []
        max_it = 100 if debug else self.max_iter

        for i in range(max_it):
            z_prev = z.clone()

            # 1. f-update (KKT of local subproblem)
            f = (G + rho * (z - u)) / denom

            # 2. z-update: Simplex projection (Consensus and constraints)
            z_proj = self.project_simplex(f + u - f_min, budgets)
            z = torch.clamp(z_proj + f_min, max=f_max)

            # 3. u-update: Dual variable (Lagrange multipliers)
            u = u + (f - z)

            # 4. Residual and Objective Tracking
            # Optim: Avoid .item() (CPU sync) in every iteration. Only check every 10 steps.
            if i % 10 == 0 or i == max_it - 1:
                res_r = torch.norm(f - z, dim=1).max().item()
                res_s = torch.norm(rho * (z - z_prev), dim=1).max().item()
                
                if debug and (any(i + 1 == cp for cp in checkpoints)):
                    # Calculate Current Objective: Maximize sum(G*z - Z*z^2)
                    obj_val = (G * z - Z * (z**2)).sum().item()
                    print(f"  [Checkpoint {i+1:3d}] Objective: {obj_val:12.4f} | Prim Res: {res_r:.2e} | Dual Res: {res_s:.2e}")

                if not debug and (res_r < self.tol and res_s < self.tol):
                    break
        
        if debug:
            final_obj = (G * z - Z * (z**2)).sum().item()
            print(f"ADMM Final (Iter {i+1}): Objective: {final_obj:12.4f}")

        return z