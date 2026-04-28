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
        # We can use mask and gather or argmax on cumulative validity
        # rho_idx shape: (Batch,)
        rho_idx = torch.sum(valid.float(), dim=1).long() - 1
        
        # Selected thresholds
        chosen_theta = torch.gather(theta, 1, rho_idx.unsqueeze(1))
        
        return torch.clamp(v - chosen_theta, min=0.0)

    def solve(self, G, Z, f_min, f_max):
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
        
        for _ in range(self.max_iter):
            z_prev = z.clone()
            
            # 1. f-update: Analytical solution to (G*f - Z*f^2) + rho/2 * ||f - z + u||^2
            # Derivative w.r.t f: G - 2*Z*f + rho*(f - z + u) = 0
            # f * (rho + 2*Z) = G + rho*(z - u)
            f = (G + rho * (z - u)) / (rho + 2 * Z)
            
            # 2. z-update: Simplex projection
            z_tilde = f + u
            z_shift = z_tilde - f_min
            budgets = (self.f_max_node - f_min.sum(dim=1, keepdim=True)).clamp(min=0.0)
            
            z_proj = self.project_simplex(z_shift, budgets)
            z = torch.clamp(z_proj + f_min, max=f_max)
            
            # 3. u-update: Dual update
            u = u + (f - z)
            
            # 4. Residual check for convergence
            res_r = torch.norm(f - z, dim=1)
            res_s = torch.norm(rho * (z - z_prev), dim=1)
            
            if res_r.max() < self.tol and res_s.max() < self.tol:
                break
                
        return z
