import numpy as np
import logging

logger = logging.getLogger(__name__)

class KKTSolverADMM:
    def __init__(self, f_max_node, rho=1.0, max_iter=100, tol=1e-4):
        self.f_max_node = f_max_node
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol

    def project_simplex(self, v, z_max):
        """Projection onto simplex: sum(v)=z_max, v>=0"""
        if z_max <= 0:
            return np.zeros_like(v)

        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        idx = np.where(u * np.arange(1, len(v)+1) > (cssv - z_max))[0]

        if len(idx) == 0:
            return np.zeros_like(v)

        rho = idx[-1]
        theta = (cssv[rho] - z_max) / (rho + 1)
        return np.maximum(v - theta, 0)

    def solve(self, G, Z, f_min_vec, f_max_vec):
        n = len(G)
        Z_safe = np.maximum(Z, 1e-4)

        # check feasibility
        if np.sum(f_min_vec) > self.f_max_node:
            raise ValueError("Infeasible: sum(f_min) > f_max_node")

        # init
        f = np.zeros(n)
        z = np.zeros(n)
        u = np.zeros(n)

        # init rho theo scale bài toán
        self.rho = 2 * np.mean(Z_safe)

        for _ in range(self.max_iter):
            z_prev = z.copy()

            # ===== f-update =====
            f = (G + self.rho * (z - u)) / (2 * Z_safe + self.rho)

            # ===== z-update (projection đúng) =====
            z_tilde = f + u

            # shift về simplex
            z_shift = z_tilde - f_min_vec
            budget = self.f_max_node - np.sum(f_min_vec)

            z_proj = self.project_simplex(z_shift, budget)

            z = z_proj + f_min_vec

            # enforce upper bound
            z = np.minimum(z, f_max_vec)

            # ===== dual update =====
            u = u + (f - z)

            # ===== residual =====
            r = f - z
            s = self.rho * (z - z_prev)

            r_norm = np.linalg.norm(r)
            s_norm = np.linalg.norm(s)

            # ===== adaptive rho =====
            if r_norm > 10 * s_norm:
                self.rho *= 2
                u /= 2
            elif s_norm > 10 * r_norm:
                self.rho /= 2
                u *= 2

            # ===== stopping =====
            if r_norm < self.tol and s_norm < self.tol:
                break

        return z

class EMA:
    def __init__(self, init_step=0):
        self.step= init_step

    def update(self, prev: float, curr: float):
        if self.step<0:
            raise ValueError('n must be non-negative')
        alpha= 1/(2+self.step)
        self.step+=1
        return prev*(1-alpha) + curr*alpha

