import cupy as cp

class GPULMMPricer:
    def __init__(self, initial_forwards, n_paths=100000, horizon=5.0, dt=0.25, sigma=0.3):
        self.n_paths = int(n_paths)
        self.dt = cp.float32(dt)
        self.sigma = cp.float32(sigma)
        self.f0 = cp.asarray(initial_forwards, dtype=cp.float32)
        self.n_rates = len(self.f0)
        self.n_steps = int(horizon / dt)

        # rho = exp(-beta * |i-j|)
        self.rho = self._build_correlation_matrix(beta=0.1)
        self.L = cp.linalg.cholesky(self.rho)

    def _build_correlation_matrix(self, beta):
        grid = cp.arange(self.n_rates)
        i, j = cp.meshgrid(grid, grid)
        corr = cp.exp(-beta * cp.abs(i - j))
        return corr.astype(cp.float32)

    def generate_lmm_paths(self):
        cp.get_default_memory_pool().free_all_blocks()
        F = cp.tile(self.f0, (self.n_paths, 1)).astype(cp.float32)
        all_paths = []

        Z_all = cp.random.standard_normal((self.n_steps, self.n_paths, self.n_rates), dtype=cp.float32)

        for t_step in range(self.n_steps):
            Z = Z_all[t_step]
            val_to_sum = (self.dt * F) / (1.0 + self.dt * F)
            drift_sum = cp.cumsum(val_to_sum, axis=1)
            drift = F * (self.sigma ** 2) * drift_sum
            F = F * cp.exp((drift - 0.5 * self.sigma ** 2) * self.dt + self.sigma * cp.sqrt(self.dt) * Z)
            all_paths.append(F.copy())

        return all_paths