import cupy as cp

class GPULMMPricer:
    def __init__(self, initial_forwards, n_paths=100000, horizon=5.0, dt=0.25, sigma=0.3):
        self.n_paths = int(n_paths)
        self.dt = cp.float32(dt)
        self.sigma = cp.float32(sigma)
        self.sigma = cp.asarray(sigma, dtype=cp.float32)
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


class GPULMMBatchPricer:
    def __init__(self, n_paths=100000, n_steps=20, n_rates=20, dt=0.25):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.n_rates = n_rates
        self.dt = cp.float32(dt)

        self.kernel = cp.RawKernel(r'''
        extern "C" __global__ void generate_lmm_batch_paths(
            float *paths, const float *f0_gpu, const float *rand_gpu, const float *sigma_matrix, 
            float dt, int n_paths, int n_steps, int n_rates, int n_scenarios) 
        {
            int p_idx = blockIdx.x * blockDim.x + threadIdx.x;
            int s_idx = blockIdx.y;
            if (p_idx >= n_paths || s_idx >= n_scenarios) return;
            float F[20]; 
            for(int i=0; i < n_rates; i++) F[i] = f0_gpu[i];
            for (int t = 0; t < n_steps; t++) {
                for (int i = 0; i < n_rates; i++) {
                    float sig = sigma_matrix[s_idx * n_rates + i];
                    float z = rand_gpu[(t * n_paths * n_rates) + (p_idx * n_rates) + i];
                    float drift_sum = 0.0f;
                    for(int j=0; j <= i; j++) drift_sum += (dt * F[j]) / (1.0f + dt * F[j]);
                    float drift = F[i] * (sig * sig) * drift_sum;
                    F[i] = F[i] * expf((drift - 0.5f * sig * sig) * dt + sig * sqrtf(dt) * z);
                    paths[(((s_idx * n_steps + t) * n_paths + p_idx) * n_rates) + i] = F[i];
                }
            }
        }
        ''', 'generate_lmm_batch_paths')

    def generate_lmm_batch_paths(self, f0, sigma_list, seed=42, beta=1.5):
        cp.random.seed(seed)
        n_scenarios = len(sigma_list)
        f0_gpu, sig_gpu = cp.array(f0, dtype=cp.float32), cp.array(sigma_list, dtype=cp.float32)

        tenors = cp.arange(1, self.n_rates + 1) * self.dt
        T_i, T_j = cp.meshgrid(tenors, tenors)
        beta_gpu = cp.array(beta, dtype=cp.float32)
        corr_matrix = cp.exp(-beta_gpu * cp.abs(T_i - T_j))
        corr_matrix += cp.eye(self.n_rates, dtype=cp.float32) * 1e-6
        L_raw = cp.linalg.cholesky(corr_matrix)

        row_norms = cp.sqrt(cp.sum(L_raw ** 2, axis=1, keepdims=True))
        L = L_raw / row_norms
        rand_raw = cp.random.standard_normal((self.n_steps, self.n_paths, self.n_rates), dtype=cp.float32)
        rand_correlated = cp.zeros_like(rand_raw)
        for t in range(self.n_steps):
            rand_correlated[t] = rand_raw[t] @ L.T
        # rand_gpu = cp.einsum('lk,ijk->ijl', L, rand_raw)
        rand_gpu = cp.ascontiguousarray(rand_correlated)
        rand_gpu = rand_gpu.ravel()
        # rand_gpu = cp.random.standard_normal((self.n_steps, self.n_paths, self.n_rates), dtype=cp.float32)
        paths_gpu = cp.zeros((n_scenarios, self.n_steps, self.n_paths, self.n_rates), dtype=cp.float32)
        self.kernel(((self.n_paths+255)//256, n_scenarios), (256, 1),
                    (paths_gpu, f0_gpu, rand_gpu, sig_gpu, self.dt, self.n_paths, self.n_steps, self.n_rates, n_scenarios))
        return paths_gpu

