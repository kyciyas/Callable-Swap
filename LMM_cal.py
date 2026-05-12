import cupy as cp

class GPULMMCalibrator:
    def __init__(self, engine, target_prices, f0, strike = 0.035):
        self.engine, self.f0 = engine, f0
        self.target_prices = cp.array(target_prices, dtype=cp.float32)
        self.strike = strike

    def calculate_prices_gpu(self, batch_paths):
        r_final = batch_paths[:, -1, :, :]
        return cp.mean(cp.maximum(r_final - self.strike, 0), axis=1)

    def run_lmm_optimization(self, init_sigmas, max_iter=10):
        params = cp.array(init_sigmas, dtype=cp.float32)
        n_params, eps, lr = len(params), 1e-3, 0.3
        for i in range(max_iter):
            sig_batch = cp.tile(params, (n_params + 1, 1))
            for j in range(n_params): sig_batch[j+1, j] += eps
            batch_paths = self.engine.generate_lmm_batch_paths(self.f0, sig_batch.get())
            all_prices = self.calculate_prices_gpu(batch_paths)
            errors = all_prices[0] - self.target_prices
            jacobian = (all_prices[1:] - all_prices[0]) / eps
            update = cp.linalg.solve(jacobian + cp.eye(n_params, dtype=cp.float32)*1e-2, errors)
            params -= lr * update
            params = cp.maximum(params, 0.01)
            print(f" LMM Iter {i+1}: RMS Error = {cp.sqrt(cp.mean(errors**2)):.6f}")
            if cp.sqrt(cp.mean(errors**2)) < 1e-5: break
        return params.get()