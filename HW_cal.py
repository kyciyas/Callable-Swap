import cupy as cp
import numpy as np


class GPUBatchCalibrator:
    def __init__(self, engine, market_rate, hw_input, hw_ois, strike=0.035, init_a=0.1, init_s=0.01):
        self.engine = engine
        self.market_rate = cp.array(market_rate, dtype=cp.float32)
        self.hw_input = hw_input
        self.hw_ois = hw_ois
        self.strike = strike
        self.init_a = init_a
        self.init_s = init_s

        self.max_iter = 1000

    def calculate_prices_gpu(self, batch_paths):
        r_at_expiry = batch_paths[:, :, -1]
        payoffs = cp.maximum(r_at_expiry - self.strike, 0)
        return cp.mean(payoffs, axis=1)

    def compute_gradient_batch(self, a, sigma):
        eps = 1e-4
        a_batch = [a, a + eps, a]
        sigma_batch = [sigma, sigma, sigma + eps]

        # path and price
        batch_paths = self.engine.generate_batch_paths(a_batch, sigma_batch)
        prices = self.calculate_prices_gpu(batch_paths)

        p_base = float(prices[0])
        p_a_eps = float(prices[1])
        p_s_eps = float(prices[2])

        grad_a = (p_a_eps - p_base) / eps
        grad_s = (p_s_eps - p_base) / eps

        return p_base, grad_a, grad_s

    def run_optimization(self):
        a = float(self.init_a)
        s = float(self.init_s)
        target = float(self.market_rate[0])
        lr = 0.5

        print(f"\n[CALIBRATION] Starting... Target: {target:.6f}, Initial Sigma: {s:.6f}")

        for i in range(self.max_iter):
            price_base, ga, gs = self.compute_gradient_batch(a, s)
            error = price_base - target

            a -= lr * error * ga
            s -= lr * error * gs

            a = max(a, 0.001)
            s = max(s, 0.0001)

            print(f" Iter {i + 1:2d}: a={a:.4f}, s={s:.4f}, Error={error:.6f}")
            if abs(error) < 1e-6: break

        return np.array([a, s])
