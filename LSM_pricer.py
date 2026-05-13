import cupy as cp

class GPULSMPricer:
    def __init__(self, paths_gpu, bwd_gpu, dt, strike):
        self.paths = cp.asarray(paths_gpu, dtype=cp.float32)
        self.dt = cp.float32(dt)
        self.strike = cp.float32(strike)
        self.bwd_gpu = cp.asarray(bwd_gpu, dtype=cp.float32)

    def compute_swap_matrix_gpu(self):
        p_count = int(self.paths.shape[0])
        s_count = int(self.paths.shape[1])

        swap_values = cp.zeros((p_count, s_count), dtype=cp.float32)

        for t in range(s_count - 2, -1, -1):
            payoff = (self.paths[:, t + 1] - self.strike) * self.dt
            df = self.bwd_gpu[t + 1] / self.bwd_gpu[t]
            swap_values[:, t] = payoff + df * swap_values[:, t + 1]
        return swap_values

    def run_lsm_gpu(self, exercise_steps):
        swap_matrix = self.compute_swap_matrix_gpu()
        p_count = int(self.paths.shape[0])
        s_count = int(self.paths.shape[1])

        cashflows = cp.copy(swap_matrix[:, -1])

        current_t = s_count - 1

        for t in sorted(exercise_steps, reverse=True):
            if t >= self.paths.shape[1] - 1 or t == 0: continue

            df = self.bwd_gpu[current_t] / self.bwd_gpu[t]
            cashflows *= df
            current_t = t

            exercise_value = swap_matrix[:, t]
            itm_mask = exercise_value > 0

            if int(cp.sum(itm_mask).get()) > 10:
                X = self.paths[itm_mask, t]
                A = cp.vander(X, 3)
                Y = cashflows[itm_mask]

                try:
                    coeffs = cp.linalg.lstsq(A, Y, rcond=None)[0]
                    cont_val = A @ coeffs
                    ex_val = exercise_value[itm_mask]

                    should_ex = ex_val > cont_val
                    itm_indices = cp.where(itm_mask)[0]
                    exercise_indices = itm_indices[should_ex]
                    cashflows[exercise_indices] = ex_val[should_ex]
                except:
                    continue
        return cp.mean(cashflows * (self.bwd_gpu[current_t] / self.bwd_gpu[0]))