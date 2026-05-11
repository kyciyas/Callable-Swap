import cupy as cp

class GPULMM_LSMPricer:
    def __init__(self, lmm_paths, dt, strike):
        if isinstance(lmm_paths, list):
            self.paths = cp.stack(lmm_paths).astype(cp.float32)
        else:
            self.paths = cp.asarray(lmm_paths, dtype=cp.float32)

        self.dt = cp.float32(dt)
        self.strike = cp.float32(strike)

        s = self.paths.shape
        self.n_steps = int(s[0])
        self.n_paths = int(s[1])
        self.n_rates = int(s[2])

    def compute_swap_matrix(self):
        p_idx = int(self.n_paths)
        s_idx = int(self.n_steps)

        swap_values = cp.zeros((p_idx, s_idx), dtype=cp.float32)

        for t in range(s_idx):
            fwd_curve = self.paths[t]
            payoff_t = cp.zeros((p_idx,), dtype=cp.float32)
            df_cum = cp.ones((p_idx,), dtype=cp.float32)

            max_j = self.n_rates - t
            for j in range(max_j):
                df_cum /= (1.0 + fwd_curve[:, j] * self.dt)
                payoff_t += (fwd_curve[:, j] - self.strike) * self.dt * df_cum
            swap_values[:, t] = payoff_t

        return swap_values

    def run_lsm(self, exercise_steps):
        swap_matrix = self.compute_swap_matrix()
        p_num = int(self.n_paths)

        cashflows = cp.zeros((p_num,), dtype=cp.float32)
        cashflows[:] = swap_matrix[:, -1]

        for t in sorted(exercise_steps, reverse=True):
            if t >= self.n_steps - 1:
                continue

            df = 1.0 / (1.0 + self.paths[t, :, 0] * self.dt)
            cashflows *= df

            exercise_value = swap_matrix[:, t]
            itm_mask = exercise_value > 0

            if int(cp.sum(itm_mask).item()) > 10:
                X = self.paths[t, itm_mask, 0]
                Y = cashflows[itm_mask]

                A = cp.vander(X, 3)
                try:
                    coeffs = cp.linalg.lstsq(A, Y, rcond=None)[0]
                    continuation_value = A @ coeffs

                    should_ex = exercise_value[itm_mask] > continuation_value
                    itm_indices = cp.where(itm_mask)[0]
                    exercise_indices = itm_indices[should_ex]

                    cashflows[exercise_indices] = exercise_value[itm_mask][should_ex]
                except:
                    continue

        final_df = 1.0 / (1.0 + self.paths[0, :, 0] * self.dt)
        result = cp.mean(cashflows * final_df)

        return cp.maximum(cp.float32(0.0), result)
