import cupy as cp

class GPULMM_LSMPricer:
    def __init__(self, lmm_paths, dt, strike, bwd_gpu):
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

        self.bwd_gpu = cp.array(bwd_gpu, dtype=cp.float32).ravel()

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
                # df_cum /= (1.0 + fwd_curve[:, j] * self.dt)
                # payoff_t += (fwd_curve[:, j] - self.strike) * self.dt * df_cum
                df_from_t_to_j = self.bwd_gpu[t + j + 1] / self.bwd_gpu[t]
                payoff_t += (fwd_curve[:, j] - self.strike) * self.dt * df_from_t_to_j

            swap_values[:, t] = payoff_t

        return swap_values

    def run_lsm(self, exercise_steps):
        swap_matrix = self.compute_swap_matrix()
        p_num = int(self.n_paths)

        cashflows = cp.zeros((p_num,), dtype=cp.float32)
        cashflows[:] = swap_matrix[:, -1]

        sorted_steps = sorted(exercise_steps, reverse=True)

        for i in range(len(sorted_steps)):
            t_curr = sorted_steps[i]
            if t_curr >= self.n_steps - 1:
                continue

            # 역방향으로 건너뛸 차기 조기행사 시점을 추적
            t_next = sorted_steps[i - 1] if i > 0 else (self.n_steps - 1)

            ########################################################################
            # (기존) df = 1.0 / (1.0 + self.paths[t, :, 0] * self.dt) 식의 억지 할인 제거
            # (변경) t_next 시점 현금흐름을 t_curr 시점으로 당기는 순수한 OIS 무위험 구간 할인율 매핑
            df_between_steps = self.bwd_gpu[t_next + 1] / self.bwd_gpu[t_curr + 1]
            cashflows *= df_between_steps
            ########################################################################

            exercise_value = swap_matrix[:, t_curr]
            itm_mask = exercise_value > 0

            if int(cp.sum(itm_mask).item()) > 10:
                X = self.paths[t_curr, itm_mask, 0]
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

        ########################################################################
        # (기존) final_df = 1.0 / (1.0 + self.paths[0, :, 0] * self.dt) 식의 단일커브 최종할인 제거
        # (변경) 최초 조기행사 스텝(t_first)에 정체되어 있는 가치를 최종 현재가치(시점 0)로 완전히 복원
        t_first = sorted_steps[-1]
        final_df = self.bwd_gpu[t_first + 1] / self.bwd_gpu[0]
        result = cp.mean(cashflows * final_df)
        ########################################################################

        return cp.maximum(cp.float32(0.0), result)
