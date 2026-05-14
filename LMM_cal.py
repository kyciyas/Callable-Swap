import cupy as cp
from scipy.optimize import minimize


class GPULMMCalibrator:
    # 1. 생성자(__init__) 단계에서 무위험 할인 커브인 bwd_gpu를 직접 주입받아 바인딩
    def __init__(self, engine, market_rate, f0, bwd_gpu, strike=0.035, dt=0.25):
        self.engine, self.f0 = engine, f0
        self.market_rates = cp.array(market_rate, dtype=cp.float32)
        self.strike = strike
        self.dt = cp.float32(dt)
        # main.py에서 전달된 self.ois_list_lmm 배열을 GPU 메모리(CuPy)로 업로드하여 고정 멤버 변수화
        self.bwd_gpu = cp.array(bwd_gpu, dtype=cp.float32)

    # 2. 내부 가격 평가 함수는 멤버 변수인 self.bwd_gpu를 다이렉트로 참조하여 연산
    def calculate_prices_gpu(self, batch_paths):
        # batch_paths 차원 구조: (n_scenarios, n_steps, n_paths, n_rates)
        n_scenarios = batch_paths.shape[0]
        n_steps = batch_paths.shape[1]
        n_paths = batch_paths.shape[2]

        # 시나리오 및 패스 차원에 매핑되는 가치 누적 버퍼 동적 할당
        total_npv = cp.zeros((n_scenarios, n_paths), dtype=cp.float32)

        for t in range(n_steps):
            # t 타임스텝 시점에 시뮬레이션된 만기(t) 그리드의 선도금리 F 추출
            fwd_rate = batch_paths[:, t, :, t]  # shape: (n_scenarios, n_paths)
            # Floating Leg 현금흐름 연산 = (선도금리 - 행사금리) * dt
            payoff = (fwd_rate - self.strike) * self.dt
            # [Multi-curve 원리] 할인은 멤버 변수에 선언된 국고채 프록시 OIS 할인 인자(self.bwd_gpu)를 결합
            total_npv += payoff * self.bwd_gpu[t + 1]

        # 몬테카를로 패스 평균을 취해 시나리오별 최종 스왑션 가격 산출
        return cp.mean(cp.maximum(total_npv, 0.0), axis=1)

    def run_lmm_optimization(self, init_sigmas, max_iter=10):
        params = cp.array(init_sigmas, dtype=cp.float32)
        n_params, eps, lr = len(params), 1e-3, 0.3
        for i in range(max_iter):
            sig_batch = cp.tile(params, (n_params + 1, 1))
            for j in range(n_params): sig_batch[j+1, j] += eps
            batch_paths = self.engine.generate_lmm_batch_paths(self.f0, sig_batch.get())
            all_prices = self.calculate_prices_gpu(batch_paths)
            errors = all_prices[0] - self.market_rates
            jacobian = (all_prices[1:] - all_prices[0]) / eps
            update = cp.linalg.solve(jacobian + cp.eye(n_params, dtype=cp.float32)*1e-2, errors)
            params -= lr * update
            params = cp.maximum(params, 0.01)
            print(f" LMM Iter {i+1}: RMS Error = {cp.sqrt(cp.mean(errors**2)):.6f}")
            if cp.sqrt(cp.mean(errors**2)) < 1e-5: break
        return params.get()