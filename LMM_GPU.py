import cupy as cp
import numpy as np
class GPULMMPricer:
    def __init__(self, fwd_gpu, bwd_gpu, beta=0.1, n_paths=100000, horizon=5.0, dt=0.25, sigma=0.3):
        self.n_paths = int(n_paths)
        self.dt = cp.float32(dt)
        sigma = self.convert_garch_to_pure_lmm_vol(fwd_gpu, sigma)
        self.sigma = cp.float32(sigma)
        self.sigma = cp.asarray(sigma, dtype=cp.float32)
        self.f0 = cp.asarray(fwd_gpu, dtype=cp.float32)
        self.n_rates = len(self.f0)
        self.n_steps = int(horizon / dt)
        self.bwd_gpu = cp.asarray(bwd_gpu, dtype=cp.float32)
        self.beta = beta


        # rho = exp(-beta * |i-j|)
        self.rho = self._build_correlation_matrix()
        self.L = cp.linalg.cholesky(self.rho)

    def convert_garch_to_pure_lmm_vol(self, fwd_gpu, sigma):
        n_rates = len(fwd_gpu)  # 테너 개수 (20개)
        pure_sigma_lmm = []

        k = 0.12
        for i in range(n_rates):
            time_to_maturity = (i + 1) * self.dt

            vol_i = 0.27 * sigma * np.exp(-k * time_to_maturity)
            pure_sigma_lmm.append(float(vol_i))
        print(pure_sigma_lmm)
        return pure_sigma_lmm

    def _build_correlation_matrix(self):
        grid = cp.arange(self.n_rates)
        i, j = cp.meshgrid(grid, grid)
        corr = cp.exp(-self.beta * cp.abs(i - j))
        return corr.astype(cp.float32)

    def generate_lmm_paths(self):
        cp.get_default_memory_pool().free_all_blocks()
        F = cp.tile(self.f0, (self.n_paths, 1)).astype(cp.float32)
        all_paths = []

        Z_all = cp.random.standard_normal((self.n_steps, self.n_paths, self.n_rates), dtype=cp.float32)

        for t_step in range(self.n_steps):
            Z = Z_all[t_step]

            # [Multi-curve 적용 핵심 1] OIS 할인인자로부터 t_step 시점의 무위험 선도금리(OIS Forward) 동적 추출
            # P(t, T_i) / P(t, T_{i+1}) 관계식을 이용해 단리(Simple) 기준의 OIS Forward 계산
            df_current = self.bwd_gpu[t_step]
            df_next = self.bwd_gpu[t_step + 1]
            ois_fwd = (df_current / df_next - 1.0) / self.dt

            # [Multi-curve 적용 핵심 2] 전통적인 LMM 자기 할인 성분 계산
            val_to_sum = (self.dt * F) / (1.0 + self.dt * F)
            drift_sum = cp.cumsum(val_to_sum, axis=1)

            # 기본 LMM 드리프트 결합 (상관관계 Cholesky L을 고려하여 가중치 제어 가능)
            base_drift = F * (self.sigma ** 2) * drift_sum

            # [Multi-curve 적용 핵심 3] 시장 인덱스 선도금리(F)와 무위험 OIS 선도금리 간의 베이시스 스프레드 보정 항 가산
            # 무위험 자산의 드리프트 척도(OIS-measure) 하에서 자산이 확산되도록 보정치 유도
            basis_spread = F - ois_fwd
            multi_curve_drift = base_drift + (self.sigma * basis_spread)  # 모델 캘리브레이션 척도에 맞춘 스프레드 조정

            # 3. 보정된 다중 커브 드리프트를 사용하여 차기 시점 선도금리 확산 연산
            F = F * cp.exp((multi_curve_drift - 0.5 * self.sigma ** 2) * self.dt + self.sigma * cp.sqrt(self.dt) * Z)
            all_paths.append(F.copy())

        return all_paths


class GPULMMBatchPricer:
    def __init__(self, bwd_gpu, n_paths=100000, n_steps=20, n_rates=20, dt=0.25):
        self.bwd_gpu = cp.asarray(bwd_gpu, dtype=cp.float32)
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self.n_rates = int(n_rates)
        self.dt = cp.float32(dt)

        # 고정 배열(float F[20]) 하드코딩을 완벽히 제거하고 글로벌 메모리 오프셋으로 연산하는 CUDA 커널
        self.kernel = cp.RawKernel(r'''
        extern "C" __global__ void generate_lmm_batch_paths(
            float *paths, const float *f0_gpu, const float *rand_gpu, const float *sigma_matrix, 
            const float *ois_discount, float dt, int n_paths, int n_steps, int n_rates, int n_scenarios) 
        {
            int p_idx = blockIdx.x * blockDim.x + threadIdx.x;
            int s_idx = blockIdx.y;
            if (p_idx >= n_paths || s_idx >= n_scenarios) return;

            // 대형 매트릭스 연산 시 32비트 정수 오버플로우 방지를 위한 64비트 오프셋 계산
            long long thread_base_idx = ((long long)s_idx * n_steps * n_paths + p_idx) * n_rates;

            for (int t = 0; t < n_steps; t++) {
                // [Multi-curve 원리] 국고채 프록시 OIS 할인 인자로부터 해당 타임스텝의 무위험 선도금리 역산
                float df_current = ois_discount[t];
                float df_next = ois_discount[t + 1];
                float ois_fwd = (df_current / df_next - 1.0f) / dt;

                for (int i = 0; i < n_rates; i++) {
                    float sig = sigma_matrix[s_idx * n_rates + i];
                    float z = rand_gpu[((long long)t * n_paths * n_rates) + ((long long)p_idx * n_rates) + i];

                    // 직전 스텝(t-1)의 값을 가져옵니다. 최초 스텝(t=0)일 때는 입력 데이터 f0_gpu를 참조합니다.
                    float f_old_i = (t == 0) ? f0_gpu[i] : paths[thread_base_idx + ((long long)(t - 1) * n_paths * n_rates) + i];

                    // 고정 크기 배열 제한을 깨부순 동적 적립식 드리프트 합산 루프
                    float drift_sum = 0.0f;
                    for (int j = 0; j <= i; j++) {
                        float f_old_j = (t == 0) ? f0_gpu[j] : paths[thread_base_idx + ((long long)(t - 1) * n_paths * n_rates) + j];
                        drift_sum += (dt * f_old_j) / (1.0f + dt * f_old_j);
                    }

                    // 기존 LMM 고유 누적 드리프트 성분
                    float base_drift = f_old_i * (sig * sig) * drift_sum;

                    // [Multi-curve 핵심] 변동금리 인덱스 선도금리와 무위험 OIS 선도금리 간의 베이시스 스프레드 보정 항 가산
                    float basis_spread = f_old_i - ois_fwd;
                    float multi_curve_drift = base_drift + (sig * basis_spread);

                    // 하드코딩 버퍼 없이 최종 연산 결과를 출력 매트릭스의 정확한 글로벌 주소에 다이렉트로 할당
                    long long write_idx = ((long long)(s_idx * n_steps + t) * n_paths + p_idx) * n_rates + i;
                    paths[write_idx] = f_old_i * expf((multi_curve_drift - 0.5f * sig * sig) * dt + sig * sqrtf(dt) * z);
                }
            }
        }
        ''', 'generate_lmm_batch_paths')

    def generate_lmm_batch_paths(self, f0, sigma_list, seed=42, beta=1.5):
        cp.random.seed(seed)
        n_scenarios = len(sigma_list)

        # 입력 데이터를 GPU 메모리(VRAM)로 안전하게 캐스팅하여 업로드
        f0_gpu = cp.array(f0, dtype=cp.float32)
        sig_gpu = cp.array(sigma_list, dtype=cp.float32)
        ois_discount_gpu = cp.array(self.bwd_gpu, dtype=cp.float32)

        # 상관관계 행렬(Cholesky Decomposition) 빌드 파트
        tenors = cp.arange(1, self.n_rates + 1) * self.dt
        T_i, T_j = cp.meshgrid(tenors, tenors)
        beta_gpu = cp.array(beta, dtype=cp.float32)
        corr_matrix = cp.exp(-beta_gpu * cp.abs(T_i - T_j))
        corr_matrix += cp.eye(self.n_rates, dtype=cp.float32) * 1e-6
        L_raw = cp.linalg.cholesky(corr_matrix)

        row_norms = cp.sqrt(cp.sum(L_raw ** 2, axis=1, keepdims=True))
        L = L_raw / row_norms

        # 몬테카를로 표준정규난수 생성 및 상관관계 맵핑 연산
        rand_raw = cp.random.standard_normal((self.n_steps, self.n_paths, self.n_rates), dtype=cp.float32)
        rand_correlated = cp.zeros_like(rand_raw)
        for t in range(self.n_steps):
            rand_correlated[t] = rand_raw[t] @ L.T

        rand_gpu = cp.ascontiguousarray(rand_correlated).ravel()

        # 하드코딩 제약이 풀린 4차원 Tensor 형태의 출력 버퍼 동적 할당
        paths_gpu = cp.zeros((n_scenarios, self.n_steps, self.n_paths, self.n_rates), dtype=cp.float32)

        # 블록 크기 256 기반 최적 파티셔닝 레이아웃으로 CUDA 그리드 실행
        grid_x = (self.n_paths + 255) // 256
        self.kernel(
            (grid_x, n_scenarios), (256, 1),
            (paths_gpu, f0_gpu, rand_gpu, sig_gpu, ois_discount_gpu, self.dt, self.n_paths, self.n_steps, self.n_rates,
             n_scenarios)
        )

        return paths_gpu
