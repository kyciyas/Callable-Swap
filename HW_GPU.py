import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import cupy as cp

class GPUHullWhitePricer:
    def __init__(self, n_paths=100_000, n_steps=500, T=5.0, a=0.1, sigma=0.01):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.T = T
        self.dt = np.float32(T / n_steps)
        self.a = np.float32(a)
        self.sigma = np.float32(sigma)

        # Random values were generated away from CUDA kernel
        self.mod = SourceModule("""
        #include <math.h>

        extern "C" {
            __global__ void generate_paths(
                float *paths, 
                float *fwd_gpu, 
                float *rand_gpu, 
                float a, float sigma, float dt,
                int n_paths, int n_steps) 
            {
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid >= n_paths) return;

                float current_r = fwd_gpu[0];
                paths[tid * n_steps] = current_r;

                const float s_sq_2a = (sigma * sigma) / (2.0f * a);
                const float sqrt_dt = sqrtf(dt);

                for (int t = 1; t < n_steps; t++) {
                    float z = rand_gpu[tid * n_steps + t];

                    float time_t = t * dt;
                    float theta = (fwd_gpu[t] - fwd_gpu[t-1]) / dt 
                                  + a * fwd_gpu[t] 
                                  + s_sq_2a * (1.0f - expf(-2.0f * a * time_t));

                    current_r += (theta - a * current_r) * dt + sigma * sqrt_dt * z;
                    paths[tid * n_steps + t] = current_r;
                }
            }
        }
        """, no_extern_c=True)

        self.func = self.mod.get_function("generate_paths")

    def generate_paths(self, hw_input, seed=42):
        np.random.seed(seed)
        total_elements = self.n_paths * self.n_steps
        rand_data = np.random.standard_normal(total_elements).astype(np.float32)
        rand_gpu = cuda.to_device(rand_data)

        paths_gpu = cuda.mem_alloc(total_elements * 4)
        fwd_gpu = cuda.to_device(hw_input.astype(np.float32))

        threads_per_block = 256
        blocks_per_grid = (self.n_paths + threads_per_block - 1) // threads_per_block

        try:
            self.func(paths_gpu, fwd_gpu, rand_gpu,
                      self.a, self.sigma, self.dt,
                      np.int32(self.n_paths), np.int32(self.n_steps),
                      block=(threads_per_block, 1, 1), grid=(blocks_per_grid, 1))
        except Exception as e:
            print(f"CUDA Launch Failed: {e}")
            return None

        result_paths = np.empty((self.n_paths, self.n_steps), dtype=np.float32)
        cuda.memcpy_dtoh(result_paths, paths_gpu)

        # Mem clear
        paths_gpu.free()
        fwd_gpu.free()
        rand_gpu.free()

        return result_paths


class GPUOptHWPricer:
    def __init__(self, n_paths=100_000, n_steps=500, T=5.0):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = np.float32(T / n_steps)

        self.kernel = cp.RawKernel(r'''
        extern "C" __global__ void generate_opt_paths(
            float *paths, float *fwd_gpu, float *bwd_gpu, float *rand_gpu, 
            float *a_vec, float *sigma_vec, 
            float dt, int n_paths, int n_steps, int n_scenarios) 
        {
            int path_idx = blockIdx.x * blockDim.x + threadIdx.x;
            int sce_idx = blockIdx.y * blockDim.y + threadIdx.y;

            if (path_idx >= n_paths || sce_idx >= n_scenarios) return;

            float a = a_vec[sce_idx];
            float sigma = sigma_vec[sce_idx];
            float current_r = fwd_gpu[0];

            int base_idx = (sce_idx * n_paths + path_idx) * n_steps;
            paths[base_idx] = current_r;

            const float s_sq_2a = (sigma * sigma) / (2.0f * a);
            const float sqrt_dt = sqrtf(dt);

            for (int t = 1; t < n_steps; t++) {
                float z = rand_gpu[path_idx * n_steps + t];
                float time_t = t * dt;

                float theta_base = (fwd_gpu[t] - fwd_gpu[t-1]) / dt  + a * fwd_gpu[t]  + s_sq_2a * (1.0f - expf(-2.0f * a * time_t));

                float ois_fwd_t = -logf(bwd_gpu[t]) / (t * dt);
                float basis_spread = fwd_gpu[t] - ois_fwd_t; 

                float theta = theta_base + a * basis_spread;

                current_r += (theta - a * current_r) * dt + sigma * sqrt_dt * z;
                paths[base_idx + t] = current_r;
            }
        }
        ''', 'generate_opt_paths')

    def generate_batch_paths(self, hw_input, hw_ois, a_list, sigma_list, seed=42):
        cp.random.seed(seed)

        n_scenarios = len(a_list)
        fwd_gpu = cp.array(hw_input, dtype=cp.float32)
        bwd_gpu = cp.array(hw_ois, dtype=cp.float32)
        a_gpu = cp.array(a_list, dtype=cp.float32)
        sigma_gpu = cp.array(sigma_list, dtype=cp.float32)

        rand_gpu = cp.random.standard_normal(self.n_paths * self.n_steps, dtype=cp.float32)
        paths_gpu = cp.zeros((n_scenarios, self.n_paths, self.n_steps), dtype=cp.float32)

        self.kernel(((self.n_paths + 255) // 256, n_scenarios), (256, 1),
                    (paths_gpu, fwd_gpu, bwd_gpu, rand_gpu, a_gpu, sigma_gpu,
                     self.dt, self.n_paths, self.n_steps, n_scenarios))

        return paths_gpu
