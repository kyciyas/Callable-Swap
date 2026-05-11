import Datahandler
import Model_selection
import HW_GPU
import LMM_GPU
import LSM_pricer
import LSM_pricer_LMM
import cupy as cp
import numpy as np
import pandas as pd
import time
import gc


def run_valuation_pipeline(country_code, api_key=None):
    print(f"\n[INFO] {country_code} market engine is running...")

    handler = Datahandler.Datahandler(country=country_code)
    rates = handler.fetch_from_ecos(api_key=api_key) if country_code == "KR" else handler.fetch_from_yfinance()
    engine = Model_selection.InterestRateDataEngine(rates, handler.calendar, handler.settlement_days)

    hw_init = engine.get_hull_white_input(T=5.0, n_steps=500)
    lmm_init = engine.get_lmm_input(horizon=5.0, dt=0.25)

    strike, bump = 0.035, 0.0001

    def calculate_metrics(model_type, init_rates, sigma_val):
        results = {}
        # Base, Up, Dn, Vega 시나리오
        scenarios = {'Base': 0, 'Up': bump, 'Dn': -bump, 'Vega': 0}

        for name, shift in scenarios.items():
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()

            curr_sigma = (sigma_val + 0.01) if name == 'Vega' else sigma_val
            curr_rates = init_rates + (shift if name != 'Vega' else 0)

            if model_type == "HW":
                pricer = HW_GPU.GPUHullWhitePricer(n_paths=100000, sigma=float(curr_sigma))
                raw_paths = pricer.generate_paths(curr_rates)
                paths_gpu = cp.asarray(raw_paths, dtype=cp.float32)
                lsm = LSM_pricer.GPULSMPricer(paths_gpu, 5.0 / 500, strike)
                val = lsm.run_lsm_gpu(exercise_steps=[int((y / 5.0) * 500) for y in [1, 2, 3, 4]])

                del paths_gpu, raw_paths, pricer, lsm
            else:
                pricer = LMM_GPU.GPULMMPricer(curr_rates, n_paths=100000, sigma=float(curr_sigma))
                lmm_paths = pricer.generate_lmm_paths()
                lsm = LSM_pricer_LMM.GPULMM_LSMPricer(lmm_paths, 0.25, strike)
                val = lsm.run_lsm(exercise_steps=[4, 8, 12, 16])

                del lmm_paths, pricer, lsm

            results[name] = float(val.get())

            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        b, u, d, v = results['Base'], results['Up'], results['Dn'], results['Vega']
        delta = (u - d) / (2 * bump) * bump * 100
        gamma = (u - 2 * b + d) / (bump ** 2) * 0.0001
        vega = (v - b) * 100
        hr = delta / 0.045  # Hedge Ratio (Vanilla Delta 4.5bp 가정)

        return {"Val": b * 100, "Delta": delta, "Gamma": gamma, "Vega": vega, "HR": hr}

    hw_metrics = calculate_metrics("HW", hw_init, 0.01)
    lmm_metrics = calculate_metrics("LMM", lmm_init, 0.3)

    return {"HW": hw_metrics, "LMM": lmm_metrics}

if __name__ == "__main__":
    start_all = time.time()
    ecos_key = ""

    final_res = {
        "KOREA": run_valuation_pipeline("KR", ecos_key),
        "USA": run_valuation_pipeline("US")
    }

    for country in ["KOREA", "USA"]:
        res = final_res[country]
        hw, lmm = res["HW"], res["LMM"]

        print(f"\n{'=' * 30} {country} Integrated Risk Metrics Report {'=' * 30}")
        print(f"{'Metric':<12} | {'HW Model':<15} | {'LMM Model':<15} | {'Spread'}")
        print("-" * 75)
        for k in ["Val", "Delta", "Gamma", "Vega", "HR"]:
            print(f"{k:<12} | {hw[k]:>13.4f} | {lmm[k]:>13.4f} | {hw[k] - lmm[k]:>10.4f}")
        print("-" * 75)

    print(f"\nTotal Execution Time: {time.time() - start_all:.2f} seconds")
