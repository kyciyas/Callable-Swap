import Datahandler
import Model_selection
import HW_GPU
import LMM_GPU
import LSM_pricer
import LSM_pricer_LMM
import Volatility
import HW_cal
import LMM_cal

import cupy as cp
import numpy as np
import pandas as pd
import time
import gc

np.random.seed(42)
cp.random.seed(42)

def run_valuation_pipeline(country_code, api_key=None, target_price=0.0125, strike_price=0.035):
    print(f"\n[INFO] {country_code} market engine is running...")
    handler = Datahandler.Datahandler(country=country_code)
    rates, rates_list = handler.fetch_from_ecos(
        api_key=api_key) if country_code == "KR" else handler.fetch_from_yfinance()

    vol_engine = Volatility.VolatilityEngine(data=rates_list['10Y'])
    rel_sigma_garch = vol_engine.get_comparison_report()['garch']

    engine_temp = Model_selection.InterestRateDataEngine(rates, handler.calendar, handler.settlement_days)
    hw_init = engine_temp.get_hull_white_input(T=5.0, n_steps=500)
    lmm_init = engine_temp.get_lmm_input(horizon=5.0, dt=0.25)

    # HW Calibration
    hw_calib = HW_cal.GPUBatchCalibrator(HW_GPU.GPUOptHWPricer(n_paths=100_000), [target_price], hw_init, strike=strike_price)
    opt_hw = hw_calib.run_optimization(init_s=float(rates['10Y']) * rel_sigma_garch)
    opt_a, opt_sig_hw = float(opt_hw[0]), float(opt_hw[1])

    # LMM Calibration
    lmm_calib = LMM_cal.GPULMMCalibrator(LMM_GPU.GPULMMBatchPricer(n_paths=100_000, n_rates=len(lmm_init)), [target_price] * len(lmm_init), lmm_init,strike=strike_price)
    opt_sig_lmm = lmm_calib.run_lmm_optimization(init_sigmas=[rel_sigma_garch] * len(lmm_init))

    def calculate_metrics(model_type, init_rates, sigma_val, a_val=0.1):
        results, bump = {}, 0.0001
        for name, shift in {'Base': 0, 'Up': bump, 'Dn': -bump, 'Vega': 0}.items():
            gc.collect();
            cp.get_default_memory_pool().free_all_blocks()
            if model_type == "HW":
                s = (sigma_val + 0.01) if name == 'Vega' else sigma_val
                pricer = HW_GPU.GPUHullWhitePricer(n_paths=100000, sigma=float(s), a=a_val)
                raw_paths = pricer.generate_paths(np.array(init_rates, dtype=np.float32).flatten() + (shift if name != 'Vega' else 0))
                lsm = LSM_pricer.GPULSMPricer(cp.asarray(raw_paths), 5.0 / 500, strike=strike_price)
                val = lsm.run_lsm_gpu(exercise_steps=[int((y / 5.0) * 500) for y in [1, 2, 3, 4]])
            else:
                if isinstance(sigma_val, (np.ndarray, cp.ndarray)):
                    s_scalar = float(cp.mean(cp.asarray(sigma_val)))
                else:
                    s_scalar = float(sigma_val)

                s_input = (s_scalar + 0.05) if name == 'Vega' else s_scalar
                curr_rates = np.array(init_rates, dtype=np.float32) + (shift if name != 'Vega' else 0)

                pricer = LMM_GPU.GPULMMPricer(curr_rates, n_paths=100000, sigma=s_input)
                val = LSM_pricer_LMM.GPULMM_LSMPricer(pricer.generate_lmm_paths(), 0.25, strike=strike_price).run_lsm(exercise_steps=[4, 8, 12, 16])

            results[name] = float(val.get())
        b, u, d, v = results['Base'], results['Up'], results['Dn'], results['Vega']

        delta_1bp = (u - d) / 2


        gamma = (u - 2 * b + d) / (bump ** 2) * 0.0001

        vega = (v - b) * 100

        vanilla_dv01 = 0.00045
        hr = delta_1bp / vanilla_dv01

        return {
            "Val": b * 100,
            "Delta": delta_1bp * 10000,
            "Gamma": gamma,
            "Vega": vega,
            "HR": hr
        }

    return {"HW": calculate_metrics("HW", hw_init, opt_sig_hw, opt_a),
            "LMM": calculate_metrics("LMM", lmm_init, opt_sig_lmm)}


if __name__ == "__main__":
    start = time.time()
    ecos_key = ""
    res_kr = run_valuation_pipeline("KR", ecos_key)
    res_us = run_valuation_pipeline("US")
    for c, r in [("KOREA", res_kr), ("USA", res_us)]:
        print(f"\n{c} REPORT\n{pd.DataFrame(r).round(4)}")
    print(f"\nTotal Time: {time.time() - start:.2f}s")