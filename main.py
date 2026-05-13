import Datahandler
import Model_selection
import HW_GPU
import LMM_GPU
import LSM_pricer
import LSM_pricer_LMM
import Volatility
import HW_cal
import LMM_cal
from matplotlib import pyplot as plt

from datetime import datetime
import cupy as cp
import numpy as np
import pandas as pd
import QuantLib as ql
import time
import gc
import os

class Callableswap:
    def __init__(self, country_code = 'KR'):
        self.__gpu_init()

        ##### input parameters #####
        self.country_code = country_code
        self.ecos_api = 'sample'
        self.market_rate = 0.0125
        self.strike_rate = 0.035
        self.years = 5.0
        self.steps = 500
        self.dt = 0.25
        self.n_paths = 100_000
        self.load_parameters_from_csv()
        self.ois = 0.035
        self.ois_list_hw = []
        self.ois_list_lmm = []

        ##### random seed fix #####
        np.random.seed(42)
        cp.random.seed(42)

        ##### gather rates data #####
        self.rates = 0.0
        self.rates_list = {}
        self.rate_data_handler = Datahandler.Datahandler(country=self.country_code)
        self.gather_rate_data()

        ##### load OIS curve #####
        self.ois_data_handler = Datahandler.OisDataHandler(tag=self.country_code, rates_dict=self.rates_list)
        self.gather_OIS_data()
        self.make_ois_curve()

        ##### vol calculation engine setup #####
        self.vol_engine = Volatility.VolatilityEngine(data=self.rates_list['10Y'])
        self.hw_period, self.lmm_period = self.get_exercise_steps()

    def load_parameters_from_csv(self, file_path="input.csv"):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[ERROR] File not found: '{file_path}'")

        with open(file_path, 'r', encoding='utf-8-sig') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if len(lines) < 3:
            raise ValueError(
                f"[ERROR] Insufficient data rows in '{file_path}'. (At least 3 lines including header required)")

        if self.country_code == "KR":
            target_line = lines[1]
        elif self.country_code == "US":
            target_line = lines[2]
        else:
            raise ValueError(f"[ERROR] Unsupported country code: '{self.country_code}' (Only 'KR' or 'US' is allowed)")

        tokens = [t.strip() for t in target_line.split(',') if t.strip()]

        if len(tokens) < 8:
            raise ValueError(f"[ERROR] Insufficient parameters in the row for '{self.country_code}'. (At least 8 parameters required)")

        self.country_code = tokens[0].split()[-1].strip()
        self.ecos_api     = str(tokens[1]).strip()
        self.market_rate  = float(tokens[2])
        self.strike_rate  = float(tokens[3])
        self.years        = float(tokens[4])
        self.steps        = int(tokens[5])
        self.dt           = float(tokens[6])
        self.n_paths      = int(tokens[7])

        print("\n" + "="*50)
        print(f" [LOADED PARAMETERS VIA FIXED ROW - {self.country_code} MARKET]")
        print("="*50)
        print(f" - Country Code : {self.country_code}")
        print(f" - ECOS API Key : {self.ecos_api}")
        print(f" - Market Rate  : {self.market_rate:.4f} ({self.market_rate * 100:.2f}%)")
        print(f" - Strike Rate  : {self.strike_rate:.4f} ({self.strike_rate * 100:.2f}%)")
        print(f" - Swap Years   : {self.years} Years")
        print(f" - Model Steps  : {self.steps} Steps")
        print(f" - LMM Delta T  : {self.dt}")
        print(f" - Simu Paths   : {self.n_paths:,} Paths")
        print("="*50 + "\n")

    def set_HW_periods(self, array):
        self.hw_period = array

    def set_LMM_periods(self, array):
        self.lmm_period = array

    def __gpu_init(self):
        try:
            device = cp.cuda.Device(0)
            device.synchronize()

            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()

            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            print(f"[INFO] GPU Memory Cleared: {mempool.used_bytes() / 1024 ** 2:.2f} MB used")

        except Exception as e:
            print(f"[ERROR] Failed to clear GPU memory: {e}")

    def gather_rate_data(self):
        print(f"\n[INFO] {self.country_code} market engine is running...")
        self.rates, self.rates_list = self.rate_data_handler.fetch_from_ecos(api_key=self.ecos_api) if self.country_code == "KR" else self.rate_data_handler.fetch_from_yfinance()

    def gather_OIS_data(self):
        if self.country_code == 'KR':
            self.ois = self.ois_data_handler.fetch_live_kofr(api_key=self.ecos_api)
        elif self.country_code == 'US':
            self.ois = self.ois_data_handler.fetch_live_sofr()
        eval_date = ql.Date(datetime.today().day, datetime.today().month, datetime.today().year)
        self.ql_data = self.ois_data_handler.build_ois_curve(evaluation_date=eval_date, rate = self.ois)

    def make_ois_curve(self):
        vectorized_discount = np.vectorize(self.ql_data.discount)

        hw_time_array = np.linspace(0, self.years, self.steps + 1)
        self.ois_list_hw = vectorized_discount(hw_time_array).astype(np.float32)

        n_lmm_grids = int(self.years / self.dt)
        lmm_time_array = np.arange(0, n_lmm_grids + 1) * self.dt

        self.ois_list_lmm = vectorized_discount(lmm_time_array).astype(np.float32)

    def calculate_vol(self):
        rel_sigma_garch = self.vol_engine.get_comparison_report()['garch']

        return rel_sigma_garch

    def model_init(self):
        yield_curve = Model_selection.InterestRateDataEngine(self.rates, self.rate_data_handler.calendar, self.rate_data_handler.settlement_days)
        hw_curve = yield_curve.get_hull_white_input(years=self.years, n_steps=self.steps)
        lmm_curve = yield_curve.get_lmm_input(years=self.years, dt=self.dt)

        return hw_curve, lmm_curve

    def vol_calibration(self, hw_curve, lmm_curve, rel_sigma_garch):
        hw_calib = HW_cal.GPUBatchCalibrator(HW_GPU.GPUOptHWPricer(n_paths=self.n_paths), [self.market_rate], hw_curve, self.ois_list_hw, strike=self.strike_rate)
        opt_hw = hw_calib.run_optimization(init_s=float(self.rates['10Y']) * rel_sigma_garch)
        opt_a, opt_sig_hw = float(opt_hw[0]), float(opt_hw[1])

        lmm_calib = LMM_cal.GPULMMCalibrator(LMM_GPU.GPULMMBatchPricer(n_paths=self.n_paths, n_rates=len(lmm_curve)), [self.market_rate] * len(lmm_curve), lmm_curve, strike=self.strike_rate)
        opt_sig_lmm = lmm_calib.run_lmm_optimization(init_sigmas=[rel_sigma_garch] * len(lmm_curve))

        return opt_a, opt_sig_hw, opt_sig_lmm

    def get_exercise_steps(self):
        lock_out = 1.0
        frequency = 0.25

        exercise_years = []
        current_y = lock_out
        while current_y < self.years:
            exercise_years.append(current_y)
            current_y += frequency

        hw_steps = [int((y / self.years) * self.steps) for y in exercise_years]
        lmm_steps = [int(y / self.dt) for y in exercise_years]

        return hw_steps, lmm_steps

    def calculate_metrics(self, model_type, init_rates, sigma_val, a_val=0.1):
        results = {}
        bump = 0.0001 # 1bp

        for name, shift in {'Base': 0, 'Up': bump, 'Dn': -bump, 'Vega': 0}.items():
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()

            if model_type == "HW":
                s = (sigma_val + 0.01) if name == 'Vega' else sigma_val
                pricer = HW_GPU.GPUHullWhitePricer(n_paths=self.n_paths, sigma=float(s), a=a_val)
                raw_paths = pricer.generate_paths(np.array(init_rates, dtype=np.float32).flatten() + (shift if name != 'Vega' else 0), self.ois_list_hw)
                lsm = LSM_pricer.GPULSMPricer(cp.asarray(raw_paths), self.ois_list_hw, self.years/self.steps, strike=self.strike_rate)
                val = lsm.run_lsm_gpu(exercise_steps=self.hw_period)
            else:
                if isinstance(sigma_val, (np.ndarray, cp.ndarray)):
                    s_scalar = float(cp.mean(cp.asarray(sigma_val)))
                else:
                    s_scalar = float(sigma_val)

                s_input = (s_scalar + 0.05) if name == 'Vega' else s_scalar
                curr_rates = np.array(init_rates, dtype=np.float32) + (shift if name != 'Vega' else 0)

                pricer = LMM_GPU.GPULMMPricer(curr_rates, n_paths=self.n_paths, sigma=s_input)
                val = LSM_pricer_LMM.GPULMM_LSMPricer(pricer.generate_lmm_paths(), self.dt , strike=self.strike_rate).run_lsm(
                    exercise_steps=self.lmm_period)

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

    def run(self):
        print(f"\n[EXECUTE] {self.country_code} 파이프라인 연산을 시작합니다.")

        rel_sigma_garch = self.calculate_vol()
        hw_curve, lmm_curve = self.model_init()
        opt_a, opt_sig_hw, opt_sig_lmm = self.vol_calibration(hw_curve=hw_curve, lmm_curve=lmm_curve, rel_sigma_garch=rel_sigma_garch)

        metrics_report = {
            "HW": self.calculate_metrics("HW", hw_curve, opt_sig_hw, opt_a),
            "LMM": self.calculate_metrics("LMM", lmm_curve, opt_sig_lmm)
        }

        return metrics_report

if __name__ == "__main__":
    start = time.time()

    kr_engine = Callableswap('KR')
    res_kr = kr_engine.run()

    us_engine = Callableswap('US')
    res_us = us_engine.run()

    for c, r in [("KOREA", res_kr), ("USA", res_us)]:
        print(f"\n{c} REPORT\n{pd.DataFrame(r).round(4)}")

    print(f"\nTotal Time: {time.time() - start:.2f}s")
