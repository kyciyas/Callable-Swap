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

        self.ois = 0.035
        self.hw_curve = []
        self.lmm_curve = []
        self.ois_list_hw = []
        self.ois_list_lmm = []
        self.hw_period = []
        self.lmm_period = []
        self.rates = 0.0
        self.rates_list = {}
        self.hull_white_a = 0.03

        self.load_parameters_from_csv()

        ##### random seed fix #####
        np.random.seed(42)
        cp.random.seed(42)

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

    def model_init(self):
        yield_curve = Model_selection.InterestRateDataEngine(self.rates, self.rate_data_handler.calendar, self.rate_data_handler.settlement_days, years=self.years, steps=self.steps, dt=self.dt)
        self.hw_curve = yield_curve.get_hull_white_input()
        self.lmm_curve = yield_curve.get_lmm_input()

    def vol_calibration(self):
        price_engine = HW_GPU.GPUOptHWPricer(hw_input= self.hw_curve, hw_ois=self.ois_list_hw, n_paths=self.n_paths)
        hw_calib = HW_cal.GPUBatchCalibrator(price_engine, market_rate = [self.market_rate], hw_input = self.hw_curve, hw_ois = self.ois_list_hw, strike=self.strike_rate, init_s=float(self.rates['10Y']) * self.rel_sigma_garch)
        opt_hw = hw_calib.run_optimization()
        opt_a, opt_sig_hw = float(opt_hw[0]), float(opt_hw[1])
        lmm_pricer_engine = LMM_GPU.GPULMMBatchPricer(self.ois_list_lmm, n_paths=self.n_paths, n_rates=len(self.lmm_curve))
        lmm_calib = LMM_cal.GPULMMCalibrator(lmm_pricer_engine,f0=self.lmm_curve, market_rate=[self.market_rate] * len(self.lmm_curve), bwd_gpu=self.ois_list_lmm, strike=self.strike_rate, dt=self.dt_vector)
        opt_sig_lmm = lmm_calib.run_lmm_optimization(init_sigmas=[self.rel_sigma_garch] * len(self.lmm_curve))
        return opt_a, opt_sig_hw, opt_sig_lmm

    def get_exercise_steps(self):
        lock_out = 1.0
        frequency = 0.25

        exercise_years = []
        current_y = lock_out
        while current_y < self.years:
            exercise_years.append(current_y)
            current_y += frequency
        self.hw_period = [int((y / self.years) * self.steps) for y in exercise_years]
        self.lmm_period = [int(y / self.dt) for y in exercise_years]

    def calculate_metrics(self, model_type, init_rates, sigma_val, a_val=0.1, beta = 1.0):
        results = {}
        bump = 0.0001 # 1bp

        for name, shift in {'Base': 0, 'Up': bump, 'Dn': -bump, 'Vega': 0}.items():
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()

            if model_type == "HW":
                s = (sigma_val + 0.01) if name == 'Vega' else sigma_val
                pricer = HW_GPU.GPUHullWhitePricer(n_paths=self.n_paths, sigma=float(s), a=a_val)
                raw_paths = pricer.generate_paths(np.array(init_rates, dtype=np.float32).flatten() + (shift if name != 'Vega' else 0), self.ois_list_hw)
                lsm = LSM_pricer.GPULSMPricer(cp.asarray(raw_paths), self.ois_list_hw, self.years/self.steps, strike=self.strike_rate, exercise_steps=self.hw_period)
                val = lsm.run_lsm_gpu()
            else:
                s_input = (sigma_val + 0.05) if name == 'Vega' else sigma_val
                curr_rates = np.array(init_rates, dtype=np.float32) + (shift if name != 'Vega' else 0)

                pricer = LMM_GPU.GPULMMPricer(curr_rates, self.ois_list_lmm, beta=beta, n_paths=self.n_paths, sigma=s_input)
                val = LSM_pricer_LMM.GPULMM_LSMPricer(pricer.generate_lmm_paths(), strike=self.strike_rate, bwd_gpu=self.ois_list_lmm, dt=self.dt_vector).run_lsm(
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

    def generate_actual_dt_vector(self):
        # 1. 평가일 및 만기일 설정
        today = datetime.today()
        start_date = ql.Date(today.day, today.month, today.year)
        months_per_step = 3
        dynamic_tenor = ql.Period(months_per_step, ql.Months)
        end_date = start_date + ql.Period(int(self.years), ql.Years)


        # 2. QuantLib 정식 스케줄러 가동 (Holiday, Business Convention 완벽 반영)
        schedule = ql.Schedule(
            start_date,
            end_date,
            dynamic_tenor,
            self.rate_data_handler.calendar,
            self.rate_data_handler.business_convention,
            self.rate_data_handler.business_convention,
            ql.DateGeneration.Forward,
            True  # <- EndOfMonth = True 반영 완료
        )

        # 3. 각 구간별 실제 일수 계산 비율(Day Count Fraction) 추출
        dt_list = []
        for i in range(len(schedule) - 1):
            dt_frac = self.rate_data_handler.day_count.yearFraction(schedule[i], schedule[i + 1])
            dt_list.append(dt_frac)

        # 프라이서와 최적화 엔진이 참조할 실제 스케줄 배열 고정 변수화
        self.dt_vector = np.array(dt_list, dtype=np.float32)
        # self.steps = len(self.dt_vector)  # 실제 영업일 밀림으로 조정된 최종 스텝 수 동기화

        print(f"[스케줄러] 영업일 조율 완료. 총 스텝 수: {self.steps}")
        print(f"[스케줄러] 실제 dt 구간 구조: {self.dt_vector[:4]} ... {self.dt_vector[-1]}")

    def run(self):
        print(f"\n[EXECUTE] {self.country_code} 파이프라인 연산을 시작합니다.")
        ##### gather rates data #####
        self.rate_data_handler = Datahandler.Datahandler(country=self.country_code)
        self.gather_rate_data()
        self.generate_actual_dt_vector()

        ##### load OIS curve #####
        self.ois_data_handler = Datahandler.OisDataHandler(tag=self.country_code, rates_dict=self.rates_list)
        self.gather_OIS_data()
        self.make_ois_curve()

        ##### vol calculation engine setup #####
        self.vol_engine = Volatility.VolatilityEngine(data=self.rates_list['10Y'])
        self.get_exercise_steps()
        self.rel_sigma_garch = self.vol_engine.get_comparison_report()['garch']
        self.model_init()

        # opt_a, opt_sig_hw, opt_sig_lmm = self.vol_calibration()

        metrics_report = {
            "HW": self.calculate_metrics("HW", self.hw_curve, self.rel_sigma_garch * self.strike_rate, self.hull_white_a),
            "LMM": self.calculate_metrics("LMM", self.lmm_curve, self.rel_sigma_garch, beta = 1.0)
        }
        lmm_dv01_report = self.calculate_key_rate_dv01(optimized_sigma=self.rel_sigma_garch, model_type="LMM", beta=1.0)
        hw_dv01_report = self.calculate_key_rate_dv01(optimized_sigma=self.rel_sigma_garch, model_type="HW", a_val=self.hull_white_a)

        return metrics_report, lmm_dv01_report, hw_dv01_report

    def calculate_key_rate_dv01(self, optimized_sigma, model_type="LMM", a_val=0.1, beta=1.0):
        print(f"\n" + "#" * 60)
        print(f" [MIDDLE-OFFICE RISK DISPATCH: KEY RATE DELTA & DV01 BUCKETS]")
        print("#" * 60)
        print(f" - Target Model Engine : {model_type}")
        print(f" - Bumping Magnitude  : +1bp (0.0001)")
        print("#" * 60)

        bump = 0.0001  # 1bp
        dv01_buckets = {}

        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

        # 1. 베이스 라인 가격 산출
        if model_type == "HW":
            base_pricer = HW_GPU.GPUHullWhitePricer(n_paths=self.n_paths, sigma=float(optimized_sigma), a=a_val)
            base_raw_paths = base_pricer.generate_paths(np.array(self.hw_curve, dtype=np.float32).flatten(),
                                                        self.ois_list_hw)
            base_lsm = LSM_pricer.GPULSMPricer(cp.asarray(base_raw_paths), self.ois_list_hw, self.years / self.steps,
                                               strike=self.strike_rate, exercise_steps=self.hw_period)
            base_price = float(base_lsm.run_lsm_gpu())
            target_curve = np.array(self.hw_curve, dtype=np.float32).flatten()
        else:
            base_pricer = LMM_GPU.GPULMMPricer(np.array(self.lmm_curve, dtype=np.float32), self.ois_list_lmm, beta=beta,
                                               n_paths=self.n_paths, sigma=optimized_sigma)
            base_lsm = LSM_pricer_LMM.GPULMM_LSMPricer(base_pricer.generate_lmm_paths(), self.strike_rate,
                                                       self.ois_list_lmm, dt=self.dt_vector)
            base_price = float(base_lsm.run_lsm(exercise_steps=self.lmm_period))
            target_curve = np.array(self.lmm_curve, dtype=np.float32).flatten()

        print(f" -> Base Fair Value (NPV) = {base_price:.6f}")
        print(f" -> Key Rate Bumping Sequence Initiated...")


        n_risk_nodes = len(self.dt_vector)

        # 500스텝 구조인 HW 모델과 20스텝 구조인 LMM 모델의 펌핑 루프를 일원화합니다.
        for node_idx in range(n_risk_nodes):
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()

            bumped_curve = target_curve.copy()

            if model_type == "HW":
                target_t = (node_idx + 1) * self.dt  # 예: 0.25Y, 0.50Y, 0.75Y...
                hw_timeline = np.linspace(0, self.years, self.steps + 1)

                # 500개 각 칸마다 해당 거점 테너와의 인접도 가중치 계산 (삼각형 마커 범핑)
                for s_idx in range(len(bumped_curve)):
                    t_current = hw_timeline[s_idx]
                    # 거점 테너 주변 0.25Y 반경 이내의 스텝들에 가중치 분산 배정
                    weight = max(0.0, 1.0 - abs(t_current - target_t) / 0.25)
                    bumped_curve[s_idx] += bump * weight

                bumped_pricer = HW_GPU.GPUHullWhitePricer(n_paths=self.n_paths, sigma=float(optimized_sigma), a=a_val)
                bumped_raw_paths = bumped_pricer.generate_paths(bumped_curve, self.ois_list_hw)
                bumped_lsm = LSM_pricer.GPULSMPricer(cp.asarray(bumped_raw_paths), self.ois_list_hw,
                                                     self.years / self.steps, strike=self.strike_rate,
                                                     exercise_steps=self.hw_period)
                bumped_price = float(bumped_lsm.run_lsm_gpu())

            else:
                bumped_curve[node_idx] += bump
                bumped_pricer = LMM_GPU.GPULMMPricer(bumped_curve, self.ois_list_lmm, beta=beta, n_paths=self.n_paths,
                                                     sigma=optimized_sigma)
                bumped_lsm = LSM_pricer_LMM.GPULMM_LSMPricer(bumped_pricer.generate_lmm_paths(), self.strike_rate,
                                                             self.ois_list_lmm, dt=self.dt_vector)
                bumped_price = float(bumped_lsm.run_lsm(exercise_steps=self.lmm_period))

            node_tenor_name = f"Tenor_{(node_idx + 1) * self.dt:.2f}Y"
            dv01_buckets[node_tenor_name] = bumped_price - base_price

            print(
                f"   [버킷] {node_tenor_name} Node 1bp Bumped NPV: {bumped_price:.6f} | DV01: {dv01_buckets[node_tenor_name]:+.6f}")

        print("#" * 60)
        print(f" [KEY RATE DELTA COMPLETE - DISPATCHED TO MIDDLE-OFFICE LEDGER]")
        print("#" * 60 + "\n")

        return dv01_buckets

if __name__ == "__main__":
    start = time.time()

    kr_engine = Callableswap('KR')

    res_kr, kr_lmm_dv01, kr_hw_dv01 = kr_engine.run()

    us_engine = Callableswap('US')
    res_us, us_lmm_dv01, us_hw_dv01 = us_engine.run()

    # 2. 기존 Greeks 지표 테이블 출력 (원본 무결성 유지)
    for c, r in [("KOREA", res_kr), ("USA", res_us)]:
        print(f"\n" + "=" * 50)
        print(f" [{c} BASE RISK METRICS REPORT]")
        print("=" * 50)
        print(pd.DataFrame(r).round(4))
        print("=" * 50)

    for c, lmm_d, hw_d in [("KOREA", kr_lmm_dv01, kr_hw_dv01), ("USA", us_lmm_dv01, us_hw_dv01)]:
        print(f"\n" + "#" * 50)
        print(f" [{c} MIDDLE-OFFICE KEY RATE DV01 DISPATCH]")
        print("#" * 50)
        dv01_df = pd.DataFrame({
            "HW_DV01": pd.Series(hw_d),
            "LMM_DV01": pd.Series(lmm_d)
        })
        print(dv01_df.round(6))  # DV01은 수치가 미세하므로 소수점 6자리까지 표현하는 것이 마켓 표준입니다.
        print("#" * 50)
        ########################################################################

    print(f"\nTotal Time: {time.time() - start:.2f}s")
