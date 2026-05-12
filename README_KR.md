# GPU 가속 기반 멀티 모델 Callable Swap 통합 평가 및 최적화 엔진
> **NVIDIA CUDA 기반 고성능 몬테카를로 시뮬레이션 및 이중 켈리브레이션 시스템**

> **Language**: [English](./README.md) | [한국어]

- 한국(KRW)과 미국(USD) 시장의 **Bermudan Callable Swap** 가치를 GPGPU를 활용하여 고속으로 계산하는 것을 목적으로 함
- GARCH 분석 및 Batch Calibration을 결합하여, Hull-White 모델과 LMM 모델의 Greeks 및 Hedge ratio를 계산함

---

## 프로젝트 구조 및 파일별 상세 역할
총 4개의 레이어로 구성되어 있으며, 데이터 수집, 최적화, 가격 계산 엔진, 실행파일로 구성되어 있음


| 레이어 | 파일명 | 상세 역할                                                                                                               |
| :--- | :--- |:--------------------------------------------------------------------------------------------------------------------|
| **Data** | `Datahandler.py` | **데이터 수집 및 처리**: ECOS(한국은행) 및 yfinance API를 이용하여 당일 1년, 5년 10년물의 10일분 데이터 수집. 가장 최근 스냅샷과 과거 데이터를 dictionary 형태로 반환. |
| | `Volatility.py` | **통계 분석 엔진**: Datahandler에서 수집된 데이터를 기반으로 GARCH(1,1) 및 EWMA fitting 수행. 향후 최적화를 위한 초기값 생성.                          |
| **Optimization** | `HW_cal.py` / `LMM_cal.py` | **이중 켈리브레이터**: 주어진 Swaption 가격을 기준으로 Gauss-Newton 최적화를 수행. 기본 행사가는 3.5%.                                            |
| **Engine** | `Model_selection.py` | **수익률 곡선 구축**: QuantLib을 이용하여 yield curve를 부트스트래핑함.                                                                 |
| | `HW_GPU.py` / `LMM_GPU.py` | **병렬 시뮬레이터**: 주어진 파라미터를 이용하여 이자율 경로 생성. Hull-White 및 LMM를 CUDA를 이용하여 계산.                                            |
| | `LSM_pricer.py` / `LSM_pricer_LMM.py` | **조기행사 결정 엔진**: GPU 내에서 Longstaff-Schwartz(LSM) 수행.                                                                 |
| **Controller** | `main.py` | **실행 파일**                                                                                                    |

`Datahandler.py` → `Volatility.py` → `Model_selection.py` → `HW_cal.py / LMM_cal.py` → `HW_GPU.py / LMM_GPU.py` → `LSM_pricer.py / LSM_pricer_LMM.py`

---
## 데이터 분석 결과 (Final Risk Metrics)
- 입력 데이터는 2026년 05월 11일 기준 ECOS와 yfinance의 1년, 5년 10년 국채의 10일분 데이터
- 버뮤단 옵션은 행사가 3.5%에 목표가 1.25% 로 설정

### [KOREA REPORT (KRW)]

| 지표 | Hull-White (HW) | LMM |
| :--- | :---: | :---: |
| **Val (평가가치)** | 3.2974 | 0.8003 |
| **Delta (1bp)** | 1.7722 | 2.4747 |
| **Gamma** | 0.0045 | -0.0405 |
| **Vega** | 1.8004 | 0.1222 |
| **HR (헤지비율)** | 0.3938 | 0.5499 |

### [USA REPORT (USD)]

| 지표 | Hull-White (HW) | LMM |
| :--- | :---: | :---: |
| **Val (평가가치)** | 3.0152 | 1.1573 |
| **Delta (1bp)** | 1.7343 | 2.4618 |
| **Gamma** | 0.0085 | -0.1497 |
| **Vega** | 1.8022 | 0.1180 |
| **HR (헤지비율)** | 0.3854 | 0.5471 |

### 주요 리서치 인사이트
1.  **모델 리스크(Model Risk)의 실증**: 한국 시장 기준, HW(3.2974)와 LMM(0.8003) 간 거대한 가치 스프레드가 관찰되었습니다. 이는 단일 인자 모델이 금리 곡선의 구조적 뒤틀림(Twist)을 과대평가함을 시사하며, 다요인 모델인 LMM을 통한 보정이 필수적임을 입증합니다.
2.  **네거티브 컨벡시티(Negative Convexity) 포착**: LMM 모델 결과에서 **음수 감마(KR: -0.0405, US: -0.1497)**가 뚜렷하게 관측되었습니다. 이는 금리 상승 시 조기행사 확률 증가로 가격 상승폭이 제한되는 Callable 상품 특유의 리스크를 GPU 시뮬레이션으로 완벽히 재현한 결과입니다.
3.  **Hedge Ratio 최적화**: 0.39 ~ 0.55 수준의 합리적인 헤지 비율이 산출되었습니다. 이는 옵션 1계약 방어를 위해 원금의 약 40~55% 수준의 일반 스왑 물량이 필요함을 의미하는 실무적 지표입니다.

---

## 성능 및 기술적 최적화 (HPC)

### [실행 성능 리포트]
*   **시뮬레이션 규모**: 시나리오당 100,000개 경로 생성.
*   **연산 범위**: 2개 국가 시장(KRW, USD)에 대해 2개 모델(HW, LMM)의 총 8가지 시나리오 동시 분석.
*   **전체 실행 시간**: **61.95초**
*   **포함 내역**: 실시간 데이터 수집 + 수익률 곡선 구축 + GPU 시뮬레이션 + LSM 후진 귀납법 + Greeks 산출.
*   **실시간 대응력**: 데이터 수집 이후, 단일 모델에 대한 가치 평가 및 Greeks 산출은 1초 미만의 속도로 완결됩니다.

### [하드웨어 사양]
*   **CPU**: AMD Ryzen 9 5950X (16코어, 3.4 GHz)
*   **GPU**: NVIDIA GeForce RTX 3070 (8GB VRAM, Ampere 아키텍처)
*   **OS**: Windows 10 / CUDA 13.x 기반

### [핵심 최적화 기술]
*   **Zero-Copy VRAM**: 경로 데이터 생성부터 회귀 분석까지 데이터를 CPU로 복사하지 않고 GPU 메모리 내에서 처리하여 PCIe 병목을 제거했습니다.
*   **Batch Jacobian 연산**: 최적화 시 수치 미분을 위한 여러 시나리오를 단 한 번의 커널 호출로 병렬 처리하여 최적화 속도를 CPU 대비 10배 이상 단축했습니다.

---

## 모델의 한계점 및 향후 과제 (Limitation)
*   **Volatility Surface 미반영**: 현재는 테너별 변동성까지만 최적화합니다. Smile/Skew 현상을 반영하기 위해 SABR 모델 등으로의 확장이 필요합니다.
*   **상관계수 고정**: LMM 내 테너 간 상관관계를 결정론적으로 가정합니다. 시장 급변기에 대비한 상관관계 Calibration 로직이 추가되어야 합니다.
*   **Single-Curve Framework**: 현대 금융 표준인 Multi-Curve (OIS-Libor Basis) 부트스트래핑은 향후 고도화 과제입니다.
