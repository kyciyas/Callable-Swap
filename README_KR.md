# GPU 가속 기반 멀티 모델 Callable Swap 통합 평가 및 최적화 엔진
> **NVIDIA CUDA 기반 고성능 몬테카를로 시뮬레이션 및 이중 켈리브레이션 시스템**

> **Language**: [English](./README.md) | [한국어]

- 한국(KRW)과 미국(USD) 시장의 **Bermudan Callable Swap** 가치를 GPGPU를 활용하여 고속으로 계산하는 것을 목적으로 함
- GARCH 분석 및 Batch Calibration을 결합하여, Hull-White 모델과 LMM 모델의 Greeks 및 Hedge ratio를 계산함
- Longstaff-Schwartz (LSM)을 이용하여 가격을 결정하기 위한 변동성 값을 최적화
---

## 프로젝트 구조 및 파일별 상세 역할
총 4개의 레이어로 구성되어 있으며, 데이터 수집, 최적화, 가격 계산 엔진, 실행파일로 구성되어 있음


| 레이어 | 파일명 | 상세 역할                                                                                                                                                                                                                |
| :--- | :--- |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Data** | `Datahandler.py` | **데이터 수집 및 처리**: ECOS(한국은행) 및 yfinance API를 이용하여 당일 1년, 5년 10년물의 10일분 데이터 수집. 가장 최근 스냅샷과 과거 데이터를 dictionary 형태로 반환.                                                                                                  |
| | `Volatility.py` | **통계 분석 엔진**: Datahandler에서 수집된 데이터를 기반으로 GARCH(1,1) 및 EWMA fitting 수행. 향후 최적화를 위한 초기값 생성.                                                                                                                           |
| **Optimization** | `HW_cal.py` / `LMM_cal.py` | **이중 켈리브레이터**: 주어진 Swaption 가격을 기준으로 Gauss-Newton 최적화를 수행.                                                                                                                                                           |
| **Engine** | `Model_selection.py` | **수익률 곡선 구축**: QuantLib을 이용하여 yield curve를 부트스트래핑함.                                                                                                                                                                  |
| | `HW_GPU.py` / `LMM_GPU.py` | **병렬 시뮬레이터**: 주어진 파라미터를 이용하여 이자율 경로 생성. Hull-White 및 LMM를 CUDA를 이용하여 계산. LMM은 Rebonato Parametrization ($\rho_{ij} = e^{-\beta \times \left\vert T_{i} - T_{j} \right\vert}$)을 이용하고 Cholesky decomposition 사용.       |
| | `LSM_pricer.py` / `LSM_pricer_LMM.py` | **조기행사 결정 엔진**: GPU 내에서 LSM 수행.                                                                                                                                                                                      |
| **Controller** | `main.py` | **실행 파일**                                                                                                                                                                                                            |

`Datahandler.py` → `Volatility.py` → `Model_selection.py` → `HW_cal.py / LMM_cal.py` → `HW_GPU.py / LMM_GPU.py` → `LSM_pricer.py / LSM_pricer_LMM.py`

---
## 데이터 분석 결과 (Final Risk Metrics)
- 입력 데이터는 2026년 05월 11일 기준 ECOS와 yfinance의 1년, 5년 10년 국채의 10일분 데이터
- 버뮤단 옵션은 행사가 3.5%에 목표가 1.25% 로 설정
- Rebonato Parametrization 의 beta = 1.5 사용

### [KOREA REPORT (KRW)]

| 지표 | Hull-White (HW) | LMM |
| :--- | :---: | :---: |
| **Val (평가가치)** | 3.3588 | 0.9210 |
| **Delta (1bp)** | 1.8299 | 2.6241 |
| **Gamma** | 0.0339 | -0.6263 |
| **Vega** | 1.7867 | 0.1118 |
| **HR (헤지비율)** | 0.4067 | 0.5831 |

### [USA REPORT (USD)]

| 지표 | Hull-White (HW) | LMM |
| :--- | :---: | :---: |
| **Val (평가가치)** | 3.0183 | 1.1385 |
| **Delta (1bp)** | 1.7174 | 2.5285 |
| **Gamma** | -0.0142 | -0.2952 |
| **Vega** | 1.8013 | 0.1155 |
| **HR (헤지비율)** | 0.3816 | 0.5619 |


### 주요 리서치 인사이트
- **Val (평가가치)**: 모델에서 계산한 공정가치로 실제 행사가가 3.5%, 목표가가 1.25%인 Callable swap의 가치
#### HW와 LMM의 비교
1. **Val**: 목표가인 1.25%를 기준으로 한국과 미국 시장 모두 HW는 고평가를, LMM은 저평가를 하고 있음.
2. **Delta**: 한국과 미국 시장 모두 LMM이 HW보다 높은 값을 보이고 있으며 1bp 변화에 따른 파생상품 가격 변화는 LMM이 높다는 것이 확인됨.
3. **Gamma**: HW 모델은 한국 시장과 미국 시장에서 0.0339, -0.0142, LMM은 -0.6263, -0.2952를 기록 하였으며 조기 행사 가능성 으로 인한
negative convexity 를 확인 하였음
4. **Vega**: HW 모델은 한국과 미국시장 모두 1.1을 넘는 수치를 기록하였고, LMM은 0.11 수준으로 낮게 계산되었으며 이는 LMM 모델이 테너별
변동성을 반영하여 단순한 변동성에 영향을 적게 받음을 확인 할 수 있음
5. **Hedge Ratio**: 한국과 미국 시장 모두 LMM이 HW보다 높은 HR를 보이고 있으며, 비록 HW 모델이 낮은 HR에도 불구하고 고평가된 가치로 인하여
향후 큰 금리 변동시 큰 오차가 발생할 가능성이 내재되어 있음

#### 한국과 미국 시장의 비교
1. HW와 LMM의 가격 차이는 한국과 미국이 각각 2.4378, 1.8798로 미국 시장의 변화량이 더 적음을 볼 수 있으며, 

---

## 성능 및 기술적 최적화 (HPC)

### [실행 성능 리포트]
*   **시뮬레이션 규모**: 시나리오당 100,000개 경로 생성.
*   **연산 범위**: 2개 국가 시장(KRW, USD)에 대해 2개 모델(HW, LMM)의 총 4가지 시나리오 동시 분석.
*   **전체 실행 시간**: 평균 **40초**
*   **포함 내역**: 실시간 데이터 수집 (ECOS 및 yfinance) + 수익률 곡선 구축 + 파라미터 최적화 + LSM + Greeks 산출.

### [하드웨어 사양]
*   **CPU**: AMD Ryzen 9 5950X (16코어, 3.4 GHz)
*   **GPU**: NVIDIA GeForce RTX 3070 (8GB VRAM, Ampere 아키텍처)
*   **OS**: Windows 10 / CUDA 13.x 기반

---

## 모델의 한계점 및 향후 과제 (Limitation)
*   **Volatility Surface 미반영**: 행사가 (3.5%) 및 목표가 (1.25%)를 직접 입력해야 하는 구조로 시장의 상품 특성을 자동으로 업데이트 할 필요성 존재. 
*   **상관계수 모델의 임의적 선택**: LMM 내 테너 간 상관관계를 단순한 모형인 Rebonato Parametrization을 이용하며 beta 값에 대한 최적화 과정이 없음.
*   **Single-Curve Framework**: Multi-Curve (OIS-Libor Basis) 부트스트래핑 적용이 필요함.
