# GPU 가속 기반 멀티 모델 Callable Swap 통합 평가 및 최적화 엔진
> **Swaption의 가치를 HW 모델과 Libor Market Model(LMM) 기반으로 고속연산을 수행하는 자산 평가 시스템**

> **Language**: [English](./README.md) | [한국어]

- 한국(KRW)과 미국(USD) 시장의 **Callable Swap**의 가치를 **GPGPU (CUDA)**를 활용하여 고속으로 계산하는 것을 목적으로 함
- GARCH 분석 및 Batch Calibration을 결합하여, Hull-White 모델과 LMM의 파라미터 최적화를 수행
- Longstaff-Schwartz (LSM) 알고리즘을 이용하여 근사적으로 조기행사 여부를 반영하여 가격을 결정
- OpenAPI의 한계로 IRS(Forward) 및 OIS(Backward) rate를 구할 수 없기에 국고채 1y, 5y, 10y 정보를 프록시로 이용하여 multi curve 구현
---

## 프로젝트 구조 및 파일별 상세 역할
총 4개의 레이어로 구성되어 있으며, 데이터 수집, 최적화, 가격 계산 엔진, 실행파일로 구성되어 있음


| 레이어 | 파일명 | 상세 역할                                                                                                                                                                                   |
| :--- | :--- |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Data** | `Datahandler.py` | **데이터 수집 및 처리**: ECOS(한국은행) 및 yfinance API를 이용하여 당일 기준 1년, 5년 10년 만기 국고채 데이터와 전일 KOFR 및 SOFR 초단기 금리 수집. IRS 예측 관습과 무위험 OIS 할인 관습을 적용.                 |
| | `Volatility.py` | **통계 분석 엔진**: Datahandler에서 수집된 데이터를 기반으로 GARCH(1,1) 및 EWMA fitting 수행. 최적화를 위한 초기 변동성 값 추출.                                                                                            |
| **Optimization** | `HW_cal.py` / `LMM_cal.py` | **켈리브레이터**: 비선형 최적화 알고리즘(Levenberg-Marquardt 등)을 사용하여 역사적 변동성 구조에 캘리브레이션.                                                                                                                          |
| **Engine** | `Model_selection.py` | **수익률 곡선 구축**: QuantLib을 이용하여 yield curve를 부트스트래핑함.                                                                                                                                     |
| | `HW_GPU.py` / `LMM_GPU.py` | **병렬 시뮬레이터**: 주어진 파라미터를 기반으로 GPU를 이용하여 이자율 경로 생성. LMM은 Rebonato Parametrization ($\rho_{ij} = e^{-\beta \times \left\vert T_{i} - T_{j} \right\vert}$)을 이용하고 Cholesky decomposition 사용. |
| | `LSM_pricer.py` / `LSM_pricer_LMM.py` | **조기행사 결정 엔진**: CUDA를 이용하여 LSM 수행.                                                                                                                                                      |
| **Controller** | `main.py` | **실행 파일**                                                                                                                                                                               |

`Datahandler.py` → `Volatility.py` → `Model_selection.py` → `HW_cal.py / LMM_cal.py` → `HW_GPU.py / LMM_GPU.py` → `LSM_pricer.py / LSM_pricer_LMM.py`

---
## Multi-Curve Pricing 적용

### 이론적 배경
- Single curve는 미래 현금 흐름을 예측하는 Forward curve 와 이를 현재 가치로 계산하는 Discount curve를 같다고 간주함 (Libor가 risk free rate)
- 2008년 금융 위기 이후 대형 은행의 파산으로 인하여 은행간 신용 위험 (credit risk)와 자금의 유동성 위험 (liquidity risk) 발생
- 이로 인하여 tenor basis가 확대되었으며 이를 반영하기 위하여 multi curve 도입이 필요해짐

### Callable Swap에 multi-curve framework 적용이 필요한 이유
- **헷지 비용의 기준**: Callable Swap 매도 후 반대 포지션을 구성할 때, 헷지 포트폴리오의 조달 비용은 무위험 금리(OIS/KOFR/SOFR)를 따름
- **현금흐름의 기준**: 스왑 계약서상 변동금리(Floating Leg) 지급 조건은 신용/유동성 위험을 포함한 준거 금리의 Forward 커브를 따름
- **헷지 비용과 현금흐름 이자율의 불일치 해결**: 현금 흐름과 헷지 비용을 계산하기 위한 금리가 다르기에 이를 각기 따로 적용해야 함

### Hull-White 모델
#### Single curve
- **Model**: $dr_t = \left( \theta(t) - a r_t \right) dt + \sigma dW_t$
- **Discount factor**: $P(t,T) = A(t,T) \exp\left(-B(t,T)r_t\right)$
- **Forward rate**: $F(t; T_1, T_2) = \frac{1}{\tau} \left( \frac{P(t,T_1)}{P(t,T_2)} - 1 \right)$

| 파라미터 기호 | 영문 명칭                | 국문 명칭        | 주요 역할 및 성격                                                                                                                       |
| :---: |:---------------------|:-------------|:---------------------------------------------------------------------------------------------------------------------------------|
| $a$ | Mean Reversion Speed | 평균 회귀 속도     | 단기금리가 장기 평균 수준($\theta(t)/a$)으로 되돌아오는 속도를 결정하는 상수. 값이 클수록 금리가 평균으로 강하게 끌려가며 미래 금리의 변동 범위가 작아짐.   |
| $\sigma$ | Volatility           | 단기금리 변동성     | 단기금리에 가해지는 무작위 충격의 크기를 조절하는 상수.    |
| $\theta(t)$ | Drift Term           | 결정론적 드리프트 함수 | 시간이 지남에 따라 변하는 시간 의존적 함수(Time-dependent function). |

#### Multi curve (실제 구현)
- 현재 시장의 이자율 기간 구조에 대한 해석적 해.
$$\theta_{base}(t) = \frac{\partial f_F(0, t)}{\partial t} + a f_F(0, t) + \frac{\sigma^2}{2a}\left( 1 - e^{-2at} \right)$$
- 이산적인 시간 간격을 이용하여 현재 시점에서 시장의 할인채 가격으로부터 선도이자율을 산출
$$f_D(0, t) = -\frac{1}{\Delta t} \ln \left( \frac{P_D(0, t)}{P_D(0, t-\Delta t)} \right)$$
- Forward rate 와 Discount rate 사이의 베이시스 스프레드를 시점별로 계산
$$\Delta(t) = f_D(0, t) - f_f(0, t)$$
- 이산적 오차를 보정
$$\theta(t) = \theta_{base}(t) + a \Delta(t) = \frac{\partial f_F(0, t)}{\partial t} + a f_D(0, t) + \frac{\sigma^2}{2a}\left( 1 - e^{-2at} \right)$$
- 오일러-마루야마(Euler-Maruyama) 이산화 기법을 사용하여 다음 타임스텝의 무위험 할인 단기금리 경로를 무작위로 전개
$$r_{t+\Delta t}^D = r_t^D + \left( \theta(t) - a r_t^D \right) \Delta t + \sigma \sqrt{\Delta t} Z_t$$

### Libor Market Model (LMM)
#### Single curve
- **Model (SDE)**: $dF_i(t) = F_i(t) \mu_i(t) dt + F_i(t) \sigma_i dW_t^i$
- **Drift ($\mu_i(t)$)**: $\mu_i(t) = \sum_{j=0}^{i} \frac{\tau_j F_j(t)}{1 + \tau_j F_j(t)} \sigma_i \sigma_j \rho_{ij}$
- **Discount factor**: $P(t, T) = \prod_{k=\eta(t)}^{n} \frac{1}{1 + \tau_k F_k(t)}$
- **Forward rate**: $F_i(t) = F(t; T_i, T_{i+1}) \quad \left(\text{Single Forward Rate, where } P(t,T) \text{ is derived from } F_i(t)\right)$

#### Multi curve (실제 구현)
- 프록시 OIS 이산 단리 선도금리 역산
  $$f_D(t) = \frac{1}{\Delta t} \left( \frac{P_D(0, t)}{P_D(0, t+\Delta t)} - 1 \right)$$
- 테너 베이시스 스프레드 추출
  $$\Delta_i(t) = F_i(t) - f_D(t)$$
- 멀티 커브 통합 LMM 드리프트 보정
  $$\mu_i^{MC}(t) = \sum_{j=\eta(t)}^{i} \frac{\Delta t \cdot F_j(t)}{1 + \Delta t \cdot F_j(t)} \sigma_i \sigma_j \rho_{ij} + \frac{\Delta_i(t)}{1 + \Delta t \cdot F_i(t)}$$
- Lognormal 포워드 레이트 경로 전개
  $$F_i(t+\Delta t) = F_i(t) \cdot \exp\left( \left( \mu_i^{MC}(t) - \frac{1}{2}\sigma_i^2 \right) \Delta t + \sigma_i \sqrt{\Delta t} Z_t \right)$$
## 핵심 구현 기능

### LMM 가우스-뉴턴 벡터 하강
- 경사 하강법 사용시 발생할 수 있는 문제: 경사 하강법은 하나의 변동성 변수만을 이용하기에 테너간 상관관계 계산시 에러가 수렴하지 못하는 상황이 발생할 수 있음
- 이를 막기 위해서 가우스-뉴턴 벡터 하강법을 이용하여 테너별 상관관계를 유지하면서 최적화 수행

### Multi-Curve LSM Asset Valuation
- 현금흐름 예측(Forward)과 현재가치 할인(Discount)의 기준 커브를 완전히 분리하여 연산
- OIS 금리 curve를 사용해야 하지만 OpenAPI의 한계로 인하여 Forward 및 Discount 모두 국고채 1y, 5y, 10y를 사용함
- 향후 Datahandler.py에 데이터 형식을 연동하여 프록시 구현 수정 가능

### Key Rate Delta & DV01
- **구간별 인접도 가중치 범핑 (Interval Bumping)**: 각 분기별 거점 테너(Key Rate Node) 리스크를 분리 측정함
- **하이브리드 테너 동기화**: HW 모델과 LMM 모델의 1bp 민감도(DV01) 리스크 버킷을 동일한 테너 축 위에서 계산되도록 설정함

### 일반화된 가변 테너 스케쥴러 및 영업일 관습 적용
- 자동화된 가변 그리드: CSV 파일에서 years, steps, dt는 각각 만기까지 남은 년수, HW 모델 계산 단계 수, 지급 간격 (년)으로 이를 바탕으로 매 구간의 실제 영업일 비율을 계산함
- 월말 효과 적용: 한국과 미국은 각각 수정익일영업방식과 익일영업방식을 적용하였으며, 월말효과 연동 역시 적용됨
- 이종 영업일 관습 이원화: Forward rate 및 Discount rate 계산을 위해 Following Business Day Convention을 적용하였으며 국가별 세부 휴일 달력을 이원화 함

## 데이터 결과 및 분석 1. 프리미엄 가격을 이미 아는 상품의 프리미엄 가격을 재산정

- 입력 데이터는 2026년 05월 18일 기준 ECOS와 yfinance의 1년, 5년 10년 국고채의 10일분 데이터 및 가장 최근의 SOFR 및 KOFR
- 옵션은 계약 체결시 fixed leg를 3.5%로, 옵션 프리미엄을 1.25%로 각각 가정하고 변동성과 HW의 평균 회귀 파라미터를 최적화 함
- 5년동안 매 3개월 단위로 옵션 행사가 가능한 상품을 가정하고 계산 수행
- Rebonato Parametrization 의 beta = 1.0 사용

### 1. KOREA BASE RISK METRICS REPORT (KRW)
| 지표 (Metrics) | Hull-White (HW) | Libor Market Model (LMM) |
| :--- | :---: | :---: |
| **Val (평가가치)** | 3.6029 | 1.0038 |
| **Delta (델타, 1bp)** | 2.4802 | 3.5127 |
| **Gamma (감마)** | -0.0734 | -0.0362 |
| **Vega (베가)** | 1.8576 | -0.0584 |
| **HR (헤지비율)** | 0.5512 | 0.7806 |


### 2. USA BASE RISK METRICS REPORT (USD)
| 지표 (Metrics) | Hull-White (HW) | Libor Market Model (LMM) |
| :--- | :---: | :---: |
| **Val (평가가치)** | 3.1855 | 1.1710 |
| **Delta (델타, 1bp)** | 2.3653 | 3.5576 |
| **Gamma (감마)** | -0.0578 | -0.0327 |
| **Vega (베가)** | 1.8551 | -0.0674 |
| **HR (헤지비율)** | 0.5256 | 0.7906 |

---

### 3. KOREA KEY RATE DV01 DISPATCH (KRW)
| 거점 만기 (Tenor) | HW_DV01 | LMM_DV01 |
| :--- | :---: | :---: |
| **Tenor_0.25Y** | 0.000003 | 0.000025 |
| **Tenor_0.50Y** | 0.000003 | 0.000020 |
| **Tenor_0.75Y** | 0.000003 | 0.000024 |
| **Tenor_1.00Y** | 0.000008 | 0.000021 |
| **Tenor_1.25Y** | 0.000012 | 0.000022 |
| **Tenor_1.50Y** | 0.000013 | 0.000022 |
| **Tenor_1.75Y** | 0.000013 | 0.000020 |
| **Tenor_2.00Y** | 0.000013 | 0.000019 |
| **Tenor_2.25Y** | 0.000014 | 0.000020 |
| **Tenor_2.50Y** | 0.000015 | 0.000020 |
| **Tenor_2.75Y** | 0.000015 | 0.000020 |
| **Tenor_3.00Y** | 0.000015 | 0.000021 |
| **Tenor_3.25Y** | 0.000015 | 0.000019 |
| **Tenor_3.50Y** | 0.000015 | 0.000020 |
| **Tenor_3.75Y** | 0.000014 | 0.000019 |
| **Tenor_4.00Y** | 0.000014 | 0.000020 |
| **Tenor_4.25Y** | 0.000014 | -0.000000 |
| **Tenor_4.50Y** | 0.000014 | -0.000002 |
| **Tenor_4.75Y** | 0.000014 | -0.000002 |
| **Tenor_5.00Y** | 0.000006 | 0.000002 |

---

### 4. USA KEY RATE DV01 DISPATCH (USD)
| 거점 만기 (Tenor) | HW_DV01 | LMM_DV01 |
| :--- | :---: | :---: |
| **Tenor_0.25Y** | 0.000003 | 0.000025 |
| **Tenor_0.50Y** | 0.000003 | 0.000021 |
| **Tenor_0.75Y** | 0.000003 | 0.000025 |
| **Tenor_1.00Y** | 0.000008 | 0.000021 |
| **Tenor_1.25Y** | 0.000012 | 0.000021 |
| **Tenor_1.50Y** | 0.000012 | 0.000022 |
| **Tenor_1.75Y** | 0.000013 | 0.000021 |
| **Tenor_2.00Y** | 0.000013 | 0.000019 |
| **Tenor_2.25Y** | 0.000013 | 0.000020 |
| **Tenor_2.50Y** | 0.000013 | 0.000020 |
| **Tenor_2.75Y** | 0.000013 | 0.000020 |
| **Tenor_3.00Y** | 0.000013 | 0.000021 |
| **Tenor_3.25Y** | 0.000013 | 0.000020 |
| **Tenor_3.50Y** | 0.000013 | 0.000020 |
| **Tenor_3.75Y** | 0.000013 | 0.000019 |
| **Tenor_4.00Y** | 0.000013 | 0.000020 |
| **Tenor_4.25Y** | 0.000013 | -0.000000 |
| **Tenor_4.50Y** | 0.000013 | -0.000001 |
| **Tenor_4.75Y** | 0.000013 | -0.000003 |
| **Tenor_5.00Y** | 0.000007 | 0.000002 |


### 결과
- **Val (평가가치)**: 모델에서 계산한 공정가치로 실제 행사가가 3.5%, underlying swap fixed rate가 1.25%인 Callable swap의 프리미엄
#### HW와 LMM의 비교
1. **Val**: 가정에서 적용한 fair rate 1.25%를 기준으로 한국과 미국 시장 모두 HW는 고평가를, LMM은 저평가를 하고 있음.
2. **Delta**: 한국과 미국 시장 모두 LMM이 HW보다 높은 값을 보이고 있으며 1bp 변화에 따른 파생상품 가격 변화는 LMM이 높게 계산됨.
3. **Gamma**: 한국 시장과 미국 시장에서 모든 모델에서 음수를 기록하였음.
4. **Vega**: HW 모델은 한국과 미국시장 모두 1.8를 넘는 수치를 기록하였고, LMM은 음수로 나타났음.
5. **Hedge Ratio**: 한국과 미국 시장 모두 LMM이 HW보다 높은 HR를 보이고 있음.
6. **Key rate DV01**: HW 모형은 초반에 안정적이다가 향후 일정한 rate를 보여주며 LMM은 높은 변동성을 보여준 뒤 만기 직전에 음수로 전환됨

#### 한국과 미국 시장의 비교
1. HW와 LMM의 스프레드는 한국이 미국보다 높게 나타남.
2. 한국 시장과 미국 시장의 Delta 차이는 HW와 LMM이 각 0.1149, -0.0449로 큰 차이를 보이지 않음
3. 한국 시장과 미국 시장의 Gamma 차이는 HW와 LMM이 각 -0.0372, -0.0251로 큰 차이를 보이지 않음
4. 한국 시장의 미국 시장의 Vega 및 HR 역시 유의미한 차이를 보인다고 하기 힘듬
5. 두 시장에서 Key rate DV01은 거의 일치된 구조를 갖음을 확인 할 수 있음

#### 분석
1. 한국과 미국 모두 Forward curve 및 discount rate가 한국과 미국의 국고채 1y, 5y, 10y로 구성되어 있으며, 프록시의 한계로 인하여 분석 결과에 큰 차이를 보이지 않는걸로 여겨짐
2. 임의로 적용된 fair rate인 1.25%를 기준으로 HW 모델은 프리미엄 가격을 고평가, LMM은 저평가 하는 것이 확인되고 이는 LMM이 HW 모델과 다르게 테너별 변동성 상관관계와 기간 구조의 뒤틀림을 반영하였기 때문임
3. LMM의 Vega가 음수로 나오는 것은 특정 테너의 변동성이 오를때 조기 행사 확률이 급격히 높아져 기초자산의 가치를 잠식하기 때문으로 보여짐
4. Gamma가 HW, LMM 모두 음수이며 이는 옵션이 실행될 수 있음을 보여주는 주요 단서임 
5. HW 모델의 Key rate DV01은 평균 회귀를 가정하기에 단기 잉자율 변화는 장기 채권 가격에 미치는 영향이 적다고 분석하여 초기 변동성이 낮게 계산되는걸로 해석할 수 있음 
6. LMM의 Key rate DV01은 테너별 금리가 독립적이기에 단기 금리 변화의 민감도가 높게 나타나며 만기에 가까워지면 가치가 음수로 전환되며 이는 테너간 상관관계를 다요인으로 분석한 결과로 해석 할 수 있음 
7. 단순히 하나의 모델만을 분석하는 것이 아니라 다양한 모델과 지표를 이용해야 함을 확인함
---
## 데이터 결과 및 분석 2. 단일 변동성(Garch)를 이용하여 옵션 프리미엄 계산 결과
- Fixed leg = 3.5%로 가정함 (은행 수수료 배제)
- Garch 기반 변동성을 이용하여 HW 모델 및 LMM의 변동성을 결정론적으로 계산
- LMM 변동성을 사무엘슨 효과를 이용하여 계산 (Volatility Conversion Theory 적용)

### 1. KOREA BASE RISK METRICS REPORT (KRW)

| Metric | Hull-White (HW) | Libor Market Model (LMM) |
| :--- | :---: | :---: |
| **Val (Fair Value)** | 2.3684 | 0.7928 |
| **Delta (1bp)** | 2.2920 | 3.4904 |
| **Gamma** | 0.0403 | 0.0881 |
| **Vega** | 1.9957 | -0.0121 |
| **HR (Hedge Ratio)** | 0.5093 | 0.7756 |

### 2. USA BASE RISK METRICS REPORT (USD)

| Metric | Hull-White (HW) | Libor Market Model (LMM) |
| :--- | :---: | :---: |
| **Val (Fair Value)** | 1.8846 | 1.1331 |
| **Delta (1bp)** | 2.4763 | 3.5847 |
| **Gamma** | 0.0267 | 0.0523 |
| **Vega** | 1.9215 | -0.0185 |
| **HR (Hedge Ratio)** | 0.5503 | 0.7966 |

---

### 3. KOREA KEY RATE DV01 DISPATCH (KRW)

| Tenor | HW_DV01 | LMM_DV01 |
| :--- | :---: | :---: |
| **Tenor_0.25Y** | 0.000002 | 0.000019 |
| **Tenor_0.50Y** | 0.000002 | 0.000032 |
| **Tenor_0.75Y** | 0.000002 | 0.000007 |
| **Tenor_1.00Y** | 0.000006 | 0.000028 |
| **Tenor_1.25Y** | 0.000011 | 0.000021 |
| **Tenor_1.50Y** | 0.000011 | 0.000021 |
| **Tenor_1.75Y** | 0.000008 | 0.000016 |
| **Tenor_2.00Y** | 0.000009 | 0.000016 |
| **Tenor_2.25Y** | 0.000009 | 0.000010 |
| **Tenor_2.50Y** | 0.000009 | 0.000010 |
| **Tenor_2.75Y** | 0.000010 | 0.000015 |
| **Tenor_3.00Y** | 0.000010 | 0.000015 |
| **Tenor_3.25Y** | 0.000011 | 0.000017 |
| **Tenor_3.50Y** | 0.000011 | 0.000017 |
| **Tenor_3.75Y** | 0.000012 | 0.000022 |
| **Tenor_4.00Y** | 0.000012 | 0.000011 |
| **Tenor_4.25Y** | 0.000013 | -0.000003 |
| **Tenor_4.50Y** | 0.000013 | -0.000005 |
| **Tenor_4.75Y** | 0.000014 | 0.000001 |
| **Tenor_5.00Y** | 0.000006 | -0.000006 |

### 4. USA KEY RATE DV01 DISPATCH (USD)

| Tenor | HW_DV01 | LMM_DV01 |
| :--- | :---: | :---: |
| **Tenor_0.25Y** | 0.000001 | 0.000026 |
| **Tenor_0.50Y** | 0.000001 | 0.000034 |
| **Tenor_0.75Y** | 0.000001 | 0.000010 |
| **Tenor_1.00Y** | 0.000005 | 0.000026 |
| **Tenor_1.25Y** | 0.000010 | 0.000024 |
| **Tenor_1.50Y** | 0.000010 | 0.000021 |
| **Tenor_1.75Y** | 0.000010 | 0.000017 |
| **Tenor_2.00Y** | 0.000010 | 0.000017 |
| **Tenor_2.25Y** | 0.000013 | 0.000015 |
| **Tenor_2.50Y** | 0.000011 | 0.000016 |
| **Tenor_2.75Y** | 0.000010 | 0.000019 |
| **Tenor_3.00Y** | 0.000012 | 0.000018 |
| **Tenor_3.25Y** | 0.000011 | 0.000016 |
| **Tenor_3.50Y** | 0.000011 | 0.000018 |
| **Tenor_3.75Y** | 0.000012 | 0.000018 |
| **Tenor_4.00Y** | 0.000013 | 0.000018 |
| **Tenor_4.25Y** | 0.000012 | 0.000001 |
| **Tenor_4.50Y** | 0.000013 | 0.000002 |
| **Tenor_4.75Y** | 0.000014 | -0.000000 |
| **Tenor_5.00Y** | 0.000008 | -0.000004 |

### 결과
- **Val (평가가치)**: 모델에서 계산한 공정가치로 실제 행사가가 3.5%인 Callable swap의 프리미엄
#### HW와 LMM의 비교
1. **Val**: 한국과 미국 모두 HW 모델이 LMM보다 높은 프리미엄을 예측함 
2. **Delta**: 한국과 미국 시장 모두 LMM이 HW보다 높은 값을 보이고 있으며 1bp 변화에 따른 파생상품 가격 변화는 LMM이 높게 계산됨.
3. **Gamma**: 모두 양수를 보이며 negative convexity 효과를 보여주지 않음.
4. **Vega**: HW 모델은 한국과 미국시장 모두 1.9를 넘는 수치를 기록하였고, LMM은 음수로 나타났음.
5. **Hedge Ratio**: 한국과 미국 시장 모두 LMM이 HW보다 높은 HR를 보이고 있음.
6. **Key rate DV01**: HW 모형은 초반에 안정적이다가 향후 일정한 rate를 보여주며 LMM은 높은 변동성을 보여준 뒤 만기 직전에 음수로 전환됨

#### 한국과 미국 시장의 비교
1. HW와 LMM의 스프레드는 한국이 미국보다 높게 나타남.
2. 한국 시장과 미국 시장의 Delta 차이는 HW와 LMM이 각 -0.1843, -1.1923로 큰 차이를 보이지 않음
3. 한국 시장과 미국 시장의 Gamma 차이는 HW와 LMM이 각 0.0136, 0.0358로 큰 차이를 보이지 않음
4. 한국 시장의 미국 시장의 Vega 및 HR 역시 유의미한 차이를 보인다고 하기 힘듬
5. 두 시장에서 Key rate DV01은 거의 일치된 구조를 갖음을 확인 할 수 있음

#### 분석
1. 한국과 미국 모두 Forward curve 및 discount rate가 한국과 미국의 국고채 1y, 5y, 10y로 구성되어 있으며, 프록시의 한계로 인하여 분석 결과에 큰 차이를 보이지 않는걸로 여겨짐
2. HW 모델은 프리미엄 가격을 고평가, LMM은 저평가 하는 것이 확인되고 이는 LMM이 HW 모델과 다르게 테너별 변동성 상관관계와 기간 구조의 뒤틀림을 반영하였기 때문임
3. LMM의 Vega가 양수로 나오는 것은 특정 테너의 변동성이 오를때 조기 행사 확률이 급격히 높아져 기초자산의 가치를 잠식하기 때문으로 보여짐
4. Gamma가 HW, LMM 모두 양수이며 이는 Long convexity 상태임음 보여줌
5. HW 모델의 Key rate DV01은 평균 회귀를 가정하기에 단기 잉자율 변화는 장기 채권 가격에 미치는 영향이 적다고 분석하여 초기 변동성이 낮게 계산되는걸로 해석할 수 있음 
6. LMM의 Key rate DV01은 테너별 금리가 독립적이기에 단기 금리 변화의 민감도가 높게 나타나며 만기에 가까워지면 가치가 음수로 전환되며 이는 테너간 상관관계를 다요인으로 분석한 결과로 해석 할 수 있음 
7. 단순히 하나의 모델만을 분석하는 것이 아니라 다양한 모델과 지표를 이용해야 함을 확인함
---

## 성능 및 기술적 최적화 (HPC)

### [실행 성능 리포트]
*   **시뮬레이션 규모**: 시나리오당 100,000개 경로 생성.
*   **연산 범위**: 2개 국가 시장(KRW, USD)에 대해 2개 모델(HW, LMM)의 총 4가지 시나리오 동시 분석.
*   **전체 실행 시간**: 평균 **240초**
*   **포함 내역**: 실시간 데이터 수집 (ECOS 및 yfinance) + 수익률 곡선 구축 + 파라미터 최적화 + LSM + Greeks 산출.

### [하드웨어 사양]
*   **CPU**: AMD Ryzen 9 5950X (16코어, 3.4 GHz)
*   **GPU**: NVIDIA GeForce RTX 3070 (8GB VRAM, Ampere 아키텍처)
*   **OS**: Windows 10 / CUDA 13.x 기반

---

## 향후 계획
### Stochastic alpha-beta-gamma 변동성 적용
- 