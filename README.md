# GPU-Accelerated Multi-Model Callable Swap Integrated Valuation and Optimization Engine
> **An asset valuation system that performs high-speed computation of Swaption values based on the HW model and the Libor Market Model (LMM)**

> **Language**: [English] | [한국어](./README_KR.md)

- Aimed at computing the value of **Callable Swaps** in the Korean (KRW) and US (USD) markets at high speed by utilizing **GPGPU (CUDA)**
- Performs parameter optimization for the Hull-White model and LMM by combining GARCH analysis and Batch Calibration
- Determines prices by approximately reflecting early exercise decisions using the Longstaff-Schwartz (LSM) algorithm
- Due to the limitations of open APIs, IRS (Forward) and OIS (Backward) rates cannot be obtained; thus, a multi-curve is implemented using 1y, 5y, and 10y government bond information as a proxy
---

## Project Structure and Detailed Role of Each File
Consists of a total of 4 layers, divided into data collection, optimization, pricing engine, and the execution file



| Layer | File Name | Detailed Role |
| :--- | :--- | :--- |
| **Data** | `Datahandler.py` | **Data Collection and Processing**: Collects 1-year, 5-year, and 10-year maturity government bond data as of the current day, along with the previous day's KOFR and SOFR overnight rates using ECOS (Bank of Korea) and yfinance APIs. Applies IRS forecasting conventions and risk-free OIS discounting conventions. |
| | `Volatility.py` | **Statistical Analysis Engine**: Performs GARCH(1,1) and EWMA fitting based on the data collected from Datahandler. Extracts initial volatility values for optimization. |
| **Optimization** | `HW_cal.py` / `LMM_cal.py` | **Calibrator**: Optimizes parameters by calibrating to the historical volatility structure using non-linear optimization algorithms (such as Levenberg-Marquardt). |
| **Engine** | `Model_selection.py` | **Yield Curve Construction**: Bootstraps the yield curve using QuantLib. |
| | `HW_GPU.py` / `LMM_GPU.py` | **Parallel Simulator**: Generates interest rate paths using the GPU based on given parameters. LMM utilizes Rebonato Parametrization ($\rho_{ij} = e^{-\beta \times \left\vert T_{i} - T_{j} \right\vert}$) and uses Cholesky decomposition. |
| | `LSM_pricer.py` / `LSM_pricer_LMM.py` | **Early Exercise Decision Engine**: Performs LSM utilizing CUDA. |
| **Controller** | `main.py` | **Execution File** |

`Datahandler.py` → `Volatility.py` → `Model_selection.py` → `HW_cal.py / LMM_cal.py` → `HW_GPU.py / LMM_GPU.py` → `LSM_pricer.py / LSM_pricer_LMM.py`

---

## Multi-Curve Pricing Application

### Theoretical Background
- Single curve considers the Forward curve, which forecasts future cash flows, and the Discount curve, which calculates them into present value, to be the same (Libor is the risk free rate)
- Due to the bankruptcy of large banks after the 2008 financial crisis, interbank credit risk and liquidity risk of funds occurred
- Due to this, the tenor basis expanded, and the introduction of multi curve became necessary to reflect this

### Reasons why multi-curve framework application is required for Callable Swap
- **Standard of hedge cost**: When constructing an opposite position after selling a Callable Swap, the funding cost of the hedge portfolio follows the risk-free rate (OIS/KOFR/SOFR)
- **Standard of cash flow**: The floating rate (Floating Leg) payment condition under the swap contract follows the Forward curve of the reference rate that includes credit/liquidity risk
- **Resolution of mismatch between hedge cost and cash flow interest rate**: Since the interest rates for calculating cash flows and hedge costs are different, they must be applied separately

### Hull-White Model
#### Single curve
- **Model**: $dr_t = \left( \theta(t) - a r_t \right) dt + \sigma dW_t$
- **Discount factor**: $P(t,T) = A(t,T) \exp\left(-B(t,T)r_t\right)$
- **Forward rate**: $F(t; T_1, T_2) = \frac{1}{\tau} \left( \frac{P(t,T_1)}{P(t,T_2)} - 1 \right)$


| Parameter Symbol | English Name | Korean Name | Main Role and Characteristic |
| :---: |:---------------------|:-------------|:---------------------------------------------------------------------------------------------------------------------------------|
| $a$ | Mean Reversion Speed | 평균 회귀 속도     | A constant that determines the speed at which the short rate returns to the long-term average level ($\theta(t)/a$). The larger the value, the more strongly the interest rate is pulled toward the average, and the smaller the fluctuation range of future interest rates becomes.   |
| $\sigma$ | Volatility           | 단기금리 변동성     | A constant that adjusts the magnitude of the random shock applied to the short rate.    |
| $\theta(t)$ | Drift Term           | 결정론적 드리프트 함수 | A time-dependent function that changes over time. |

#### Multi curve (Actual Implementation)
- Analytic solution for the current market's interest rate term structure.
$$\theta_{base}(t) = \frac{\partial f_F(0, t)}{\partial t} + a f_F(0, t) + \frac{\sigma^2}{2a}\left( 1 - e^{-2at} \right)$$
- Calculates the forward rate from the market's zero-coupon bond price at the current point in time using discrete time intervals
$$f_D(0, t) = -\frac{1}{\Delta t} \ln \left( \frac{P_D(0, t)}{P_D(0, t-\Delta t)} \right)$$
- Calculates the basis spread between the Forward rate and Discount rate by time period
$$\Delta(t) = f_D(0, t) - f_f(0, t)$$
- Corrects discrete error
$$\theta(t) = \theta_{base}(t) + a \Delta(t) = \frac{\partial f_F(0, t)}{\partial t} + a f_D(0, t) + \frac{\sigma^2}{2a}\left( 1 - e^{-2at} \right)$$
- Randomly develops the risk-free discount short rate path of the next time step using the Euler-Maruyama discretization technique
$$r_{t+\Delta t}^D = r_t^D + \left( \theta(t) - a r_t^D \right) \Delta t + \sigma \sqrt{\Delta t} Z_t$$

### Libor Market Model (LMM)
#### Single curve
- **Model (SDE)**: $dF_i(t) = F_i(t) \mu_i(t) dt + F_i(t) \sigma_i dW_t^i$
- **Drift ($\mu_i(t)$)**: $\mu_i(t) = \sum_{j=0}^{i} \frac{\tau_j F_j(t)}{1 + \tau_j F_j(t)} \sigma_i \sigma_j \rho_{ij}$
- **Discount factor**: $P(t, T) = \prod_{k=\eta(t)}^{n} \frac{1}{1 + \tau_k F_k(t)}$
- **Forward rate**: $F_i(t) = F(t; T_i, T_{i+1}) \quad \left(\text{Single Forward Rate, where } P(t,T) \text{ is derived from } F_i(t)\right)$

#### Multi curve (Actual Implementation)
- **Inverse calculation of proxy OIS discrete simple forward rate**
  $$f_D(t) = \frac{1}{\Delta t} \left( \frac{P_D(0, t)}{P_D(0, t+\Delta t)} - 1 \right)$$
- **Extraction of tenor basis spread**
  $$\Delta_i(t) = F_i(t) - f_D(t)$$
- **Multi-curve integrated LMM drift correction**
  $$\mu_i^{MC}(t) = \sum_{j=\eta(t)}^{i} \frac{\Delta t \cdot F_j(t)}{1 + \Delta t \cdot F_j(t)} \sigma_i \sigma_j \rho_{ij} + \frac{\Delta_i(t)}{1 + \Delta t \cdot F_i(t)}$$
- **Lognormal forward rate path development**
  $$F_i(t+\Delta t) = F_i(t) \cdot \exp\left( \left( \mu_i^{MC}(t) - \frac{1}{2}\sigma_i^2 \right) \Delta t + \sigma_i \sqrt{\Delta t} Z_t \right)$$

## Key Implementation Features

### LMM Gauss-Newton Vector Descent
- Problem that can occur when using gradient descent: Since gradient descent utilizes only a single volatility variable, a situation may arise where the error fails to converge when calculating correlations between tenors
- To prevent this, optimization is performed while maintaining correlations by tenor using the Gauss-Newton vector descent method

### Multi-Curve LSM Asset Valuation
- Operates by completely separating the reference curves for cash flow forecasting (Forward) and present value discounting (Discount)
- Although the OIS interest rate curve should be used, due to the limitations of open APIs, 1y, 5y, and 10y government bonds are used for both Forward and Discount
- The proxy implementation can be modified in the future by linking the data format to Datahandler.py

### Key Rate Delta & DV01
- **Interval Bumping by Section**: Separately measures the risk of the base tenor (Key Rate Node) for each quarter
- **Hybrid Tenor Synchronization**: Configures the 1bp sensitivity (DV01) risk buckets of the HW model and LMM model to be calculated on the same tenor axis

### Application of Generalized Variable Tenor Scheduler and Business Day Conventions
- Automated variable grid: In the CSV file, years, steps, and dt represent the remaining years to maturity, the number of calculation steps for the HW model, and the payment interval (years) respectively, and based on this, the actual business day fraction of each section is calculated
- Application of end-of-month effect: Modified Following Business Day Convention and Following Business Day Convention are applied to Korea and the US respectively, and the end-of-month effect linkage is also applied
- Dualization of heterogeneous business day conventions: The Following Business Day Convention was applied for calculating the Forward rate and Discount rate, and the detailed holiday calendars by country were dualized

## Data Results and Analysis (Final Risk Metrics)
- Input data consists of 10 days of data for 1-year, 5-year, and 10-year government bonds from ECOS and yfinance as of May 18, 2026, and the most recent SOFR and KOFR
- The Bermudan option is set with a strike rate of 3.5% and a market benchmark swap rate of 1.25%
- The calculation is performed assuming a product where the option can be exercised every 3 months for 5 years
- Applies beta = 1.0 of the Rebonato Parametrization

### 1. KOREA BASE RISK METRICS REPORT (KRW)
| Metrics         | Hull-White (HW) | Libor Market Model (LMM) |
|:----------------| :---: | :---: |
| **Val**         | 3.6029 | 1.0038 |
| **Delta (1bp)** | 2.4802 | 3.5127 |
| **Gamma**       | -0.0734 | -0.0362 |
| **Vega**        | 1.8576 | -0.0584 |
| **HR**         | 0.5512 | 0.7806 |

---
### 2. USA BASE RISK METRICS REPORT (USD)
| Metrics         | Hull-White (HW) | Libor Market Model (LMM) |
|:----------------| :---: | :---: |
| **Val**         | 3.1855 | 1.1710 |
| **Delta (1bp)** | 2.3653 | 3.5576 |
| **Gamma**       | -0.0578 | -0.0327 |
| **Vega**        | 1.8551 | -0.0674 |
| **HR**          | 0.5256 | 0.7906 |

---

### 3. KOREA KEY RATE DV01 DISPATCH (KRW)
| Tenor | HW_DV01 | LMM_DV01 |
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
| Tenor | HW_DV01 | LMM_DV01 |
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

---

### Results
- **Val (Valuation Value)**: The fair value calculated by the model, representing the value of the Callable swap where the actual strike rate is 3.5% and the underlying swap fixed rate is 1.25%
#### Comparison between HW and LMM
1. **Val**: Based on the underlying swap fixed rate of 1.25%, HW overvalues while LMM undervalues in both the Korean and US markets.
2. **Delta**: In both the Korean and US markets, LMM shows higher values than HW, and the derivative price change per 1bp shift is calculated to be higher in LMM.
3. **Gamma**: A negative value was recorded in all models for both the Korean and US markets.
4. **Vega**: The HW model recorded figures exceeding 1.8 in both the Korean and US markets, while LMM appeared as a negative value.
5. **Hedge Ratio**: In both the Korean and US markets, LMM shows a higher HR than HW.
6. **Key rate DV01**: The HW model is stable in the beginning and shows a constant rate afterward, while LMM shows high volatility and then shifts to a negative value right before maturity.

#### Comparison between Korean and US Markets
1. The spread between HW and LMM appears higher in Korea than in the US.
2. The Delta differences between the Korean and US markets are -1.0325 and -1.1923 for HW and LMM respectively, showing no significant difference.
3. The Gamma differences between the Korean and US markets are -0.0372 and -0.0251 for HW and LMM respectively, showing no significant difference.
4. It is difficult to say that Vega and HR of the Korean market and the US market show any meaningful difference.
5. It can be confirmed that the Key rate DV01 has a nearly identical structure in both markets.

#### Analysis
1. In both Korea and the US, the Forward curve and discount rate consist of Korean and US 1y, 5y, and 10y government bonds, and it is considered that they do not show a large difference in the analysis results due to the limitations of the proxy.
2. Based on the arbitrarily applied market rate of 1.25%, it is confirmed that the HW model overvalues the bond price and LMM undervalues it, which is because LMM reflects the volatility correlations by tenor and the twisting of the term structure, unlike the HW model.
3. The reason LMM's Vega comes out negative is considered to be that when the volatility of a specific tenor rises, the early exercise probability sharply increases, eroding the value of the underlying asset.
4. Gamma is negative for both HW and LMM, which is a major clue showing that the option can be exercised.
5. Since the Key rate DV01 of the HW model assumes mean reversion, it analyzes that short-term interest rate changes have a small impact on long-term bond prices, which can be interpreted as the initial volatility being calculated low.
6. Since the tenors' interest rates are independent in the Key rate DV01 of LMM, the sensitivity to short-term interest rate changes appears high, and as it approaches maturity, the value converts to a negative value, which can be interpreted as a result of analyzing the correlations between tenors with multiple factors.
7. Confirmed that rather than analyzing just a single model, various models and indicators must be utilized.
---

## Performance and Technical Optimization (HPC)

### [Execution Performance Report]
*   **Simulation Scale**: Generated 100,000 paths per scenario.
*   **Computation Scope**: Simultaneous analysis of a total of 4 scenarios across 2 models (HW, LMM) for 2 country markets (KRW, USD).
*   **Total Execution Time**: Average of **240 seconds**
*   **Included Contents**: Real-time data collection (ECOS and yfinance) + Yield curve construction + Parameter optimization + LSM + Greeks calculation.

### [Hardware Specifications]
*   **CPU**: AMD Ryzen 9 5950X (16 Cores, 3.4 GHz)
*   **GPU**: NVIDIA GeForce RTX 3070 (8GB VRAM, Ampere architecture)
*   **OS**: Based on Windows 10 / CUDA 13.x
* 
---
