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

## Data Results and Analysis 1. Recalculation of the Premium Price for a Product with an Already Known Premium Price

- Input data consists of 10-day historical data of 1-year, 5-year, and 10-year government bonds from ECOS and yfinance as of May 18, 2026, alongside the most recent SOFR and KOFR rates.
- The volatilities and Hull-White mean reversion parameters were optimized assuming a fixed leg of 3.5% and an option premium of 1.25% at contract inception.
- Calculations were performed assuming a product where the option can be exercised every 3 months over a 5-year period.
- Beta = 1.0 from the Rebonato Parametrization was utilized.

### 1. KOREA BASE RISK METRICS REPORT (KRW)


| Metric | Hull-White (HW) | Libor Market Model (LMM) |
| :--- | :---: | :---: |
| **Val (Fair Value)** | 3.6029 | 1.0038 |
| **Delta (1bp)** | 2.4802 | 3.5127 |
| **Gamma** | -0.0734 | -0.0362 |
| **Vega** | 1.8576 | -0.0584 |
| **HR (Hedge Ratio)** | 0.5512 | 0.7806 |

### 2. USA BASE RISK METRICS REPORT (USD)


| Metric | Hull-White (HW) | Libor Market Model (LMM) |
| :--- | :---: | :---: |
| **Val (Fair Value)** | 3.1855 | 1.1710 |
| **Delta (1bp)** | 2.3653 | 3.5576 |
| **Gamma** | -0.0578 | -0.0327 |
| **Vega** | 1.8551 | -0.0674 |
| **HR (Hedge Ratio)** | 0.5256 | 0.7906 |

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
| **Tenor_0.25Y** | 0.000001 | 0.000026 |
| **Tenor_0.50Y** | 0.000001 | 0.000021 |
| **Tenor_0.75Y** | 0.000001 | 0.000025 |
| **Tenor_1.00Y** | 0.000005 | 0.000021 |
| **Tenor_1.25Y** | 0.000010 | 0.000021 |
| **Tenor_1.50Y** | 0.000010 | 0.000022 |
| **Tenor_1.75Y** | 0.000013 | 0.000021 |
| **Tenor_2.00Y** | 0.000013 | 0.000019 |
| **Tenor_2.25Y** | 0.000013 | 0.000020 |
| **Tenor_2.50Y** | 0.000011 | 0.000020 |
| **Tenor_2.75Y** | 0.000010 | 0.000020 |
| **Tenor_3.00Y** | 0.000012 | 0.000021 |
| **Tenor_3.25Y** | 0.000011 | 0.000020 |
| **Tenor_3.50Y** | 0.000011 | 0.000020 |
| **Tenor_3.75Y** | 0.000012 | 0.000019 |
| **Tenor_4.00Y** | 0.000013 | 0.000020 |
| **Tenor_4.25Y** | 0.000012 | -0.000000 |
| **Tenor_4.50Y** | 0.000013 | -0.000001 |
| **Tenor_4.75Y** | 0.000014 | -0.000003 |
| **Tenor_5.00Y** | 0.000008 | 0.000002 |

### Results
- **Val (Fair Value)**: The fair value calculated by the models, representing the premium of the Callable swap with an actual strike rate of 3.5% and an underlying swap fixed rate of 1.25%.

#### Comparison between HW and LMM
1. **Val**: Relative to the arbitrarily applied fair rate of 1.25%, the HW model overvalues while the LMM undervalues the premium in both the Korean and US markets.
2. **Delta**: In both the Korean and US markets, the LMM exhibits a higher value than the HW model, indicating that the derivative price change per 1bp shift is calculated to be higher under the LMM.
3. **Gamma**: All models in both the Korean and US markets recorded negative values.
4. **Vega**: The HW model recorded figures exceeding 1.8 in both the Korean and US markets, whereas the LMM appeared negative.
5. **Hedge Ratio**: In both the Korean and US markets, the LMM exhibits a higher HR than the HW model.
6. **Key rate DV01**: The HW model remains stable in the initial stage and exhibits a constant rate subsequently, whereas the LMM demonstrates high volatility before converting to negative values right before maturity.

#### Comparison between Korean and US Markets
1. The spread between the HW and LMM models appears higher in Korea than in the US.
2. The Delta difference between the Korean and US markets is -0.1843 and -1.1923 for the HW and LMM models respectively, showing no significant disparity.
3. The Gamma difference between the Korean and US markets is 0.0136 and 0.0358 for the HW and LMM models respectively, showing no significant disparity.
4. The Vega and HR metrics also cannot be considered to exhibit meaningful differences between the Korean and US markets.
5. It can be verified that the Key rate DV01 possesses an almost identical structure across both markets.

#### Analysis
1. In both Korea and the US, the forward curves and discount rates are composed of domestic 1y, 5y, and 10y government bonds. It is deemed that no significant differences are observed in the analysis results due to the limitations of these proxies.
2. Based on the arbitrarily applied fair rate of 1.25%, it is confirmed that the HW model overvalues the premium price while the LMM undervalues it. This is because the LMM, unlike the HW model, reflects the volatility correlations by tenor and the twist of the term structure.
3. The reason why LMM's Vega appears negative is considered to be that when the volatility of a specific tenor rises, the early exercise probability increases rapidly, thereby eroding the value of the underlying asset.
4. Gamma is negative for both HW and LMM, which serves as a major clue indicating that the option can be executed.
5. Since the HW model assumes mean reversion, it analyzes that short-term interest rate changes have a minor impact on long-term bond prices, which can be interpreted as the reason why the initial volatility is calculated low.
6. In the LMM, since interest rates for each tenor are independent, the sensitivity to short-term interest rate changes appears high. As it approaches maturity, the value converts to negative, which can be interpreted as a result of analyzing the inter-tenor correlation through a multi-factor approach.
7. Confirmed the necessity of utilizing various models and metrics rather than analyzing merely a single model.
---

## Data Results and Analysis 2. Option Premium Calculation Results Using Single Volatility (GARCH)
- Fixed leg is assumed to be 3.5% (excluding bank fees)
- HW model and LMM volatilities determined deterministically using GARCH-based volatility
- LMM volatility calculated using the Samuelson effect (Volatility Conversion Theory applied)

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

### Results
- **Val (Fair Value)**: The fair value calculated by the models, representing the premium of the Callable swap with an actual strike rate of 3.5%
#### Comparison between HW and LMM
1. **Val**: In both Korean and US markets, the HW model predicts a higher premium than the LMM.
2. **Delta**: In both Korean and US markets, the LMM shows a higher value than the HW model, meaning that the derivative price change per 1bp shift is calculated to be higher in the LMM.
3. **Gamma**: Both show positive values and do not exhibit the negative convexity effect.
4. **Vega**: The HW model recorded figures exceeding 1.9 in both Korean and US markets, while the LMM appeared as negative.
5. **Hedge Ratio**: In both Korean and US markets, the LMM exhibits a higher HR than the HW model.
6. **Key rate DV01**: The HW model is stable in the beginning phase and shows a constant rate later on, whereas the LMM shows high volatility first and then converts to negative right before maturity.

#### Comparison between Korean and US Markets
1. The spread between HW and LMM appears higher in Korea than in the US.
2. The Delta difference between the Korean and US markets shows no significant disparity, recorded at -0.1843 and -1.1923 for HW and LMM, respectively.
3. The Gamma difference between the Korean and US markets shows no significant disparity, recorded at 0.0136 and 0.0358 for HW and LMM, respectively.
4. Vega and HR also hard to be considered as having meaningful differences between the Korean and US markets.
5. It can be verified that the Key rate DV01 possesses an almost identical structure across both markets.

#### Analysis
1. In both Korea and the US, the forward curve and discount rate are composed of the domestic government bonds of 1y, 5y, and 10y. It is deemed that no significant differences are shown in the analysis results due to the limitation of proxies.
2. It is confirmed that the HW model overvalues the premium price while the LMM undervalues it. This is because the LMM, unlike the HW model, reflects the volatility correlation by tenor and the twist of the term structure.
3. The reason why LMM's Vega appears positive is considered to be that when the volatility of a specific tenor rises, the early exercise probability increases rapidly, eroding the value of the underlying asset.
4. Gamma is positive for both HW and LMM, which demonstrates a Long convexity state.
5. Since the HW model assumes mean reversion, it analyzes that short-term interest rate changes have a minor impact on long-term bond prices, which can be interpreted as the reason why the initial volatility is calculated low.
6. In the LMM, since interest rates for each tenor are independent, the sensitivity to short-term interest rate changes appears high. As it approaches maturity, the value converts to negative, which can be interpreted as a result of analyzing the inter-tenor correlation through a multi-factor approach.
7. Confirmed the necessity of utilizing various models and metrics rather than analyzing merely a single model.

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

## Future Extensions

### Counterparty Credit Risk Analytics (XVA)

Planned extension of the Monte Carlo framework toward counterparty risk analytics:

* Expected Exposure (EE)
* Potential Future Exposure (PFE)
* Credit Valuation Adjustment (CVA)
* Proxy credit curve construction using public market data
* Wrong-Way Risk (WWR) analysis
* Netting and collateral agreement simulation

### Volatility Modeling

* SABR model implementation for swaption volatility surface calibration
* Comparison between HW, LMM, and SABR-based pricing frameworks
* GPU-accelerated calibration and sensitivity analysis
---
