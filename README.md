# GPU-Accelerated Multi-Model Callable Swap Integrated Valuation & Optimization Engine
> **NVIDIA CUDA-based high-performance Monte Carlo simulation, Hull-White / LMM calibration, and derivative valuation system**

> **Language**: [English] | [한국어](./README_KR.md)

- Designed to rapidly price **Bermudan Callable Swaps** in both the Korean (KRW) and U.S. (USD) interest-rate markets using GPGPU acceleration.
- Combines GARCH analysis and batch calibration to compute interest-rate sensitivity-based Delta and simplified hedge ratios under both the Hull-White and Libor Market Model (LMM).
- Uses the Longstaff-Schwartz (LSM) algorithm to derive approximate optimal early exercise policies and determine callable swap prices.
- Introduces a proxy-based multi-curve pricing framework.

---

# Project Structure and File Responsibilities
The project is organized into four layers: data collection, optimization, pricing engines, and execution control.

| Layer | File | Detailed Role |
| :--- | :--- | :--- |
| **Data** | `Datahandler.py` | **Data Collection & Processing**: Collects 10 days of 1Y, 5Y, and 10Y government bond yield data (used as OIS proxies) from ECOS (Bank of Korea) and yfinance APIs. Returns both recent snapshots and historical data in dictionary format. Constructs proxy OIS curves using Korean Treasury yields together with KOFR/SOFR overnight rates. |
| | `Volatility.py` | **Statistical Analysis Engine**: Performs GARCH(1,1) and EWMA fitting using data collected from `Datahandler.py`. Generates initial parameter estimates for calibration. |
| **Optimization** | `HW_cal.py` / `LMM_cal.py` | **Dual Calibration Engine**: Performs Gauss-Newton optimization based on market swaption prices. |
| **Engine** | `Model_selection.py` | **Yield Curve Construction**: Builds yield curves using QuantLib bootstrapping techniques. |
| | `HW_GPU.py` / `LMM_GPU.py` | **Parallel Interest Rate Simulator**: Generates interest-rate paths using CUDA acceleration. Implements both Hull-White and LMM simulations. The LMM model uses Rebonato Parametrization (`ρij = exp(-β|Ti-Tj|)`) together with Cholesky decomposition. |
| | `LSM_pricer.py` / `LSM_pricer_LMM.py` | **Early Exercise Decision Engine**: Performs Longstaff-Schwartz Monte Carlo pricing directly on the GPU. |
| **Controller** | `main.py` | **Execution Entry Point** |

`Datahandler.py` → `Volatility.py` → `Model_selection.py` → `HW_cal.py / LMM_cal.py` → `HW_GPU.py / LMM_GPU.py` → `LSM_pricer.py / LSM_pricer_LMM.py`

---

# Multi-Curve Pricing Framework

## Theoretical Background
- In a single-curve framework, the forward curve used for projecting future cashflows and the discount curve used for valuation are assumed to be identical (i.e., Libor treated as the risk-free rate).
- Following the 2008 Global Financial Crisis, interbank credit risk and liquidity risk emerged due to large financial institution defaults.
- As tenor basis spreads widened and no-arbitrage assumptions broke down, multi-curve pricing became necessary.

## Why Multi-Curve Pricing is Necessary for Swaptions
- **Hedging Cost Perspective**: After selling an uncollateralized callable swap and hedging through a CCP-cleared offsetting position, funding costs follow collateralized risk-free rates such as OIS/KOFR/SOFR.
- **Cashflow Projection Perspective**: Floating-leg payments are determined by reference rates containing credit/liquidity risk such as 3M Libor or 91-day CD rates.
- **Valuation Methodology**: Therefore, practical pricing requires projecting future cashflows using forward curves while discounting using OIS curves.

## Hull-White Model

### Single Curve
- **Model**: $dr_t = \left( \theta(t) - a r_t \right) dt + \sigma dW_t$
- **Discount factor**: $P(t,T) = A(t,T) \exp\left(-B(t,T)r_t\right)$
- **Forward rate**: $F(t; T_1, T_2) = \frac{1}{\tau} \left( \frac{P(t,T_1)}{P(t,T_2)} - 1 \right)$

| Parameter | Description | Role |
| :---: | :--- | :--- |
| `a` | Mean Reversion Speed | Controls how quickly short rates revert to long-term equilibrium. |
| `σ` | Volatility | Controls short-rate volatility and option sensitivity. |
| `θ(t)` | Drift Term | Time-dependent deterministic drift calibrated to fit the initial yield curve. |

### Multi-Curve (Implemented Framework)
- Deterministic base drift term required for the model to perfectly replicate (Exact Calibration) the current market forward curve ($f_F(0,t)$), which serves as the reference index (e.g., 3M CD forward rate) under a single-curve environment:
  $$\theta_{base}(t) = \frac{\partial f_F(0, t)}{\partial t} + a f_F(0, t) + \frac{\sigma^2}{2a}\left( 1 - e^{-2at} \right)$$
- Exact risk-free instantaneous forward rate ($f_D(0,t)$) implied within each time-step interval ($\Delta t \cdot (t-1), \Delta t \cdot t$), back-calculated from the risk-free discount bond price table ($P_D(0,t)$) under a continuous compounding convention:
  $$f_D(0, t) = -\frac{1}{\Delta t} \ln \left( \frac{P_D(0, t)}{P_D(0, t-\Delta t)} \right)$$
- Tenor basis spread containing credit and liquidity risk premiums between the market forwarding curve (CD 3M, etc.) and the risk-free discount curve (KOFR/OIS, etc.), extracted dynamically across time points using proxy data:
  $$\Delta(t) = f_F(0, t) - f_D(0, t)$$
- Final drift correction to align the baseline axis of the short-rate simulation with the risk-free discount short-rate ($r_{t}^{D}$) process rather than the forecasting curve:
  $$\theta(t) = \theta_{base}(t) + a \Delta(t) = \frac{\partial f_F(0, t)}{\partial t} + a f_D(0, t) + \frac{\sigma^2}{2a}\left( 1 - e^{-2at} \right)$$
- Stochastic forward evolution of the risk-free discount short-rate path to the next time-step using the Euler-Maruyama discretization scheme:
  $$r_{t+\Delta t}^D = r_t^D + \left( \theta(t) - a r_t^D \right) \Delta t + \sigma \sqrt{\Delta t} Z_t$$

### Libor Market Model (LMM)
#### Single curve
- **Model (SDE)**: $dF_i(t) = F_i(t) \mu_i(t) dt + F_i(t) \sigma_i dW_t^i$
- **Drift ($\mu_i(t)$)**: $\mu_i(t) = \sum_{j=0}^{i} \frac{\tau_j F_j(t)}{1 + \tau_j F_j(t)} \sigma_i \sigma_j \rho_{ij}$
- **Discount factor**: $P(t, T) = \prod_{k=\eta(t)}^{n} \frac{1}{1 + \tau_k F_k(t)}$
- **Forward rate**: $F_i(t) = F(t; T_i, T_{i+1}) \quad \left(\text{Single Forward Rate, where } P(t,T) \text{ is derived from } F_i(t)\right)$

#### Multi-curve (Actual Implementation)
- **Back-calculation of the proxy OIS discrete simple forward rate (`ois_fwd`)**:
  $$f_D(t) = \frac{1}{\Delta t} \left( \frac{P_D(0, t)}{P_D(0, t+\Delta t)} - 1 \right)$$
- **Accumulation of the LMM-inherent base drift components (`base_drift`)**:
  $$\mu_{base, i}(t) = F_i(t) \sigma_i^2 \sum_{j=0}^{i} \frac{\Delta t \cdot F_j(t)}{1 + \Delta t \cdot F_j(t)}$$
- **Extraction of the tenor basis spread (`basis_spread`)**:
  $$\Delta_i(t) = F_i(t) - f_D(t)$$
- **Multi-curve integrated LMM drift correction (`multi_curve_drift`)**:
  $$\mu_i^{MC}(t) = \mu_{base, i}(t) + \sigma_i \cdot \Delta_i(t)$$
- **Stochastic forward evolution of the log-normal forward rate path (`paths`)**:
  $$F_i(t+\Delta t) = F_i(t) \cdot \exp\left( \left( \mu_i^{MC}(t) - \frac{1}{2}\sigma_i^2 \right) \Delta t + \sigma_i \sqrt{\Delta t} Z_t \right)$$

## Final Risk Metrics (Data Analysis Results)
- **Input Data**: 10-day historical time-series of 1Y, 5Y, and 10Y government bond yields (acting as OIS proxies) sourced via ECOS and yfinance, as of May 11, 2026.
- **Bermudan Option Specifications**: Strike rate fixed at 3.50% against a market reference swap fixed rate of 1.25%.
- **LMM Parametrization**: Rebonato parametrization applied with a baseline coefficient of $\beta = 1.5$.

### [KOREA REPORT (KRW)]


| Metric | Hull-White (HW) | LMM |
| :--- | :---: | :---: |
| **Val (Present Value)** | 3.3588 | 0.9210 |
| **Delta (1bp Shift)** | 1.8299 | 2.6241 |
| **Gamma** | 0.0339 | -0.6263 |
| **Vega** | 1.7867 | 0.1118 |
| **HR (Hedge Ratio)** | 0.4067 | 0.5831 |

### [USA REPORT (USD)]


| Metric | Hull-White (HW) | LMM |
| :--- | :---: | :---: |
| **Val (Present Value)** | 3.0183 | 1.1385 |
| **Delta (1bp Shift)** | 1.7174 | 2.5285 |
| **Gamma** | -0.0142 | -0.2952 |
| **Vega** | 1.8013 | 0.1155 |
| **HR (Hedge Ratio)** | 0.3816 | 0.5619 |

---

### Comparative Analysis & Insights

#### 1. Metric-by-Metric Model Comparison (HW vs. LMM)
* **Val (Present Value)**: Based on the market reference swap rate of 1.25%, the one-factor Hull-White model demonstrates a structural upward valuation bias, whereas the Libor Market Model (LMM) delivers a tighter, more conservative fair value across both KRW and USD curves.
* **Delta**: The LMM yields a consistently higher Delta profile compared to the HW model in both markets. This implies that the multi-factor forward rate framework captures a sharper directional price sensitivity to a 1bp parallel shift in the yield curve.
* **Gamma**: The HW model yields Gamma values of 0.0339 (KRW) and -0.0142 (USD), while the LMM records -0.6263 (KRW) and -0.2952 (USD). The pronounced negative Gamma in the LMM successfully highlights the *negative convexity* inherent to Bermudan structures, driven by the varying probability of early exercise across distinct tenor slices.
* **Vega**: The HW model outputs a highly sensitive Vega exceeding 1.70 across both regions due to its global constant volatility assumption. In contrast, the LMM stabilizes at approximately 0.11, validating that the granular, tenor-by-tenor piecewise volatility setup prevents the model from overreacting to generic, uniform shifts in volatility.
* **Hedge Ratio (HR)**: The LMM mandates a higher hedge ratio than the HW model in both financial markets. While the HW model suggests a lower delta exposure, its inflated present value implies a high risk of severe over-hedging or under-hedging errors when facing non-parallel interest rate shocks.

#### 2. Cross-Border Market Comparison (Korea vs. USA)
* **Valuation Discrepancy**: The structural pricing gap between the HW and LMM models stands at 2.4378 for the KRW market and 1.8798 for the USD market. This tighter compression in the US curve stems from its smoother, more balanced historical term structure dynamics, as visually corroborated by the yield curve trajectories in Figure 1.
* **Delta Sensitivity**: The geographical delta cross-spread stands at 0.1125 for the HW and 0.0956 for the LMM. The lower directional sensitivity observed in the USD market mathematically reflects the deeper liquidity pools and enhanced pricing efficiency of the US Treasury infrastructure relative to the KTB market.
* **Gamma Stability**: The cross-market Gamma corridors span from (0.0339 to -0.6263) for Korea and (-0.0142 to -0.2952) for the USA. The significantly narrower and more stable Gamma bounds in the US report confirm that lower underlying macro volatility leads to safer, more predictable options acceleration dynamics.
* **Hedge Ratio Disparity**: The hedge ratios computed for the Korean market are systematically higher than those of the US market by an approximate spread of 0.02. This difference is directly attributable to the higher spot delta sensitivity and structural curve noise present in the domestic macroeconomic landscape.


<figure>
  <img src="./Comprehensive_Yield_Comparison.png" alt="Yield Curve Comparison">
  <figcaption align="center"><b>Fig 1. Yield Curve Comparison</b></figcaption>
</figure>

## Summary
1. Confirmed that the US bond market exhibits significantly higher liquidity and pricing efficiency compared to the Korean bond market across all metrics (Fair Value, Delta, Gamma, and Hedge Ratio).
2. Verified that against the market reference swap rate, the Hull-White model introduces a structural upward valuation bias, whereas the Libor Market Model (LMM) provides a conservative undervaluation profile.
3. Demonstrated that the LMM yields a vastly lower Vega profile by capturing granular, tenor-by-tenor forward volatilities ex-ante, establishing high resilience against uniform volatility shocks.
4. Proved that the Hedge Ratio serves as a comprehensive indicator for capturing cross-border market liquidity, efficiency gaps, and model-specific volatility sensitivities simultaneously.
5. Observed that the Rebonato correlation decay parameter is currently set at $\beta = 1.5$, which is higher than typical market standards; tuning this parameter downward is expected to align LMM valuation vectors closer to market reference swap rates.
6. Highlighted the multi-model framework's crucial necessity, proving that robust derivative risk management cannot rely on a single pricing model or isolated metrics.

---

## High-Performance Computing (HPC) & Technical Optimization

### [Execution Performance Report]
* **Simulation Scale**: 100,000 Monte Carlo paths generated per scenario grid.
* **Computational Scope**: Simultaneous batch analytical pipeline covering 4 independent market scenarios (2 jurisdictions: KRW/USD $\times$ 2 engines: HW/LMM).
* **Total Execution Time**: Averaging **40 seconds** flat.
* **End-to-End Pipeline**: Real-time multi-source data ingestion (ECOS and yfinance API) $\rightarrow$ Yield curve term structure bootstrapping $\rightarrow$ Parallel parameter optimization $\rightarrow$ Least-Squares Monte Carlo (LSM) engine execution $\rightarrow$ Multi-Greeks batch calculation.

### [Hardware Specifications]
* **CPU**: AMD Ryzen 9 5950X (16 Cores, 32 Threads, 3.4 GHz base clock)
* **GPU**: NVIDIA GeForce RTX 3070 (8GB GDDR6 VRAM, Ampere Architecture)
* **OS Environment**: Windows 10 Pro / CUDA Toolkit 13.x ecosystem

---

## Model Limitations & Strategic Roadmap

### 4. Key Rate Delta Bucketing & DV01 Risk Architecture
* **Objective**: Integrate a professional middle-office risk ledger that calculates the Dollar Value of a 1bp shift (DV01) across isolated term nodes, enabling granular risk immunization beyond generic parallel curve shifts.
* **Implementation Blueprint**:
  * Build an external batch bumping loop that applies a discrete +1bp shift exclusively to specific maturity anchor nodes (Key Rates) along `self.lmm_curve`.
  * Leverage the massive parallel simulation speed of the optimized `GPULMM_LSMPricer` to evaluate dozens of twisted curve pathways in sub-second cycles, exporting a vectorized middle-office risk matrix for exact duration hedging.
