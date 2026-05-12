# GPU-Accelerated Multi-Model Callable Swap Valuation & Optimization Engine

**High-Performance Monte Carlo Simulation and HW/LMM Optimization System based on NVIDIA CUDA**

**Language**: [English] | [한국어](./README_KR.md)

- This project aims to calculate the value of **Bermudan Callable Swaps** in South Korean (KRW) and US (USD) markets at high speed using GPGPU.
- It combines GARCH analysis and Batch Calibration to compute Greeks and Hedge Ratios for both Hull-White and Libor Market Models (LMM).
- It optimizes volatility values for pricing using the Longstaff-Schwartz (LSM) method.

## Project Structure & File Descriptions

The system consists of 4 layers: Data Collection, Optimization, Pricing Engine, and Execution.


| Layer            | File                                  | Detailed Role                                                                                                                                                                                            |
|:-----------------|:--------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Data**         | `Datahandler.py`                      | **Data Collection & Processing**: Collects 10 days of data for 1Y, 5Y, and 10Y treasury bonds via BOK ECOS and yfinance APIs. Returns the latest snapshot and historical data in dictionary format.      |
|                  | `Volatility.py`                       | **Statistical Analysis Engine**: Performs GARCH(1,1) and EWMA fitting based on data from Datahandler. Generates initial values for future optimization.                                                  |
| **Optimization** | `HW_cal.py` / `LMM_cal.py`            | **Dual Calibrator**: Performs Gauss-Newton optimization based on given Swaption prices.                                                                                                                  |
| **Engine**       | `Model_selection.py`                  | **Yield Curve Construction**: Bootstraps the yield curve using QuantLib.                                                                                                                                 |
|                  | `HW_gpu.py` / `LMM_gpu.py`            | **Parallel Simulator**: Generates interest rate paths using given parameters. Calculates Hull-White and LMM via CUDA. LMM utilizes Rebonato Parametrization ($\rho_{ij} = e^{-\beta \times \left\vert T_{i} - T_{j} \right\vert}$) and Cholesky decomposition. |
|                  | `LSM_pricer.py` / `LSM_pricer_LMM.py` | **Early Exercise Engine**: Performs Longstaff-Schwartz Method (LSM) within the GPU.                                                                                                                      |
| **Controller**   | `main.py`                             | **Execution File**                                                                                                                                                                                       |

### Execution Workflow
`Datahandler.py` → `Volatility.py` → `Model_selection.py` → `HW_cal.py / LMM_cal.py` → `HW_gpu.py / LMM_gpu.py` → `LSM_pricer.py / LSM_pricer_LMM.py`

## Final Risk Metrics Analysis

- **Input Data**: 10 days of 1Y, 5Y, and 10Y treasury bond data as of May 11, 2026 (ECOS & yfinance).
- **Bermudan Option**: Strike price set at 3.5%, Target price at 1.25%.
- **LMM Setting**: Beta = 1.5 used for Rebonato Parametrization.

### [KOREA REPORT (KRW)]

| Metric               | Hull-White (HW) |   LMM   |
|:---------------------|:---------------:|:-------:|
| **Val (Fair Value)** |     3.3588      | 0.9210  |
| **Delta (1bp)**      |     1.8299      | 2.6241  |
| **Gamma**            |     0.0339      | -0.6263 |
| **Vega**             |     1.7867      | 0.1118  |
| **HR (Hedge Ratio)** |     0.4067      | 0.5831  |

### [USA REPORT (USD)]

| Metric               | Hull-White (HW) |   LMM   |
|:---------------------|:---------------:|:-------:|
| **Val (Fair Value)** |     3.0183      | 1.1385  |
| **Delta (1bp)**      |     1.7174      | 2.5285  |
| **Gamma**            |     -0.0142     | -0.2952 |
| **Vega**             |     1.8013      | 0.1155  |
| **HR (Hedge Ratio)** |     0.3816      | 0.5619  |

### Analysis Results
- **Val (Fair Value)**: Theoretical fair value calculated by the model for a Callable Swap with a 3.5% strike and 1.25% market target price.

#### HW vs. LMM Comparison
1. **Val**: Based on the 1.25% target, HW tends to overvalue while LMM undervalues in both markets.
2. **Delta**: LMM shows higher Delta values than HW in both markets, indicating LMM is more sensitive to 1bp changes in interest rates.
3. **Gamma**: HW recorded 0.0339 (KR) and -0.0142 (US), while LMM recorded -0.6263 (KR) and -0.2952 (US), confirming **negative convexity** due to early exercise probability.
4. **Vega**: HW showed high sensitivity (>1.1%), whereas LMM remained stable at 0.11, as LMM pre-reflects tenor-specific volatilities.
5. **Hedge Ratio**: LMM requires a higher HR than HW. Despite HW's lower HR, its overvalued price suggests significant potential for error during large market shifts.

#### Market Comparison (KR vs. US)
1. The price gap between HW and LMM is larger in Korea (2.4378) than in the US (1.8798), as US interest rate movements are relatively more stable (as seen in Fig 1).
2. The Delta gap between HW and LMM is 0.1125 (KR) and 0.0956 (US), reflecting higher liquidity and efficiency in the US market.
3. The Gamma range for KR (0.0339 to -0.6263) is wider than for US (-0.0142 to -0.2952), further proving higher volatility stability in the US market.
4. Vega differences between markets are minimal compared to the model gap, due to LMM's tenor-specific volatility calibration.
5. The KR Hedge Ratio is approx. 200bp higher than the US, attributed to differences in market liquidity and Delta sensitivity.

![Yield Curve Comparison](./Comprehensive_Yield_Comparison.png)
**Fig 1. Yield Curve Comparison**
## Summary of Findings

1. **Market Efficiency**: Confirmed that the **US bond market possesses higher liquidity and efficiency** compared to the Korean market across all metrics (Price, Delta, Gamma, and HR).
2. **Model Bias**: Based on the 1.25% market target, the **Hull-White model consistently overvalues** the swap, while the **LMM tends to undervalue** it.
3. **Volatility Stability**: LMM exhibits significantly lower Vega (~0.11) compared to HW, proving that **LMM is more robust** against simple volatility shifts due to its tenor-specific calibration.
4. **Hedge Ratio Insights**: The HR effectively captures the differences in liquidity and efficiency between the US and Korean markets, as well as the models' sensitivity to volatility.
5. **Correlation Impact (Beta)**: The current **Beta value of 1.5** in LMM is set higher than typical market levels. Adjusting this parameter is expected to bring LMM valuation closer to the 1.25% market target.
6. **Multi-Model Necessity**: The results emphasize the importance of utilizing **multiple models and diverse risk metrics** rather than relying on a single valuation approach.

## Performance & Technical Optimization (HPC)

### [Execution Performance Report]
- **Simulation Scale**: 100,000 paths per scenario.
- **Computation Scope**: Simultaneous analysis of 4 scenarios (2 markets x 2 models).
- **Total Execution Time**: Average **40 seconds**.
- **Process Includes**: Real-time data ingestion → Yield curve bootstrapping → Parameter optimization → LSM pricing → Greeks calculation.

### [Hardware Specifications]
- **CPU**: AMD Ryzen 9 5950X (16-Core, 3.4 GHz)
- **GPU**: NVIDIA GeForce RTX 3070 (8GB VRAM, Ampere Architecture)
- **OS**: Windows 10 / CUDA 13.x

## Limitations & Future Work
- **Volatility Surface**: Currently requires manual entry for strike and target prices; needs automatic market product characteristic updates.
- **Correlation Model**: Uses a simple Rebonato Parametrization for LMM; lacks an optimization process for the Beta value.
- **Single-Curve Framework**: Integration of Multi-Curve (OIS-Libor Basis) bootstrapping is required.
