# GPU-Accelerated Multi-Model Callable Swap Integrated Evaluation & Optimization Engine
> **High-Performance Monte Carlo Simulation & Dual Calibration System based on NVIDIA CUDA**

> **Language**: [English]| [한국어](./README_KR.md) 

This project transplants numerical simulation techniques from particle physics into financial engineering to calculate the value of **Bermudan Callable Swaps** in the Korean (KRW) and US (USD) markets at ultra-high speeds. By combining statistical volatility analysis (GARCH) with market-fit batch calibration, it empirically demonstrates the **Model Risk** between the single-factor model (Hull-White) and the multi-factor model (LMM).

---

## Project Structure & Module Roles
The system is modularized into four layers, from data acquisition to GPU kernel operations.


| Layer | File Name | Detailed Role |
| :--- | :--- | :--- |
| **Data** | `Datahandler.py` | **Data Gateway**: Interfaces with ECOS (Bank of Korea) and yfinance APIs. Extracts and refines time-series for volatility analysis and snapshot data for pricing. |
| | `Volatility.py` | **Statistical Analysis Engine**: Runs GARCH(1,1) and EWMA models. Estimates historical market volatility to provide an 'Initial Guess' for the optimization loop. |
| **Optimization** | `GPUOptHWPricer.py`| **Batch Computing Kernel**: Powered by CuPy RawKernel. Parallelizes dozens of parameter scenarios in a single GPU call to maximize Jacobian calculation speed. |
| | `HW_cal.py` / `LMM_cal.py`| **Dual Calibrator**: Controls the GPU batch engine to perform Gauss-Newton optimization targeting market Swaption prices. |
| **Engine** | `Model_selection.py`| **Yield Curve Construction**: Uses QuantLib to bootstrap short-term rates and build initial forward rate curves for simulation. |
| | `HW_GPU.py` / `LMM_GPU.py` | **Parallel Simulator**: Generates large-scale paths using optimized parameters. Accelerates Hull-White OU processes and LMM correlation structures via CUDA. |
| | `LSM_pricer.py` | **Early Exercise Engine**: Performs Longstaff-Schwartz (LSM) least-squares regression within the GPU to determine optimal stopping and Bermudan value. |
| **Controller** | `main.py` | **Pipeline Orchestrator**: Executes the entire workflow (Data → Vol → Calibration → Pricing → Greeks) and generates the final risk report. |

---

## Final Risk Metrics Report
Final risk metrics calculated after optimizing models to the Market Swaption Target Price (125bp).

### [KOREA REPORT (KRW)]


| Metric | Hull-White (HW) | LMM |
| :--- | :---: | :---: |
| **Val (Value)** | 3.2974 | 0.8003 |
| **Delta (1bp)** | 1.7722 | 2.4747 |
| **Gamma** | 0.0045 | -0.0405 |
| **Vega** | 1.8004 | 0.1222 |
| **HR (Hedge Ratio)** | 0.3938 | 0.5499 |

### [USA REPORT (USD)]


| Metric | Hull-White (HW) | LMM |
| :--- | :---: | :---: |
| **Val (Value)** | 3.0152 | 1.1573 |
| **Delta (1bp)** | 1.7343 | 2.4618 |
| **Gamma** | 0.0085 | -0.1497 |
| **Vega** | 1.8022 | 0.1180 |
| **HR (Hedge Ratio)** | 0.3854 | 0.5471 |

### Key Research Insights
1.  **Empirical Proof of Model Risk**: In the Korean market, a massive value spread of approx. **250bp** was observed between HW (3.2974) and LMM (0.8003). This suggests that single-factor models overestimate the structural twists of the yield curve, proving the necessity of multi-factor LMM calibration.
2.  **Capturing Negative Convexity**: **Negative Gamma (KR: -0.0405, US: -0.1497)** was clearly observed in the LMM results. This perfectly replicates the specific risk of Callable products where the probability of early exercise increases as interest rates rise, limiting price gains.
3.  **Hedge Ratio Optimization**: Realistic hedge ratios of 0.39 to 0.55 were derived. This serves as a practical indicator that a standard swap volume of approximately 40-55% of the principal is required to hedge one option contract.

---

## Performance & HPC Optimization

### [Execution Performance Report]
*   **Simulation Scale**: 100,000 paths per scenario.
*   **Scope**: Simultaneous analysis of 8 scenarios (2 Markets × 2 Models × Greeks).
*   **Total Execution Time**: **61.95s**
*   **Included Processes**: Real-time data fetching + Bootstrapping + GPU Simulation + LSM Backwards Induction + Greeks calculation.
*   **Real-time Responsiveness**: After data collection, valuation and Greek calculation for a single model complete in **under 1 second**.

### [Hardware Specifications]
*   **CPU**: AMD Ryzen 9 5950X (16-Core, 3.4 GHz)
*   **GPU**: NVIDIA GeForce RTX 3070 (8GB VRAM, Ampere Architecture)
*   **OS**: Windows 10 / CUDA 13.x based

### [Core Optimization Techniques]
*   **Zero-Copy VRAM**: Eliminated PCIe bottlenecks by processing data entirely within GPU memory—from path generation to regression—without copying back to the CPU.
*   **Batch Jacobian Computation**: Parallelized multiple scenarios for numerical differentiation into a single kernel call, reducing optimization time by over 10x compared to CPU.

---

## Limitations & Future Work
*   **Volatility Surface**: Currently optimizes up to tenor-specific volatility. Expansion to models like SABR is needed to reflect Smile/Skew phenomena.
*   **Fixed Correlation**: Assumes deterministic correlation between tenors in LMM. Dynamic correlation calibration logic should be added for volatile market conditions.
*   **Single-Curve Framework**: Upgrading to a Modern Multi-Curve (OIS-Libor Basis) bootstrapping framework is a future milestone.
