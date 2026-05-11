# GPU-Accelerated Multi-Model Callable Swap Valuation Engine
> **Language**: [English] | [한국어](./README_KR.md)

### Cross-Border Strategic Risk Analysis: KOREA vs. USA

A high-performance quantitative framework for pricing **Bermudan Callable Swaps** using **NVIDIA GPU (CUDA)**. It simultaneously evaluates KRW and USD markets through Hull-White and LMM models, delivering sub-second Greeks and sophisticated cross-market risk insights.

## Project Structure
```text
.
├── main.py                 # Core Orchestrator: Data -> Simulation -> Comparison Report
├── Datahandler.py          # ETL: Live BOK ECOS (KR) & yfinance (US) API Integration
├── Model_selection.py      # Engine: Yield Curve Bootstrapping (QuantLib)
├── HW_GPU.py               # CUDA Kernel: Short-rate path generation (PyCUDA)
├── LMM_GPU.py              # Parallel Simulator: LMM with Cholesky correlation (CuPy)
├── LSM_pricer.py           # GPU LSM (HW): Optimal stopping for Short-rate models
└── LSM_pricer_LMM.py       # GPU LSM (LMM): Optimized for Multi-dimensional Forward rates
```

## Comparative Market Analysis (KRW vs. USD)

The engine identifies critical risk discrepancies between low-yield (KR) and high-yield (US) regimes.

### [Integrated Risk Metrics Report]


| Market | Model | Value (%) | Delta (1bp) | Gamma | Vega (1%) | Hedge Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **KOREA** | **HW** | 2.2712% | 0.0186 | 0.0622 | 1.7767 | 0.4144 |
| (KRW) | **LMM** | 0.6221% | 0.0221 | 0.1531 | 0.0164 | 0.4917 |
| **USA** | **HW** | 2.1705% | 0.0184 | -0.0032 | 1.7891 | 0.4093 |
| (USD) | **LMM** | 1.0089% | 0.0232 | -0.9432 | 0.0159 | 0.5149 |

### Cross-Market Insights & Strategic Analysis

1. **Quantifying Model Risk (HW-LMM Spread)**:
   - In the KOREA market, a significant **164.9bp spread** exists between HW and LMM.
   - **Insight**: The Hull-White (one-factor) model tends to severely overestimate option premiums in low-rate environments by failing to account for yield curve structural twists, which the LMM captures with higher precision.

2. **Empirical Evidence of Negative Convexity (USA)**:
   - The USA market exhibits **Negative Gamma** across both models (-0.0032 for HW, -0.9432 for LMM).
   - **Insight**: As USD rates approach the strike (3.5%), the LMM captures a sharp "Gamma Flip." This signals a **Negative Convexity** regime where the option's value sensitivity plateaus, requiring sophisticated dynamic hedging as the exercise probability saturates.

3. **Delta Sensitivity & Hedge Efficiency**:
   - Despite lower overall valuation, the **LMM consistently yields higher Delta and Hedge Ratios** (approx. 0.49-0.51) compared to HW.
   - **Insight**: To maintain delta-neutrality, the engine suggests a hedge ratio of ~50% of the notional. The LMM’s higher sensitivity highlights its responsiveness to multi-tenor forward rate movements that simple short-rate models miss.

4. **GPU Operational Alpha**:
   - Benchmarked execution: **8 complex MC scenarios** (2 Markets × 2 Models × Greeks) completed in **< 26 seconds** on the RTX 3070.
   - **Impact**: Enables real-time risk rebalancing and intra-day stress testing that is computationally prohibitive on legacy CPU-based systems.
   - 
## Performance & Hardware Benchmark

The engine is optimized for high-end consumer-grade hardware, achieving professional-grade throughput in large-scale Monte Carlo simulations.

### [Hardware Environment]
- **CPU**: AMD Ryzen 9 5950X (16-Core, 3.4 GHz)
- **GPU**: NVIDIA GeForce RTX 3070 (8GB VRAM, Ampere Architecture)
- **OS**: Windows 10 / CUDA 13.0

### [Execution Performance]
- **Simulation Capacity**: 100,000 paths per scenario.
- **Computation Scope**: Full risk analysis for 2 markets (KRW, USD) across 2 models (HW, LMM) involving 8 simultaneous simulations.
- **Total Execution Time**: **~25.20 seconds**
  - Includes: Live Data Fetching + Yield Curve Bootstrapping + GPU Simulation + LSM Backward Induction + Greeks Calculation.
- **Real-time Capability**: Core valuation and Greeks for a single model finish in **sub-second** time after data ingestion.

## Technical Optimization
- **Zero-Copy VRAM Pipeline**: Full path-to-regression execution within GPU memory.
- **Explicit Memory Clearing**: Mitigates `cudaErrorInvalidValue` by forcing `free_all_blocks()` and `gc.collect()` between simulation loops.
- **Correlation Engineering**: GPU-based Cholesky Decomposition for realistic multi-tenor LMM evolution.

## Limitations & Future Work

While the engine demonstrates high technical performance, several quantitative limitations exist due to data constraints:

1. **Absence of Volatility Surface (Smile/Skew)**:
   - **Current State**: The engine uses a constant (flat) volatility parameter ($\sigma$).
   - **Impact**: It does not account for the **Volatility Smile or Skew** observed in real swaption markets. This may lead to mispricing for Deep ITM or OTM options.
   - **Improvement**: Future versions could integrate **SABR or Local Volatility models** to calibrate against a full Volatility Cube.

2. **Deterministic Mean Reversion**:
   - The Hull-White mean reversion speed ($a$) and LMM correlation parameters are currently static. Real-time calibration to the market's term structure of volatility is required for industrial-grade hedging.

3. **Basis Risk**:
   - The engine assumes a single-curve framework. Modern market practice requires **Multi-Curve Bootstrapping** (separating Tenor Basis and Discounting curves), which is not yet implemented.

4. **Data Latency**:
   - Integration with ECOS and yfinance provides daily closing rates. For high-frequency trading risk management, integration with professional terminals (Bloomberg/Refinitiv) for tick-by-bit data is necessary.
