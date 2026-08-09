# EV Battery Degradation Prediction via EFC Bridge Pipeline

This repository implements a **end-to-end data science pipeline** to estimate lithium-ion battery degradation in electric vehicles (EVs), bridging simulated driving data with laboratory cell aging data. The central methodological contribution is the use of **Equivalent Full Cycles (EFC)** as a dimensionless bridging variable between two physically incompatible scales: the laboratory cell (~4.85 Ah) and the real vehicle pack (~32–80 kWh).

---

## Project Overview

The primary objective is to demonstrate that expected battery degradation can be estimated from a vehicle's driving profile alone — without longitudinal field data — by connecting two public datasets that share no direct key through dimensionless physical quantities.

### Datasets

| Dataset | Source | Description |
|---|---|---|
| **d-EVD** (Vicomtech) | [GitHub](https://github.com/Vicomtech/d-EVD_dual-Electric-Vehicle-Dataset) | SUMO simulations of 21 inter-city routes in the Basque Country, 15 vehicle × driving style profiles |
| **INR21700-M50T Aging** | [ScienceDirect](https://www.sciencedirect.com/article/pii/S2352340922002062) | 23-month aging campaign on 10 cells (Pozzato, Allam & Onori, 2022) |
| **Table 3 (paper)** | Manually transcribed | `diag_number → cycle_count` and `charge_rate` mapping per cell |

### Phase 1 Scope

Focus restricted to the **5 normal driving style vehicles**:

| vehID | Model | Nominal Capacity |
|---|---|---|
| EV1  | BMW i3   | ~39 kWh |
| EV4  | VW ID3   | ~58 kWh |
| EV7  | VW ID4   | ~77 kWh |
| EV10 | VW eUp   | ~32 kWh |
| EV13 | SUV      | ~80 kWh |

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 0  │  Setup: folder structure + table3 (metadata)        │
└────────────────────────────┬────────────────────────────────────┘
                             │
          ┌──────────────────┴───────────────────┐
          ▼                                       ▼
┌─────────────────────┐               ┌───────────────────────┐
│  Phase 1            │               │  Phase 2              │
│  Battery Processing │               │  EV Route Processing  │
│                     │               │                       │
│  Coulomb counting   │               │  Feature engineering  │
│  SoH per cell       │               │  per (vehID, route)   │
│  EFC_lab            │               │  DoD_equivalent       │
│                     │               │  C_rate_mean          │
│  battery_soh_       │               │  pct_uphill, etc.     │
│  curve_corrigido.csv│               │  ev_trip_features.csv │
└──────────┬──────────┘               └──────────┬────────────┘
           │                                      │
           └──────────────┬───────────────────────┘
                          ▼
              ┌───────────────────────┐
              │  Phase 3             │
              │  EFC Bridge          │
              │                      │
              │  EFC_vehicle =       │
              │  Σ DoD_equivalent    │
              │                      │
              │  SoH(EFC) applied    │
              │  → soh_predicted     │
              │                      │
              │  predicted_          │
              │  degradation.csv     │
              └──────────┬───────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌─────────────────────┐     ┌───────────────────────┐
│  Phase 4 — Model B  │     │  Phase 4 — Model A    │
│                     │     │                       │
│  SoH(EFC) curve     │     │  XGBoost + LOO-CV     │
│  B1: linear         │     │  route features       │
│  B2: exponential ✓  │     │  → DoD_equivalent     │
│  GPR: discarded     │     │  vehID as dummy       │
│                     │     │                       │
│  k = -0.000481      │     │  R² = 0.955 (w/dummy) │
│  R² = 0.860         │     │  R² = 0.858 (w/o)     │
└──────────┬──────────┘     └──────────┬────────────┘
           │                           │
           └──────────────┬────────────┘
                          ▼
              ┌───────────────────────┐
              │  Phases 5 & 6        │
              │                      │
              │  Validation (3 checks│
              │  Visualizations      │
              │  README              │
              └──────────────────────┘
```

---

## Physical Concepts and Nomenclature

| Concept | Formal Definition | Unit |
|---|---|---|
| **SoH** (State of Health) | `discharged_Ah(diag) / discharged_Ah(diag=1)` | dimensionless [0, 1] |
| **DoD** (Depth of Discharge) | Fraction of capacity discharged in a single cycle | dimensionless [0, 1] |
| **DoD_equivalent** | `net_energy_Wh / nominal_capacity_Wh` | dimensionless [0, 1] |
| **EFC_lab** | `0.6 × cycle_count` (fixed DoD from UDDS protocol) | equivalent cycles |
| **EFC_vehicle** | `Σ DoD_equivalent` across all trips | equivalent cycles |
| **C-rate** | `current / nominal_capacity` | h⁻¹ |
| **Capacity fade** | `1 − SoH` | dimensionless |
| **EOL** (End of Life) | Conventionally `SoH = 0.80` in EV applications | — |

The key methodological insight is that **DoD and C-rate are ratios**, not absolute values — enabling direct comparison between a 4.85 Ah lab cell and a 39,000 Wh vehicle pack, provided each quantity is normalized by its own capacity.

---

## Repository Structure

```
project/
├── data/
│   ├── raw/                          ← untouched source data, never modified
│   ├── processed/
│   │   ├── battery/
│   │   │   ├── capacity_test_10cells.csv
│   │   │   ├── table3_diagnostic_cycles.csv
│   │   │   └── battery_soh_curve_corrigido.csv   ← Phase 1 output
│   │   ├── ev/
│   │   │   ├── ev_routes/            ← d-EVD route files
│   │   │   └── ev_trip_features.csv  ← Phase 2 output
│   │   ├── bridge/
│   │   │   └── predicted_degradation.csv         ← Phase 3 output
│   │   └── models/
│   │       ├── a2_xgb_predictions.csv
│   │       └── a2_xgb_results.csv
│   └── output/                       ← final figures and reports
├── src/
│   ├── fase1_battery_processing.py
│   ├── fase2_ev_processing.py
│   ├── fase3_efc_bridge.py
│   ├── fase4_a2_xgboost.py
│   └── fase5_6_validation_output.py
└── README.md
```

---

## Detailed Methodology

### Phase 1 — Battery Dataset Processing

The `capacity_test_10cells.csv` file contains complete discharge curves (C/20) for 10 cells recorded across multiple diagnostic checkpoints (`diag_number`). For each `(cell, diag_number)` pair:

1. **Coulomb counting**: integration of `|current_A|` over `time_s` via the trapezoidal rule → `discharged_capacity_Ah`
2. **SoH**: normalized against the cell's own `diag_number=1` (pristine state) — absorbs manufacturing variation between cells
3. **Merge with Table 3**: brings `cycle_count` and `charge_rate` per diagnostic (metadata absent from the CSV)
4. **Extrapolation**: 36 diagnostics beyond the published Table 3 scope (the actual campaign ran for 23 months) were extrapolated using the last known inter-diagnostic interval per cell, flagged as `cycle_count_estimated=True`
5. **EFC_lab**: `0.6 × cycle_count` (the UDDS protocol discharges exactly 60% per cycle)

> **Excluded cells**: W3 (anomalous HPPC impedance) and W7 (inconsistent EIS measurements), as documented by the dataset authors. Retained in the CSV with `dismissed_cell=True`.

**Current sign convention**: the paper states negative current = discharge, but the `*_dis_*` files show positive current. For `capacity_test_10cells` (pure, continuous discharge), the ambiguity was resolved by integrating `|current_A|`.

### Phase 2 — d-EVD Route Processing

For each route file (21 total), filtering `vehID ∈ {EV1, EV4, EV7, EV10, EV13}`:

| Feature | Calculation | Source Column |
|---|---|---|
| `nominal_capacity_Wh` | `actualBatteryCapacity` at first step | `actualBatteryCapacity(Wh)` |
| `DoD_pct` | `100 − SoC_final` | `SoC(%)` |
| `net_energy_Wh` | `totalEnergyConsumed − totalEnergyRegenerated` (last step) | both columns |
| `DoD_equivalent` | `net_energy_Wh / nominal_capacity_Wh` | derived |
| `duration_h` | `n_steps × 1.0s / 3600` | `step` (STEP_DURATION=1s assumed) |
| `C_rate_mean` | `net_energy_Wh / nominal_capacity_Wh / duration_h` | derived |
| `aggressive_events_per_km` | `Σ(|acceleration| > 2.0 m/s²) / completedDistance_km` | `acceleration(m/s²)` |
| `pct_uphill` | `mean(slope > 0) × 100` | `slope(º)` |

Route metadata (`origin`, `destination`, `traffic_factor`, `occupancy`, `auxiliaries`, `wind`) are parsed via regex from the filename pattern `{idx}_{origin}_{destination}_{trafficFactor}_{occupancy}_{auxiliaries}_{wind}[_output].csv`.

### Phase 3 — EFC Bridge

The absence of a direct join key between the two datasets is resolved by using EFC as a common dimensionless variable:

```
[Wh consumed by vehicle] ÷ [pack capacity] = DoD_equivalent
             ↓  (cumulative sum per vehID across trips)
       EFC_vehicle
             ↓  (applied to the laboratory SoH(EFC) curve)
       Predicted SoH
```

Three degradation scenarios are produced (pooled, pessimistic=3C, optimistic=C/2), based on the linear fit `SoH = 1 + k·EFC` (Phase 3), later refined to exponential in Phase 4.

> **Scale note**: 21 simulated trips accumulate EFC ≈ 4–7 per vehicle, against a laboratory range of 0–221. Predicted SoH values close to 1.0 are not a bug — they are mathematically expected: 21 trips represent approximately 1–2% of estimated battery lifetime.

### Phase 4 — Modelling

#### Model B — SoH(EFC) Curve

Two functional forms were compared across 94 valid diagnostic points (8 cells × ~10–15 diags each):

| Model | Formula | k (pooled) | R² (pooled) |
|---|---|---|---|
| **B1 — Linear** | `SoH = 1 + k·EFC` | -0.000462 | 0.848 |
| **B2 — Exponential** ✓ | `SoH = exp(k·EFC)` | -0.000481 | 0.860 |

B2 was adopted as the official model: the R² gain of +1.15pp is modest, but the exponential form guarantees `SoH ∈ [0, 1]` for any EFC value, whereas the linear model can predict negative SoH at high EFC — a physically invalid outcome.

**GPR (Gaussian Process Regression)** was evaluated and discarded: the predictive standard deviation varied by only ~10% across the full EFC range (0.0075–0.0083), without meaningful distinction between real and extrapolated points. The relevant uncertainty (`cycle_count_estimated`) is epistemic in origin — GPR cannot capture it because it only observes position on the EFC axis, not the data provenance.

Estimated EFC at EOL (SoH = 0.80): `EFC_EOL = ln(0.80) / k ≈ 464`.

#### Model A — DoD_equivalent Prediction (XGBoost)

**Objective**: predict EFC consumption for a new route without running the full SUMO vehicle simulation.

**Decision to include `vehID` as a dummy**: the version without the dummy showed systematic residual bias by vehicle (eUp/i3 underestimated, ID4/ID3 overestimated), reflecting the fact that the same route consumes very different battery fractions depending on pack capacity. One-hot encoding of `vehID` (drop='first') enables XGBoost to learn vehicle-specific `distance → DoD` conversion coefficients without directly introducing the numerical capacity.

**Evaluation protocol**: LOO-CV (Leave-One-Out) with nested 5-fold GridSearchCV — for each outer LOO fold, hyperparameter search is performed on the n-1 training examples, preventing information leakage from the test point into model selection.

| Variant | R² LOO-CV | MAE | RMSE |
|---|---|---|---|
| XGBoost without vehID | 0.858 | 0.043 | 0.061 |
| **XGBoost with vehID dummy** ✓ | **0.955** | 0.024 | 0.034 |

The +0.097 R² gain from the dummy **is not data leakage**: between-vehicle variance accounts for only 13.7% of total variance — the dummy is not simply memorizing per-vehicle means, but enabling the model to learn interaction effects between vehicle type and route characteristics.

**Correlation analysis (A1)** found that `completedDistance_km` (r=0.86) dominates DoD_equivalent variance, followed by `pct_uphill` (r=0.48). Simulation parameters (`traffic_factor`, `occupancy`, `auxiliaries`, `wind`) showed no detectable effect — possibly due to insufficient range variation across the 21 routes, or conditional effects not captured by simple bivariate correlation.

---

## Validation Results (Phase 5)

| Check | Result |
|---|---|
| **5.1** SoH monotonically decreasing with EFC per vehID | ✅ PASS — all 5 vehicles |
| **5.2** Cumulative EFC within laboratory range (0–221) | ✅ PASS — max observed EFC: 7.25 (EV10) |
| **5.3** Vehicles with smaller packs accumulate EFC faster and reach EOL first | ✅ PASS — order consistent with nominal capacity |

**Expected degradation order** (fastest to slowest EOL):

```
EV10 (VW eUp,  ~32 kWh)  >  EV1 (BMW i3,  ~39 kWh)  >
EV4  (VW ID3,  ~58 kWh)  >  EV7 (VW ID4,  ~77 kWh)  >
EV13 (SUV,     ~80 kWh)
```

---

## Limitations

1. **Fixed DoD in laboratory protocol (60%, UDDS)**: the `SoH(EFC)` curve was calibrated under this specific protocol. Routes with DoD substantially different from 60% introduce unquantified systematic error — this is a simplifying assumption, not a universal truth.

2. **No charging events in d-EVD**: the driving dataset contains no recharging data. The effect of charge C-rate on degradation — documented in the laboratory dataset (3C degrades faster than C/4) — could not be incorporated into the real vehicle usage profile.

3. **36 diagnostics with `cycle_count_estimated=True`**: cells G1, V4, W10, W5, W8, and W9 have diagnostics beyond the published Table 3 scope, extrapolated using the last known inter-diagnostic interval. Cell G1 has a particularly high extrapolation ratio (6 of 11 diagnostics are estimated).

4. **No field ground truth**: predicted degradation was never validated against real-world battery capacity measurements from vehicles in service. The pipeline is methodologically sound, but lacks longitudinal field validation.

5. **XGBoost does not generalize to unseen vehicle models**: the `vehID` dummy acts as a proxy for pack capacity and aerodynamic profile. A 6th unseen vehicle model would fall back to the no-dummy performance (R²≈0.858), and more importantly, would lack specific calibration in the SoH(EFC) curve.

6. **Temporal scale of 21 routes**: the accumulated EFC of 4–7 represents approximately 1–2% of the estimated total lifetime (EFC_EOL≈464). Long-term projections are extrapolations of the exponential curve well beyond the per-vehicle observed range.

---

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost shap
```

| Library | Usage |
|---|---|
| `pandas` | data manipulation across all phases |
| `numpy` | numerical integration (Coulomb counting), linear algebra |
| `scipy` | `curve_fit` for exponential regression |
| `scikit-learn` | LOO-CV, GridSearchCV, OneHotEncoder, metrics |
| `xgboost` | DoD_equivalent predictive model (Phase 4 A2) |
| `shap` | XGBoost interpretability (optional) |
| `matplotlib` / `seaborn` | all visualizations |

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure processed data is in the expected paths:
#    data/processed/battery/capacity_test_10cells.csv
#    data/processed/battery/table3_diagnostic_cycles.csv
#    data/processed/ev/ev_routes/*.csv

# 3. Run phases in order
python src/fase1_battery_processing.py
python src/fase2_ev_processing.py
python src/fase3_efc_bridge.py
python src/fase4_a2_xgboost.py            # takes ~2-3 min (nested LOO-CV)
python src/fase5_6_validation_output.py
```

---

## Final Outputs

| File | Description |
|---|---|
| `battery_soh_curve_corrigido.csv` | SoH×EFC curve per cell, with extrapolation flag |
| `ev_trip_features.csv` | Driving features per (vehID, route) |
| `predicted_degradation.csv` | Predicted SoH per vehicle across 21 trips |
| `a2_xgb_predictions.csv` | XGBoost predictions vs. real DoD_equivalent |
| `fig1_soh_efc_projection.png` | SoH×EFC curve with EOL projection per vehicle |
| `fig2_dod_distribution.png` | DoD_equivalent distribution per vehicle model |
| `fig3_xgb_predicted_vs_real.png` | XGBoost model evaluation (Phase 4 A2) |
| `fig4_degradation_by_vehicle.png` | Cumulative EFC and SoH across 21 trips |
| `validation_report.txt` | Results of the 3 validation checks (Phase 5) |

---

## References

- Pozzato, G., Allam, A., & Onori, S. (2022). *Lithium-ion battery aging dataset based on electric vehicle real-driving profiles*. Data in Brief, 41, 107995. https://doi.org/10.1016/j.dib.2022.107995
- Vicomtech. (2022). *d-EVD: dual Electric Vehicle Dataset*. https://github.com/Vicomtech/d-EVD_dual-Electric-Vehicle-Dataset
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
- Lundberg, S., & Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems 30 (NeurIPS 2017).
