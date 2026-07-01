# EV Battery Degradation Prediction Pipeline

Generated on: 2026-07-01

## Overview

Complete pipeline linking electric vehicle driving data (d-EVD)
with laboratory battery aging data (Pozzato et al., 2022)
to estimate battery degradation over real vehicle usage.

## Datasets

| Dataset | Source | Use |
|---|---|---|
| d-EVD (Vicomtech) | SUMO simulation, 21 routes, 5 EV models | Real usage profile |
| INR21700-M50T (Stanford/Onori Lab) | Laboratory aging cycles | SoH(EFC) degradation curve |
| Table 3 (Pozzato et al. 2022) | Manually transcribed | Cycle metadata |

## Pipeline structure

```
Phase 0  Setup and table3_diagnostic_cycles.csv
Phase 1  battery_soh_curve_corrigido.csv   — Coulomb counting + SoH + EFC_lab
Phase 2  ev_trip_features.csv              — route features by (vehID, route)
Phase 3  predicted_degradation.csv         — cumulative EFC + predicted SoH
Phase 4  a2_xgb_predictions.csv            — XGBoost DoD_equivalente
Phase 5  Validation (3 checks)
Phase 6  Visualizations and final output
```

## Key concepts

| Concept | Definition |
|---|---|
| SoH | discharged_capacity_Ah(diag) / discharged_capacity_Ah(diag=1) |
| DoD_equivalente | net_energy_Wh / nominal_capacity_Wh (dimensionless, 0–1) |
| EFC_lab | 0.6 × cycle_count (fixed DoD from UDDS protocol) |
| EFC_vehicle | Σ DoD_equivalente across trips |
| k (exp curve) | -0.000481 — degradation rate per EFC (pooled, valid cells) |
| EFC for EOL | 464.0 (SoH=0.8) |

## Degradation model

```
SoH(EFC) = exp(k × EFC)    k = -0.000481
```

Fitted by forced-origin exponential regression (SoH=1 at EFC=0),
over 94 points from the 8 valid cells (W3 and W7 excluded for anomalous
behavior documented by the dataset authors).

## Usage prediction model (Phase 4 A2)

Nested XGBoost with LOO-CV and internal 5-fold GridSearchCV.
Features: distance, duration, topology, aggressive events, route metadata,
vehID as one-hot dummy.
R² LOO-CV: 0.955 (with vehID dummy) | 0.858 (without vehID dummy)

## Limitations

1. **Fixed DoD in laboratory protocol (60%, UDDS)**: the SoH(EFC) curve
   was fitted under this protocol. Routes with very different DoD may
   introduce systematic error that is not quantified.

2. **No charging data**: d-EVD contains no recharge events. The impact of
   charge rate (C-rate) on degradation could not be incorporated.

3. **36 diagnostics with extrapolated cycle_count** (flag `cycle_count_estimated`):
   cells G1, V4, W10, W5, W8, W9 have diagnostics outside the published
   Table 3 scope. Values were extrapolated from the last known interval.

4. **Cells W3 and W7 excluded**: discontinued by the authors due to
   anomalous impedance (W3) and inconsistent EIS measurements (W7).

5. **vehID dummy does not generalize to new vehicles**: XGBoost learns
   an implicit coefficient per vehicle model. A 6th unseen model would
   likely fall back to performance without dummy (R²≈0.858).

6. **Time scale**: the 21 routes represent EFC≈4–7 compared to
   EFC≈464 estimated for EOL. The long-term projection is
   an extrapolation of the exponential curve beyond the observed range.

## Modeled vehicles (normal style)

| vehID | Model | Approx. capacity |
|---|---|---|
| EV1  | BMW i3   | ~39 kWh |
| EV4  | VW ID3   | ~58 kWh |
| EV7  | VW ID4   | ~77 kWh |
| EV10 | VW eUp   | ~32 kWh |
| EV13 | SUV      | ~80 kWh |

## Output files

| File | Content |
|---|---|
| fig1_soh_efc_projection.png | SoH×EFC curve projected to EOL by vehicle |
| fig2_dod_distribution.png | Distribution of DoD_equivalente by vehicle |
| fig3_xgb_predicted_vs_real.png | XGBoost model evaluation (Phase 4 A2) |
| fig4_degradation_by_vehicle.png | EFC and SoH across the 21 trips |
| validation_report.txt | Results of the 3 validation checks |
| README_pipeline.md | This document |
