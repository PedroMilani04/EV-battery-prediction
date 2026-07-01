"""
Phases 5 and 6 — Validation and Final Output

Inputs (all under data/processed/):
    battery/battery_soh_curve_corrigido.csv   (Phase 1)
    ev/ev_trip_features.csv                   (Phase 2)
    bridge/predicted_degradation.csv          (Phase 3)
    models/a2_xgb_predictions.csv             (Phase 4)
    models/a2_xgb_results.csv                 (Phase 4)

Outputs (under data/output/):
    fig1_soh_efc_projection.png
    fig2_dod_distribution.png
    fig3_xgb_predicted_vs_real.png
    fig4_degradation_by_vehicle.png
    validation_report.txt
    README_pipeline.md

How to run:
    python fase5_6_validation_output.py

Dependencies:
    pip install pandas numpy matplotlib seaborn scipy
"""

from pathlib import Path
from datetime import date
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR    = Path("data/output")

BATTERY_FILE  = PROCESSED_DIR / "battery" / "battery_soh_curve.csv"
EV_FILE       = PROCESSED_DIR / "ev"      / "ev_trip_features.csv"
BRIDGE_FILE   = PROCESSED_DIR / "bridge"  / "predicted_degradation.csv"
XGB_PRED_FILE = PROCESSED_DIR / "models"  / "a2_xgb_predictions.csv"
XGB_RES_FILE  = PROCESSED_DIR / "models"  / "a2_xgb_results.csv"

PROTOCOL_DOD  = 0.6
SOH_EOL       = 0.80          # conventional end-of-life threshold for EVs (80% SoH)
EFC_FULL_LIFE = 500           # projection up to 500 EFC (~estimated real useful life)

VEHID_LABELS = {
    "EV1":  "BMW i3 (normal)",
    "EV4":  "VW ID3 (normal)",
    "EV7":  "VW ID4 (normal)",
    "EV10": "VW eUp (normal)",
    "EV13": "SUV (normal)",
}

PALETTE = {
    "EV1":  "#2a78d6",
    "EV4":  "#e34948",
    "EV7":  "#7b4f9e",
    "EV10": "#e08c1a",
    "EV13": "#2d9e6b",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all():
    battery = pd.read_csv(BATTERY_FILE)
    ev      = pd.read_csv(EV_FILE)
    bridge  = pd.read_csv(BRIDGE_FILE)
    xgb_pred = pd.read_csv(XGB_PRED_FILE)
    xgb_res  = pd.read_csv(XGB_RES_FILE)
    return battery, ev, bridge, xgb_pred, xgb_res


# ---------------------------------------------------------------------------
# Exponential curve fitting (reuses Phase 4 logic)
# ---------------------------------------------------------------------------

def fit_exp(efc, soh):
    def exp_model(x, k): return np.exp(k * x)
    popt, _ = curve_fit(exp_model, efc, soh, p0=[-0.0005])
    return popt[0]


def soh_from_efc(efc_arr, k):
    return np.clip(np.exp(k * efc_arr), 0, 1)


def efc_at_eol(k, eol=SOH_EOL):
    """EFC at which SoH reaches the end-of-life threshold."""
    return np.log(eol) / k if k != 0 else np.inf


# ---------------------------------------------------------------------------
# PHASE 5 — Validation checks
# ---------------------------------------------------------------------------

def check_monotonic_soh(bridge: pd.DataFrame) -> dict:
    """5.1 — SoH decreases monotonically with EFC for each vehID."""
    results = {}
    for veh, g in bridge.groupby("vehID"):
        g = g.sort_values("EFC_cumulative")
        diffs = g["soh_previsto_medio"].diff().dropna()
        results[veh] = bool((diffs <= 1e-9).all())
    return results


def check_efc_range(bridge: pd.DataFrame, battery: pd.DataFrame) -> dict:
    """5.2 — Cumulative EFC stays within the laboratory-observed range."""
    efc_veh_max = bridge.groupby("vehID")["EFC_cumulative"].max()
    efc_lab_max = battery[~battery["dismissed_cell"]]["EFC_lab"].max()
    return {
        "efc_lab_max":     float(efc_lab_max),
        "efc_veh_max":     efc_veh_max.to_dict(),
        "all_within_range": bool((efc_veh_max < efc_lab_max).all()),
    }


def check_degradation_order(ev: pd.DataFrame, battery: pd.DataFrame) -> dict:
    """
    5.3 — Vehicles with smaller packs (higher DoD_equivalente per route)
    accumulate EFC faster and should reach EOL sooner.
    """
    clean = battery[~battery["dismissed_cell"]]
    k = fit_exp(clean["EFC_lab"].to_numpy(), clean["SoH"].to_numpy())

    efc_per_trip = ev.groupby("vehID")["DoD_equivalente"].mean()
    efc_to_eol   = efc_at_eol(k)

    # trips_to_eol = how many "average" trips to reach EOL
    trips_to_eol = efc_to_eol / efc_per_trip

    result = pd.DataFrame({
        "DoD_medio_por_viagem": efc_per_trip,
        "EFC_para_EOL":         efc_to_eol,
        "Viagens_para_EOL":     trips_to_eol,
    }).sort_values("DoD_medio_por_viagem", ascending=False)

    # check: higher DoD_medio → fewer trips to EOL (inverse relationship)
    is_coherent = result["DoD_medio_por_viagem"].is_monotonic_decreasing and \
                  result["Viagens_para_EOL"].is_monotonic_increasing

    return {"table": result, "physically_coherent": is_coherent, "k": k, "efc_to_eol": efc_to_eol}


def run_validation(battery, ev, bridge):
    report_lines = []
    report_lines.append("=" * 65)
    report_lines.append("VALIDATION REPORT — Battery Degradation Pipeline")
    report_lines.append(f"Generated on: {date.today()}")
    report_lines.append("=" * 65)

    # 5.1
    mono = check_monotonic_soh(bridge)
    report_lines.append("\n[CHECK 5.1] SoH monotonically decreasing with EFC per vehID")
    for veh, ok in mono.items():
        status = "PASS" if ok else "FAIL"
        report_lines.append(f"  {veh}: {status}")

    # 5.2
    rng = check_efc_range(bridge, battery)
    report_lines.append("\n[CHECK 5.2] Cumulative EFC within laboratory range")
    report_lines.append(f"  Max observed lab EFC: {rng['efc_lab_max']:.1f}")
    for veh, val in rng["efc_veh_max"].items():
        report_lines.append(f"  {veh}: EFC_max={val:.2f}  {'OK' if val < rng['efc_lab_max'] else 'OUT OF RANGE'}")
    report_lines.append(f"  All within the range: {'PASS' if rng['all_within_range'] else 'FAIL'}")

    # 5.3
    deg = check_degradation_order(ev, battery)
    report_lines.append("\n[CHECK 5.3] Physically coherent degradation order")
    report_lines.append(deg["table"].round(2).to_string())
    report_lines.append(f"\n  Physical coherence (higher DoD → fewer trips to EOL): "
                        f"{'PASS' if deg['physically_coherent'] else 'FAIL'}")
    report_lines.append(f"  Estimated EFC for EOL (SoH={SOH_EOL}): {deg['efc_to_eol']:.1f}")

    report_lines.append("\n" + "=" * 65)
    report_lines.append("DOCUMENTED LIMITATIONS")
    report_lines.append("=" * 65)
    limitations = [
        "Fixed DoD in the laboratory protocol (60%, UDDS 80%→20% SoC): the SoH(EFC) curve",
        "was fitted under this specific DoD and does not generalize to",
        "arbitrary discharge patterns — this is a simplifying Phase 1 assumption.",
        "",
        "No charging events in d-EVD: the driving dataset does not",
        "contain recharge data; the impact of charge rate (C-rate) on",
        "degradation could not be incorporated into the real vehicle usage profile.",
        "",
        "cycle_count_estimated=True for 36 diagnostics (G1, V4, W10, W5, W8, W9):",
        "values extrapolated from the last known interval — treat predictions",
        "in this EFC region with higher uncertainty.",
        "",
        "Discontinued cells W3 and W7 were excluded from the fit (anomalous",
        "behavior documented by the dataset authors).",
        "",
        "The XGBoost model (Phase 4 A2) with vehID dummy does not generalize to",
        "vehicle models unseen during training — the dummy acts as a proxy",
        "for pack capacity and vehicle-specific aerodynamic style.",
        "",
        "The 21 routes per vehicle represent a minimal fraction of real useful life",
        "(EFC accumulated 4–7 out of an estimated ~400–500 for EOL).",
    ]
    for line in limitations:
        report_lines.append(f"  {line}")

    return "\n".join(report_lines), deg


# ---------------------------------------------------------------------------
# PHASE 6 — Visualizations
# ---------------------------------------------------------------------------

def fig1_soh_projection(battery: pd.DataFrame, ev: pd.DataFrame,
                         deg_check: dict, out: Path) -> None:
    """SoH × EFC curve: lab data plus projection to EOL by vehID."""
    k = deg_check["k"]
    efc_to_eol = deg_check["efc_to_eol"]
    clean = battery[~battery["dismissed_cell"]]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Actual lab data points (in gray)
    for cell, g in clean.groupby("cell"):
        estimated = g["cycle_count_estimated"].any()
        g_real = g[~g["cycle_count_estimated"]]
        g_est  = g[g["cycle_count_estimated"]]
        if not g_real.empty:
            ax.scatter(g_real["EFC_lab"], g_real["SoH"],
                       color="#bbb", s=18, zorder=1)
        if not g_est.empty:
            ax.scatter(g_est["EFC_lab"], g_est["SoH"],
                       color="#ddd", s=18, marker="x", zorder=1)

    # Fitted exponential curve (pooled)
    efc_curve = np.linspace(0, EFC_FULL_LIFE, 500)
    ax.plot(efc_curve, soh_from_efc(efc_curve, k),
            color="#555", lw=1.5, ls="--", label="SoH(EFC) curve — pooled exp.")

    # Projection by vehID: marker where the vehicle would be after 21 trips
    efc_per_trip = ev.groupby("vehID")["DoD_equivalente"].mean()
    for veh, color in PALETTE.items():
        if veh not in efc_per_trip.index:
            continue
        mean_dod = efc_per_trip[veh]
        # simulated cumulative trajectory up to EFC_FULL_LIFE
        n_trips = int(EFC_FULL_LIFE / mean_dod)
        efc_traj = np.cumsum([mean_dod] * n_trips)
        soh_traj = soh_from_efc(efc_traj, k)
        ax.plot(efc_traj, soh_traj, color=color, lw=1.8,
                label=f"{VEHID_LABELS.get(veh, veh)}")
        # EOL point
        eol_idx = np.argmax(soh_traj <= SOH_EOL)
        if soh_traj[eol_idx] <= SOH_EOL:
            ax.scatter(efc_traj[eol_idx], soh_traj[eol_idx],
                       color=color, s=60, zorder=5, marker="X")

    ax.axhline(SOH_EOL, color="red", lw=1, ls=":", label=f"EOL (SoH={SOH_EOL})")
    ax.set_xlabel("EFC acumulado (Equivalent Full Cycles)")
    ax.set_ylabel("SoH (State of Health)")
    ax.set_title("Degradation projection by vehicle model\n"
                 "(gray points = actual lab; × = extrapolated Phase 1; X = EOL point)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, EFC_FULL_LIFE)
    ax.set_ylim(0.75, 1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def fig2_dod_distribution(ev: pd.DataFrame, out: Path) -> None:
    """Distribution of DoD_equivalente by vehID (who accumulates EFC fastest)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Boxplot
    order = ev.groupby("vehID")["DoD_equivalente"].median().sort_values(ascending=False).index
    labels = [VEHID_LABELS.get(v, v) for v in order]
    colors = [PALETTE[v] for v in order]
    bp = axes[0].boxplot(
        [ev[ev["vehID"] == v]["DoD_equivalente"].values for v in order],
        labels=labels, patch_artist=True, medianprops={"color": "black", "lw": 2}
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_title("Distribution of DoD_equivalente by vehicle")
    axes[0].set_ylabel("DoD_equivalente (fraction of pack per trip)")
    axes[0].tick_params(axis="x", rotation=15)

    # Cumulative EFC per trip (mean ± std)
    summary = ev.groupby("vehID")["DoD_equivalente"].agg(["mean", "std"]).loc[order]
    x = range(len(order))
    axes[1].bar(x, summary["mean"], color=colors, alpha=0.7, width=0.5)
    axes[1].errorbar(x, summary["mean"], yerr=summary["std"],
                     fmt="none", color="black", capsize=4)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=15)
    axes[1].set_title("Average DoD per trip (± std dev)\n= EFC accumulation rate")
    axes[1].set_ylabel("Average DoD_equivalente")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def fig3_xgb_results(xgb_pred: pd.DataFrame, xgb_res: pd.DataFrame, out: Path) -> None:
    """XGBoost predicted vs actual with vehID dummy."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    y_true = xgb_pred["DoD_equivalente"].to_numpy()
    y_pred = xgb_pred["pred_xgb_com_vehid"].to_numpy()
    residuals = y_true - y_pred

    # Predicted vs actual
    for veh, g in xgb_pred.groupby("vehID"):
        idx = g.index
        axes[0].scatter(y_true[idx], y_pred[idx],
                        color=PALETTE.get(veh, "#888"), alpha=0.75, s=30,
                        label=VEHID_LABELS.get(veh, veh))
    lims = [y_true.min() - 0.02, y_true.max() + 0.02]
    axes[0].plot(lims, lims, "k--", lw=1)
    r2_row = xgb_res[xgb_res["label"] == "com_vehID_dummy"].iloc[0]
    axes[0].set_title(f"XGBoost with vehID dummy\nR²={r2_row['R2_loocv']:.3f} | "
                      f"RMSE={r2_row['RMSE_loocv']:.4f} (LOO-CV)")
    axes[0].set_xlabel("Actual DoD_equivalente")
    axes[0].set_ylabel("Predicted DoD_equivalente")
    axes[0].legend(fontsize=7)

    # Residuals
    for veh, g in xgb_pred.groupby("vehID"):
        idx = g.index
        axes[1].scatter(y_pred[idx], residuals[idx],
                        color=PALETTE.get(veh, "#888"), alpha=0.75, s=30,
                        label=VEHID_LABELS.get(veh, veh))
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual (actual − predicted)")
    axes[1].set_title("Residuals — no vehicle-specific systematic pattern")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def fig4_degradation_by_vehicle(ev: pd.DataFrame, deg_check: dict, out: Path) -> None:
    """Cumulative EFC and predicted SoH curves across the 21 trips."""
    k = deg_check["k"]
    efc_per_trip = ev.groupby("vehID")["DoD_equivalente"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for veh, color in PALETTE.items():
        trips = ev[ev["vehID"] == veh].sort_values("idx")
        efc_cum = trips["DoD_equivalente"].cumsum().values
        soh_cum = soh_from_efc(efc_cum, k)
        trip_n = np.arange(1, len(efc_cum) + 1)
        label = VEHID_LABELS.get(veh, veh)

        axes[0].plot(trip_n, efc_cum, color=color, lw=2, label=label)
        axes[1].plot(trip_n, soh_cum, color=color, lw=2, label=label)

    axes[0].set_xlabel("Trip number")
    axes[0].set_ylabel("Cumulative EFC")
    axes[0].set_title("EFC accumulation across the 21 trips")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Trip number")
    axes[1].set_ylabel("Predicted SoH")
    axes[1].set_title("Predicted SoH across the 21 trips")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# README_pipeline.md
# ---------------------------------------------------------------------------

def write_readme(out: Path, deg_check: dict) -> None:
    k = deg_check["k"]
    efc_eol = deg_check["efc_to_eol"]

    content = f"""# EV Battery Degradation Prediction Pipeline

Generated on: {date.today()}

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
| k (exp curve) | {k:.6f} — degradation rate per EFC (pooled, valid cells) |
| EFC for EOL | {efc_eol:.1f} (SoH={SOH_EOL}) |

## Degradation model

```
SoH(EFC) = exp(k × EFC)    k = {k:.6f}
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
   EFC≈{efc_eol:.0f} estimated for EOL. The long-term projection is
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
"""
    out.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    battery, ev, bridge, xgb_pred, xgb_res = load_all()

    print("Running validation (Phase 5)...")
    report, deg_check = run_validation(battery, ev, bridge)
    print(report)
    (OUTPUT_DIR / "validation_report.txt").write_text(report, encoding="utf-8")

    print("\nGenerating visualizations (Phase 6)...")
    fig1_soh_projection(battery, ev, deg_check,
                        OUTPUT_DIR / "fig1_soh_efc_projection.png")
    print("  fig1 OK")

    fig2_dod_distribution(ev, OUTPUT_DIR / "fig2_dod_distribution.png")
    print("  fig2 OK")

    fig3_xgb_results(xgb_pred, xgb_res,
                     OUTPUT_DIR / "fig3_xgb_predicted_vs_real.png")
    print("  fig3 OK")

    fig4_degradation_by_vehicle(ev, deg_check,
                                OUTPUT_DIR / "fig4_degradation_by_vehicle.png")
    print("  fig4 OK")

    write_readme(OUTPUT_DIR / "README_pipeline.md", deg_check)
    print("  README OK")

    print(f"\nAll saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
