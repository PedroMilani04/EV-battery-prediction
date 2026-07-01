"""
Phase 3 — Bridge (EFC)

Inputs (from data/processed/):
    data/processed/battery/battery_soh_curve_corrigido.csv  (Phase 1 output)
    data/processed/ev/ev_trip_features.csv                  (Phase 2 output)

Output:
    data/processed/bridge/predicted_degradation.csv

Documented assumptions and decisions (important to read before interpreting the output):

1. TRIP ORDER (item 3.1 in the plan): d-EVD has no timestamp or real
   sequence of use — each vehID runs the same 21 routes as independent
   simulations, not as a chronological record of a real vehicle. `idx` is
   only the route identifier (1 to 21), not order of occurrence. To build
   an illustrative cumulative trajectory, we sort by `idx` arbitrarily and
   reproducibly — this affects the shape of the intermediate curve, but
   not the final EFC_total (which is a sum and order invariant).

2. SoH(EFC) REGRESSION (item 3.3): we fit a line forced through the origin
   (SoH=1 at EFC=0), separately by `charge_rate` and also a pooled model
   using all non-dismissed cells. We use this to generate three scenarios:
   pessimistic (highest observed fade rate, 3C cells), optimistic
   (lowest rate, C/4 cells) and medium (pooled). Since d-EVD does not
   report vehicle charging behavior, there is no way to know which
   scenario is the "correct" one — so we report the range.

3. EFC SCALE: the sum of DoD_equivalente across 21 trips is typically
   small (order of a few EFC units) compared to the laboratory range
   observed (up to ~245 EFC). This is expected: the 21 routes represent a
   minimal fraction of the vehicle's lifetime, not the entire life. Predicted
   SoH will remain near 1.0 — not a bug.

4. Discontinued cells (W3, W7) are excluded from curve fitting
   (anomalous behavior, not representative of "normal" fade).
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
BATTERY_SOH_FILE = PROCESSED_DIR / "battery" / "battery_soh_curve.csv"
EV_TRIP_FEATURES_FILE = PROCESSED_DIR / "EV" / "ev_trip_features.csv"
OUTPUT_FILE = PROCESSED_DIR / "bridge" / "predicted_degradation.csv"


# ---------------------------------------------------------------------------
# 3.3 — Ajuste de SoH(EFC) a partir do battery dataset
# ---------------------------------------------------------------------------

def fit_soh_efc_line(efc: np.ndarray, soh: np.ndarray) -> dict:
    """
    Fit SoH = 1 + k*EFC (line forced through the origin EFC=0 -> SoH=1).
    k tends to be negative (degradation). Returns k, R² and the observed EFC
    range (to detect extrapolation later).
    """
    y = soh - 1.0
    x = efc
    k = float(np.sum(x * y) / np.sum(x * x))
    y_pred = k * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"k": k, "r2": r2, "efc_min": float(x.min()), "efc_max": float(x.max()), "n_points": len(x)}


def fit_all_curves(battery_df: pd.DataFrame) -> dict:
    clean = battery_df[~battery_df["dismissed_cell"]].copy()

    curves = {}
    curves["pooled"] = fit_soh_efc_line(clean["EFC_lab"].to_numpy(), clean["SoH"].to_numpy())

    for rate, group in clean.groupby("charge_rate"):
        if len(group) >= 3:  # avoid unstable fit with too few points
            curves[rate] = fit_soh_efc_line(group["EFC_lab"].to_numpy(), group["SoH"].to_numpy())

    # pessimistic scenario = largest |k| (degrades fastest); optimistic = smallest |k|
    rate_curves = {k: v for k, v in curves.items() if k != "pooled"}
    if rate_curves:
        pessimista_rate = min(rate_curves, key=lambda r: rate_curves[r]["k"])  # k mais negativo
        otimista_rate = max(rate_curves, key=lambda r: rate_curves[r]["k"])    # k menos negativo
        curves["_pessimista_rate"] = pessimista_rate
        curves["_otimista_rate"] = otimista_rate

    return curves


def print_curve_diagnostics(curves: dict) -> None:
    print("=== Fitted SoH(EFC) curves (line forced through the origin) ===")
    for name, c in curves.items():
        if name.startswith("_"):
            continue
        print(f"  {name:>8s}: k={c['k']:.6f}/EFC | R²={c['r2']:.3f} | "
              f"observed EFC=[{c['efc_min']:.1f}, {c['efc_max']:.1f}] | n={c['n_points']}")
    if "_pessimista_rate" in curves:
        print(f"  -> pessimistic scenario = charge rate {curves['_pessimista_rate']}")
        print(f"  -> optimistic scenario  = charge rate {curves['_otimista_rate']}")


# ---------------------------------------------------------------------------
# 3.1 + 3.2 — Sort trips (arbitrary) and accumulate EFC per vehicle
# ---------------------------------------------------------------------------

def build_vehicle_trajectories(ev_df: pd.DataFrame) -> pd.DataFrame:
    df = ev_df.sort_values(["vehID", "idx"]).copy()  # arbitrary order, see docstring
    df["trip_order"] = df.groupby("vehID").cumcount() + 1
    # EFC_cumulative = direct sum of DoD_equivalente across trips.
    # The 0.6 lab protocol factor is specific to the UDDS aging cycle
    # (80%->20% SoC) and does not apply here — each DoD_equivalente is
    # already the actual fraction of capacity used in the trip.
    df["EFC_cumulative"] = df.groupby("vehID")["DoD_equivalente"].cumsum()
    return df


# ---------------------------------------------------------------------------
# 3.4 — Aplicar SoH(EFC) sobre EFC_veiculo
# ---------------------------------------------------------------------------

def apply_soh_curve(efc: np.ndarray, k: float) -> np.ndarray:
    soh = 1.0 + k * efc
    return np.clip(soh, 0.0, 1.0)


def attach_predictions(traj_df: pd.DataFrame, curves: dict) -> pd.DataFrame:
    df = traj_df.copy()
    efc = df["EFC_cumulative"].to_numpy()

    df["soh_previsto_medio"] = apply_soh_curve(efc, curves["pooled"]["k"])

    if "_pessimista_rate" in curves:
        k_pess = curves[curves["_pessimista_rate"]]["k"]
        k_otim = curves[curves["_otimista_rate"]]["k"]
        df["soh_previsto_pessimista"] = apply_soh_curve(efc, k_pess)
        df["soh_previsto_otimista"] = apply_soh_curve(efc, k_otim)

    efc_max_lab = curves["pooled"]["efc_max"]
    df["extrapolado"] = df["EFC_cumulative"] > efc_max_lab
    return df


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    battery_df = pd.read_csv(BATTERY_SOH_FILE)
    ev_df = pd.read_csv(EV_TRIP_FEATURES_FILE)

    curves = fit_all_curves(battery_df)
    print_curve_diagnostics(curves)
    print()

    traj_df = build_vehicle_trajectories(ev_df)
    result_df = attach_predictions(traj_df, curves)

    n_extrap = result_df["extrapolado"].sum()
    if n_extrap:
        print(f"WARNING: {n_extrap} point(s) with cumulative EFC above the "
              "laboratory observed range (extrapolation).")

    print()
    print("=== Total EFC and final predicted SoH, by vehicle ===")
    summary = result_df.groupby("vehID").agg(
        EFC_total=("EFC_cumulative", "max"),
        SoH_final_medio=("soh_previsto_medio", "min"),
        SoH_final_pessimista=("soh_previsto_pessimista", "min") if "soh_previsto_pessimista" in result_df else ("soh_previsto_medio", "min"),
        SoH_final_otimista=("soh_previsto_otimista", "min") if "soh_previsto_otimista" in result_df else ("soh_previsto_medio", "min"),
    )
    print(summary)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")
    return result_df


if __name__ == "__main__":
    main()
