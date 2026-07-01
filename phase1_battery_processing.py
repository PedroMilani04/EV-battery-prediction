"""
Phase 1 — Battery dataset processing (capacity_test_10cells)

Inputs (all coming from data/processed/, per project convention:
data/raw/ holds untouched data; data/processed/ holds the same data
in cleaned tabular form, ready for analysis):

    data/processed/battery/capacity_test_10cells.csv
        columns: diag_number, cell, time_s, voltage_V, current_A

    data/processed/battery/table3_diagnostic_cycles.csv
        columns: cell, charge_rate, diag_number, cycle_count, dismissed
        (manually transcribed from Table 3 of Pozzato et al., 2022)

Output:
    data/processed/battery/battery_soh_curve.csv
        columns: cell, diag_number, discharged_capacity_Ah, SoH,
                 charge_rate, cycle_count, dismissed_cell, EFC_lab

Documented assumptions and decisions:
- Current sign convention: the paper text states negative current = discharge,
  but the provided data examples (*_dis_* with positive current, *_chg_* with
  negative current) indicate the opposite. To avoid this ambiguity, we
  integrate |current_A| — valid here because capacity_test_10cells is a pure,
  continuous C/20 discharge (~0.24A) with no mixed charge segments in the
  same file.
- SoH is calculated relative to the cell's OWN pristine measurement at
  diag_number=1, not relative to the theoretical Qnom of 4.85Ah — this
  absorbs manufacturing variation between cells.
- EFC_lab = 0.6 * cycle_count, because the aging protocol discharges exactly
  60% of capacity per full cycle (Step 5: 20% from 100%->80% SOC + Step 6:
  60% from 80%->20% SOC).
- Cells W3 and W7 were discontinued by the authors (anomalous impedance /
  inconsistent EIS measurements). They are marked in `dismissed_cell`
  rather than removed by default — see DISMISSED_POLICY.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
BATTERY_FILE = PROCESSED_DIR / "battery" / "capacity_test_10cells.csv"
TABLE3_FILE = PROCESSED_DIR / "battery" / "table3_diagnostic_cycles.csv"
OUTPUT_FILE = PROCESSED_DIR / "battery" / "battery_soh_curve.csv"

PROTOCOL_DOD = 0.6          # fixed protocol DoD for aging (Step 5+6)
QNOM_REFERENCE_AH = 4.85    # factory nominal capacity, only for sanity checking
DISMISSED_CELLS = ("W3", "W7")
DISMISSED_POLICY = "flag"   # "flag" keeps and marks; "drop" removes from output


# ---------------------------------------------------------------------------
# load and validation
# ---------------------------------------------------------------------------

def load_capacity_test(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, dtype={"cell": str})
    required = {"diag_number", "cell", "time_s", "voltage_V", "current_A"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    # Coerce numeric data and drop rows with invalid measurement samples.
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["voltage_V"] = pd.to_numeric(df["voltage_V"], errors="coerce")
    df["current_A"] = pd.to_numeric(df["current_A"], errors="coerce")

    invalid = df["time_s"].isna() | df["current_A"].isna()
    if invalid.any():
        count = invalid.sum()
        print(f"WARNING: {count} invalid row(s) in {path.name} removed due to missing time_s/current_A.")
        df = df.loc[~invalid].reset_index(drop=True)
    return df


def load_table3(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"cell", "charge_rate", "diag_number", "cycle_count", "dismissed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")
    return df


# ---------------------------------------------------------------------------
# coulomb counting and SoH calculation
# ---------------------------------------------------------------------------

def integrate_discharged_capacity(group: pd.DataFrame) -> float:
    """
    Integrate |current_A| over time_s (trapezoidal rule) -> Coulombs -> Ah.
    See module docstring note on current sign convention.
    """
    g = group.sort_values("time_s")
    t = g["time_s"].to_numpy()
    i_abs = np.abs(g["current_A"].to_numpy())
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz  # numpy >=2.0 renamed trapz
    charge_coulombs = trapz_fn(i_abs, t)
    return charge_coulombs / 3600.0  # Ah


def build_capacity_table(capacity_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cell, diag), group in capacity_df.groupby(["cell", "diag_number"]):
        rows.append({
            "cell": cell,
            "diag_number": diag,
            "discharged_capacity_Ah": integrate_discharged_capacity(group),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SoH calculation and sanity check
# ---------------------------------------------------------------------------

def attach_soh(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["cell", "diag_number"]).copy()
    baseline = df.groupby("cell")["discharged_capacity_Ah"].transform("first")
    df["SoH"] = df["discharged_capacity_Ah"] / baseline

    # sanity check: baseline should be close to factory Qnom
    first_rows = df[df.groupby("cell")["diag_number"].transform("min") == df["diag_number"]]
    off_spec = first_rows[
        (first_rows["discharged_capacity_Ah"] < QNOM_REFERENCE_AH * 0.85)
        | (first_rows["discharged_capacity_Ah"] > QNOM_REFERENCE_AH * 1.15)
    ]
    if not off_spec.empty:
        print("WARNING: initial diag_number capacity is outside ±15% of factory Qnom "
              f"({QNOM_REFERENCE_AH} Ah):")
        print(off_spec[["cell", "diag_number", "discharged_capacity_Ah"]])

    return df


# ---------------------------------------------------------------------------
# merge with Table 3 and extrapolate missing cycle counts
# ---------------------------------------------------------------------------

def merge_table3(soh_df: pd.DataFrame, table3_df: pd.DataFrame) -> pd.DataFrame:
    merged = soh_df.merge(
        table3_df, on=["cell", "diag_number"], how="left", validate="one_to_one"
    )
    missing = merged["cycle_count"].isna().sum()
    if missing:
        print(f"WARNING: {missing} row(s) without cycle_count after merge — "
              "check whether table3_diagnostic_cycles.csv covers all cells/diags.")
    return merged


def extrapolate_missing_cycle_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 3 from the paper is a partial snapshot ("until February 1, 2022"),
    while the real campaign ran for 23 months. Cells like G1, V4, W10, W5,
    W8 and W9 have diag_numbers in the real CSV beyond what the published
    table covers.

    We extrapolate cycle_count using the LAST known interval for each cell
    (not a global constant of 25 cycles) — the actual diagnostic spacing
    varies significantly over the campaign (dropping to 1-8 cycles in the
    middle, likely due to more frequent tests near accelerated degradation
    events, and returning to ~25-29 in the last known segment).

    Extrapolated rows are marked with `cycle_count_estimated=True` — treat
    them with more caution in modeling (e.g. higher uncertainty in SoH(EFC)
    in that region).
    """
    df = df.copy()
    df["cycle_count_estimated"] = False

    for cell, group in df.groupby("cell"):
        group = group.sort_values("diag_number")
        known = group.dropna(subset=["cycle_count"])
        missing_idx = group[group["cycle_count"].isna()].index

        if len(missing_idx) == 0:
            continue
        if len(known) < 2:
            print(f"WARNING: cell {cell} has fewer than 2 known diagnostics "
                  "— cannot extrapolate reliably. Rows remain NaN.")
            continue

        last_two = known.tail(2)
        diag_a, diag_b = last_two["diag_number"].to_numpy()
        cyc_a, cyc_b = last_two["cycle_count"].to_numpy()
        slope = (cyc_b - cyc_a) / (diag_b - diag_a)  # cycles per diag, local

        for idx in missing_idx:
            diag_n = df.loc[idx, "diag_number"]
            estimated = cyc_b + slope * (diag_n - diag_b)
            df.loc[idx, "cycle_count"] = estimated
            df.loc[idx, "cycle_count_estimated"] = True

        print(f"{cell}: extrapolated diag_number(s) {sorted(group.loc[missing_idx, 'diag_number'].tolist())} "
              f"using local slope of {slope:.1f} cycles/diag (based on diags {int(diag_a)}->{int(diag_b)})")

    # charge_rate and dismissed are constant per cell — propagate them to extrapolated rows
    df["charge_rate"] = df.groupby("cell")["charge_rate"].transform(lambda s: s.ffill().bfill())
    df["dismissed"] = df.groupby("cell")["dismissed"].transform(lambda s: s.ffill().bfill())
    return df


def compute_efc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EFC_lab"] = PROTOCOL_DOD * df["cycle_count"]
    return df


def apply_dismissed_policy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dismissed_cell"] = df["cell"].isin(DISMISSED_CELLS) | df["dismissed"].fillna(False)
    if DISMISSED_POLICY == "drop":
        df = df[~df["dismissed_cell"]]
    return df


# ---------------------------------------------------------------------------
# main pipeline 
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    capacity_df = load_capacity_test(BATTERY_FILE)
    table3_df = load_table3(TABLE3_FILE)

    soh_df = build_capacity_table(capacity_df)
    soh_df = attach_soh(soh_df)
    soh_df = merge_table3(soh_df, table3_df)
    soh_df = extrapolate_missing_cycle_counts(soh_df)
    soh_df = compute_efc(soh_df)
    soh_df = apply_dismissed_policy(soh_df)

    soh_df = soh_df.sort_values(["cell", "diag_number"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    soh_df.to_csv(OUTPUT_FILE, index=False)

    print(soh_df[["cell", "diag_number", "charge_rate", "cycle_count", "cycle_count_estimated",
                   "EFC_lab", "discharged_capacity_Ah", "SoH", "dismissed_cell"]])
    print(f"\nSaved to: {OUTPUT_FILE}")
    return soh_df


if __name__ == "__main__":
    main()