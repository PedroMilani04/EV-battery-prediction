"""
Phase 2 — d-EVD processing (driving routes)

Input (from data/processed/, per project convention):
    data/processed/ev_routes/*.csv
        files separated by ';', one per route/simulation configuration.
        Expected name: "{idx}_{origin}_{destination}_{trafficFactor}_
        {occupancy}_{auxiliaries}_{wind}[_output].csv"

Output:
    data/processed/ev/ev_trip_features.csv
        one row per (vehID, route_file), only for normal-style vehIDs.

Documented assumptions:
- STEP_DURATION_S = 1.0s — no explicit confirmation of the SUMO
  simulation step length was found in the public d-EVD documentation;
  the tool default was assumed. If you have the original .sumocfg,
  adjust this constant before relying on `duracao_h`/`C_rate_medio`.
- AGGRESSIVE_ACCEL_THRESHOLD = 2.0 m/s² — arbitrary threshold for counting
  "aggressive events"; adjustable, not sourced from the paper.
- DoD_pct assumes the trip starts at SoC=100% (consistent with observed
  examples so far). This has not been validated line by line yet —
  the script prints a warning if any first step is not at 100%.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
EV_ROUTES_DIR = PROCESSED_DIR / "EV"
OUTPUT_FILE = PROCESSED_DIR / "EV" / "ev_trip_features.csv"

NORMAL_STYLE_VEHIDS = {"EV1", "EV4", "EV7", "EV10", "EV13"}
STEP_DURATION_S = 1.0          # ASSUMPTION — see docstring
AGGRESSIVE_ACCEL_THRESHOLD = 2.0  # m/s²

FILENAME_PATTERN = re.compile(
    r"^(?P<idx>\d+)_(?P<origin>[A-Za-z]+)_(?P<destination>[A-Za-z]+)_"
    r"(?P<traffic_factor>[\d.]+)_(?P<occupancy>\d+)_(?P<auxiliaries>\d+)_"
    r"(?P<wind>-?[\d.]+)(?:_output)?$"
)

REQUIRED_COLUMNS = {
    "vehID", "step", "acceleration(m/s²)", "actualBatteryCapacity(Wh)", "SoC(%)",
    "speed(m/s)", "totalEnergyConsumed(Wh)", "totalEnergyRegenerated(Wh)",
    "slope(º)", "completedDistance(km)", "mWh",
}


# ---------------------------------------------------------------------------
# Filename parsing (simulation metadata)
# ---------------------------------------------------------------------------

def parse_filename(path: Path) -> dict:
    m = FILENAME_PATTERN.match(path.stem)
    if not m:
        return {"parse_ok": False}
    d = m.groupdict()
    d["parse_ok"] = True
    d["traffic_factor"] = float(d["traffic_factor"])
    d["occupancy"] = int(d["occupancy"])
    d["auxiliaries"] = int(d["auxiliaries"])
    d["wind"] = float(d["wind"])
    return d


# ---------------------------------------------------------------------------
# Carga de cada arquivo de rota
# ---------------------------------------------------------------------------

def list_route_files(directory: Path) -> list:
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return files


def load_route_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",")
    df.columns = [c.strip() for c in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")
    return df


# ---------------------------------------------------------------------------
# Features por (vehID, rota)
# ---------------------------------------------------------------------------

def compute_trip_features(df_vehicle: pd.DataFrame, route_name: str) -> dict:
    g = df_vehicle.sort_values("step")
    first, last = g.iloc[0], g.iloc[-1]

    if abs(first["SoC(%)"] - 100.0) > 1e-3:
        print(f"WARNING: {route_name} — first step has SoC={first['SoC(%)']:.2f}% "
              "(expected 100%). DoD_pct may be underestimated.")

    nominal_capacity_Wh = first["actualBatteryCapacity(Wh)"]
    dod_pct = 100.0 - last["SoC(%)"]

    net_energy_Wh = last["totalEnergyConsumed(Wh)"] - last["totalEnergyRegenerated(Wh)"]
    dod_equivalente = (
        net_energy_Wh / nominal_capacity_Wh if nominal_capacity_Wh else np.nan
    )

    n_steps = last["step"] - first["step"]
    duracao_h = (n_steps * STEP_DURATION_S) / 3600.0
    c_rate_medio = (
        net_energy_Wh / nominal_capacity_Wh / duracao_h
        if nominal_capacity_Wh and duracao_h > 0 else np.nan
    )

    completed_distance_km = last["completedDistance(km)"]
    aggressive_events = (g["acceleration(m/s²)"].abs() > AGGRESSIVE_ACCEL_THRESHOLD).sum()
    aggressive_events_per_km = (
        aggressive_events / completed_distance_km if completed_distance_km else np.nan
    )

    pct_uphill = (g["slope(º)"] > 0).mean() * 100.0

    return {
        "nominal_capacity_Wh": nominal_capacity_Wh,
        "DoD_pct": dod_pct,
        "net_energy_Wh": net_energy_Wh,
        "DoD_equivalente": dod_equivalente,
        "duracao_h": duracao_h,
        "C_rate_medio": c_rate_medio,
        "aggressive_events_per_km": aggressive_events_per_km,
        "pct_uphill": pct_uphill,
        "efficiency_mWh": last["mWh"],
        "completedDistance_km": completed_distance_km,
        "n_steps": int(n_steps),
    }


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def process_all_routes(directory: Path) -> pd.DataFrame:
    rows = []
    parse_failures = []

    for path in list_route_files(directory):
        meta = parse_filename(path)
        if not meta["parse_ok"]:
            parse_failures.append(path.name)
            continue

        df = load_route_file(path)
        df = df[df["vehID"].isin(NORMAL_STYLE_VEHIDS)]
        if df.empty:
            continue

        for veh_id, group in df.groupby("vehID"):
            feats = compute_trip_features(group, path.name)
            feats["vehID"] = veh_id
            feats["route_file"] = path.name
            feats.update({k: v for k, v in meta.items() if k != "parse_ok"})
            rows.append(feats)

    if parse_failures:
        print(f"WARNING: {len(parse_failures)} file(s) did not match the expected name pattern and were skipped:")
        for f in parse_failures:
            print(f"  - {f}")

    return pd.DataFrame(rows)


def main() -> pd.DataFrame:
    df = process_all_routes(EV_ROUTES_DIR)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(df.head(10))
    print(f"\n{len(df)} rows (vehID x route) saved to {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    main()
