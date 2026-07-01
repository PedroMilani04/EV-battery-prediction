"""
Phase 4, Step 3 — Interpretation A1: correlation analysis

Objective: this is not a predictive model — it is a descriptive analysis
to understand WHICH driving/route features are most associated with
high DoD_equivalente (i.e., trips that consume a larger battery fraction
and therefore accumulate EFC faster).

Input:
    data/processed/ev/ev_trip_features.csv  (Phase 2 output)

Outputs (all saved in data/processed/analysis/):
    correlation_matrix.csv       — full correlation matrix (Pearson and Spearman)
    correlation_with_target.csv  — ranking of each feature's correlation with DoD_equivalente
    correlation_heatmap.png      — visual heatmap of the correlation matrix
    scatter_top_features.png     — scatterplots of the 4 top-correlated features vs DoD_equivalente
    correlation_by_vehicle.csv   — the same correlation broken down by vehID (to see whether
                                    the pattern changes across vehicle models)

How to run:
    python fase4_correlation_analysis.py

Dependencies:
    pandas, numpy, matplotlib, seaborn, scipy
    pip install pandas numpy matplotlib seaborn scipy

Interpretation notes:
- Correlation is not causation. This is purely exploratory.
- Pearson measures LINEAR relationship; Spearman measures MONOTONIC
  relationship (more robust to outliers and non-linear relations, but nonparametric).
  We report both — if they diverge substantially, it is a sign of non-linear
  relationship.
- target = DoD_equivalente (it is literally EFC per trip, so this analysis
  answers "what increases EFC consumption per trip").
- Features that are redundant with the target by construction (e.g. DoD_pct,
  net_energy_Wh, C_rate_medio) are retained in the matrix but marked
  separately — they are derived almost directly from the same calculation
  and will always correlate nearly perfectly; they are not insight, they are
  expected by definition.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
INPUT_FILE = PROCESSED_DIR / "ev" / "ev_trip_features.csv"
OUTPUT_DIR = PROCESSED_DIR / "analysis"

TARGET = "DoD_equivalente"

# Candidate features for analysis (excludes identifiers and pure metadata
# columns that are not actual driving behavior).
CANDIDATE_FEATURES = [
    "C_rate_medio",
    "aggressive_events_per_km",
    "pct_uphill",
    "efficiency_mWh",
    "completedDistance_km",
    "duracao_h",
    "traffic_factor",
    "occupancy",
    "auxiliaries",
    "wind",
]

# Features redundant with the target by construction — kept in the full
# matrix, but excluded from the "insight" ranking so that definitionally
# induced correlation is not mistaken for driving-behavior correlation.
DEFINITIONALLY_REDUNDANT = ["DoD_pct", "net_energy_Wh", "nominal_capacity_Wh"]

TOP_N_SCATTER = 4


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_target = TARGET not in df.columns
    if missing_target:
        raise ValueError(f"Target column '{TARGET}' not found in {path.name}")

    missing_features = [f for f in CANDIDATE_FEATURES if f not in df.columns]
    if missing_features:
        print(f"WARNING: features not found in CSV will be ignored: {missing_features}")

    return df


# ---------------------------------------------------------------------------
# Correlation calculations
# ---------------------------------------------------------------------------

def compute_correlation_matrix(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return df[columns].corr(method="pearson")


def compute_target_correlations(df: pd.DataFrame, features: list, target: str) -> pd.DataFrame:
    rows = []
    for feat in features:
        if feat not in df.columns:
            continue
        valid = df[[feat, target]].dropna()
        if len(valid) < 3 or valid[feat].nunique() <= 1:
            pearson_r, pearson_p = np.nan, np.nan
            spearman_r, spearman_p = np.nan, np.nan
        else:
            pearson_r, pearson_p = pearsonr(valid[feat], valid[target])
            spearman_r, spearman_p = spearmanr(valid[feat], valid[target])

        rows.append({
            "feature": feat,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "n_obs": len(valid),
        })

    result = pd.DataFrame(rows)
    result["abs_pearson_r"] = result["pearson_r"].abs()
    result = result.sort_values("abs_pearson_r", ascending=False).drop(columns="abs_pearson_r")
    return result


def compute_correlation_by_vehicle(df: pd.DataFrame, features: list, target: str) -> pd.DataFrame:
    rows = []
    for veh_id, group in df.groupby("vehID"):
        for feat in features:
            if feat not in group.columns:
                continue
            valid = group[[feat, target]].dropna()
            if len(valid) < 3 or valid[feat].nunique() <= 1:
                r, p = np.nan, np.nan
            else:
                r, p = pearsonr(valid[feat], valid[target])
            rows.append({"vehID": veh_id, "feature": feat, "pearson_r": r, "pearson_p": p, "n_obs": len(valid)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_heatmap(corr_matrix: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1, square=True, cbar_kws={"label": "Pearson correlation"},
    )
    plt.title("Correlation matrix — route features (d-EVD)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_top_scatters(df: pd.DataFrame, target_corr: pd.DataFrame, target: str,
                       n: int, output_path: Path) -> None:
    insight_only = target_corr[~target_corr["feature"].isin(DEFINITIONALLY_REDUNDANT)]
    top_features = insight_only.head(n)["feature"].tolist()

    if not top_features:
        print("No features available for scatterplot.")
        return

    n_cols = 2
    n_rows = int(np.ceil(len(top_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, feat in zip(axes, top_features):
        sns.scatterplot(data=df, x=feat, y=target, hue="vehID", alpha=0.7, ax=ax, legend=False)
        sns.regplot(data=df, x=feat, y=target, scatter=False, ax=ax, color="black", line_kws={"linewidth": 1})
        r_row = target_corr[target_corr["feature"] == feat].iloc[0]
        ax.set_title(f"{feat}\nPearson r={r_row['pearson_r']:.3f} | Spearman ρ={r_row['spearman_r']:.3f}")

    for ax in axes[len(top_features):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(INPUT_FILE)
    available_features = [f for f in CANDIDATE_FEATURES if f in df.columns]
    all_columns_for_matrix = available_features + DEFINITIONALLY_REDUNDANT + [TARGET]
    all_columns_for_matrix = [c for c in all_columns_for_matrix if c in df.columns]

    # Full matrix (includes definitionally redundant features for context)
    corr_matrix = compute_correlation_matrix(df, all_columns_for_matrix)
    corr_matrix.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    # Correlation ranking with the target (all features, including redundant ones,
    # but scatter plot excludes the redundant features — see compute_target_correlations)
    target_corr = compute_target_correlations(
        df, available_features + DEFINITIONALLY_REDUNDANT, TARGET
    )
    target_corr.to_csv(OUTPUT_DIR / "correlation_with_target.csv", index=False)

    # Correlation by vehicle (behavioral insight — goes beyond the original
    # Phase 4 plan, but is easy to compute and relevant: does the relationship
    # change by vehicle model?)
    by_vehicle = compute_correlation_by_vehicle(df, available_features, TARGET)
    by_vehicle.to_csv(OUTPUT_DIR / "correlation_by_vehicle.csv", index=False)

    # Visualizações
    plot_heatmap(corr_matrix, OUTPUT_DIR / "correlation_heatmap.png")
    plot_top_scatters(df, target_corr, TARGET, TOP_N_SCATTER, OUTPUT_DIR / "scatter_top_features.png")

    # ---- Summary output to terminal ----
    print("=" * 70)
    print(f"Correlation analysis — target = {TARGET}")
    print("=" * 70)
    print(f"\n{len(df)} observations (vehID x route) loaded from {INPUT_FILE.name}\n")

    print("--- Correlation ranking with the target (all features) ---")
    print(target_corr.to_string(index=False))
    print(f"\n(*) Features in {DEFINITIONALLY_REDUNDANT} are redundant by construction")
    print("    with DoD_equivalente — high correlation there is expected, not insight.\n")

    print("--- Largest divergence between Pearson and Spearman (sign of nonlinearity) ---")
    target_corr["delta_pearson_spearman"] = (target_corr["pearson_r"] - target_corr["spearman_r"]).abs()
    print(target_corr.sort_values("delta_pearson_spearman", ascending=False).head(5)[
        ["feature", "pearson_r", "spearman_r", "delta_pearson_spearman"]
    ].to_string(index=False))

    print(f"\nFiles saved in: {OUTPUT_DIR}/")
    print("  - correlation_matrix.csv")
    print("  - correlation_with_target.csv")
    print("  - correlation_by_vehicle.csv")
    print("  - correlation_heatmap.png")
    print("  - scatter_top_features.png")


if __name__ == "__main__":
    main()
