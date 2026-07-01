"""
Phase 4, Interpretation A2 v2 — XGBoost with vehID as dummy

Changes from the previous version:
- vehID included as a dummy (one-hot encoding) — a legitimate proxy for
  pack nominal capacity, which resolves the systematic vehicle bias we
  observed in the previous version residuals (eUp/i3 underestimated,
  ID4/ID3 overestimated).
- Added: hyperparameter search via nested GridSearchCV inside LOO-CV, and
  SHAP values for interpretability.

Why XGBoost and not Random Forest or a deep Decision Tree:
- n=105 is small, for example — boosting with shallow trees (max_depth 2-4)
  generalizes better than a deep tree or RF with many trees in this regime.
- XGBoost has native regularization (lambda, alpha) that explicitly controls
  overfitting, unlike a simple Decision Tree.
- SHAP values are natively integrated in XGBoost, making interpretability
  easier with no extra cost.

Why vehID as dummy is legitimate here (vs. the previous version):
- In v1, we excluded vehID because it covaries with nominal_capacity_Wh
  (which is redundant with the target). But the systematic vehicle bias we
  observed shows there is real information in vehID beyond capacity —
  aerodynamic style, motor efficiency, weight, regeneration — that the
  model needs to generalize across vehicles. Including it as a dummy
  captures this without including numeric capacity directly.

Input:
    data/processed/ev/ev_trip_features.csv

Outputs (in data/processed/models/):
    a2_xgb_results.csv          — LOO-CV metrics (with and without vehID dummy)
    a2_xgb_predictions.csv      — predicted vs actual per observation
    a2_xgb_residuals.png        — residuals by vehID
    a2_xgb_shap.png             — SHAP summary plot
    a2_xgb_predicted_vs_real.png

Dependencies:
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
INPUT_FILE    = PROCESSED_DIR / "ev" / "ev_trip_features.csv"
OUTPUT_DIR    = PROCESSED_DIR / "models"

TARGET = "DoD_equivalente"

FEATURES_NUMERIC = [
    "completedDistance_km",
    "duracao_h",
    "pct_uphill",
    "aggressive_events_per_km",
    "traffic_factor",
    "occupancy",
    "auxiliaries",
    "wind",
    "efficiency_mWh",
]

RANDOM_STATE = 42

# Hyperparameter grid — conservative given n=105
# Shallow trees (max_depth 2-4) + moderate n_estimators + regularization
PARAM_GRID = {
    "xgb__n_estimators":  [50, 100, 200],
    "xgb__max_depth":     [2, 3, 4],
    "xgb__learning_rate": [0.05, 0.1, 0.2],
    "xgb__subsample":     [0.8, 1.0],
    "xgb__reg_lambda":    [1.0, 2.0],   # L2 — principal controle de overfitting
}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame, include_vehid: bool) -> np.ndarray:
    """
    Returns X with numeric features + optional vehID dummies.
    OneHotEncoder with drop='first' avoids perfect multicollinearity.
    """
    if include_vehid:
        preprocessor = ColumnTransformer([
            ("num", "passthrough", FEATURES_NUMERIC),
            ("veh", OneHotEncoder(drop="first", sparse_output=False), ["vehID"]),
        ])
        return preprocessor.fit_transform(df[FEATURES_NUMERIC + ["vehID"]]), preprocessor
    else:
        return df[FEATURES_NUMERIC].to_numpy(), None


def get_feature_names(preprocessor, df: pd.DataFrame) -> list:
    """Returns the names of all features after encoding, for the SHAP plot."""
    if preprocessor is None:
        return FEATURES_NUMERIC
    veh_cats = preprocessor.named_transformers_["veh"].categories_[0][1:]  # drop='first'
    return FEATURES_NUMERIC + [f"vehID_{c}" for c in veh_cats]


# ---------------------------------------------------------------------------
# Evaluation via LOO-CV with hyperparameter search
# ---------------------------------------------------------------------------

def evaluate_xgb(X: np.ndarray, y: np.ndarray, label: str) -> dict:
    """
    External LOO-CV + internal GridSearchCV (nested CV).
    For each LOO fold, GridSearchCV finds the best hyperparameters using
    the n-1 training examples — preventing test information leakage into
    hyperparameter selection.
    """
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))

    base_pipe = Pipeline([
        ("xgb", xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            verbosity=0,
        ))
    ])

    inner_cv = GridSearchCV(
        base_pipe, PARAM_GRID,
        cv=5,                    # 5-fold inside each LOO fold
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )

    best_params_list = []
    for i, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        inner_cv.fit(X_train, y_train)
        y_pred[test_idx] = inner_cv.predict(X_test)
        best_params_list.append(inner_cv.best_params_)

        if (i + 1) % 20 == 0:
            print(f"  {label}: {i+1}/{len(y)} folds completed...")

    r2   = r2_score(y, y_pred)
    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Most frequent hyperparameters across folds (mode)
    params_df   = pd.DataFrame(best_params_list)
    modal_params = params_df.mode().iloc[0].to_dict()

    return {
        "label":        label,
        "R2_loocv":     r2,
        "MAE_loocv":    mae,
        "RMSE_loocv":   rmse,
        "n":            len(y),
        "y_pred":       y_pred,
        "modal_params": modal_params,
    }


# ---------------------------------------------------------------------------
# Final training (all data, modal hyperparameters)
# for SHAP and importance analysis
# ---------------------------------------------------------------------------

def train_final_model(X: np.ndarray, y: np.ndarray, modal_params: dict) -> xgb.XGBRegressor:
    params = {
        k.replace("xgb__", ""): int(v) if k in ("xgb__n_estimators", "xgb__max_depth") else v
        for k, v in modal_params.items()
    }
    model = xgb.XGBRegressor(
        **params,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_residuals(df: pd.DataFrame, results: list, output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        y_true = df[TARGET].to_numpy()
        residuals = y_true - res["y_pred"]
        for veh, grp in df.groupby("vehID"):
            idx = grp.index.tolist()
            ax.scatter(res["y_pred"][idx], residuals[idx], alpha=0.75, s=30, label=veh)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("Predicted (DoD_equivalente)")
        ax.set_ylabel("Residual (actual − predicted)")
        ax.set_title(f"XGBoost — {res['label']}\nR²={res['R2_loocv']:.3f} | RMSE={res['RMSE_loocv']:.4f}")
        ax.legend(fontsize=8, title="vehID")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_predicted_vs_real(df: pd.DataFrame, results: list, output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        y_true = df[TARGET].to_numpy()
        for veh, grp in df.groupby("vehID"):
            idx = grp.index.tolist()
            ax.scatter(y_true[idx], res["y_pred"][idx], alpha=0.75, s=30, label=veh)
        lims = [min(y_true.min(), res["y_pred"].min()) - 0.02,
                max(y_true.max(), res["y_pred"].max()) + 0.02]
        ax.plot(lims, lims, "k--", linewidth=1, label="perfect")
        ax.set_xlabel("Actual (DoD_equivalente)")
        ax.set_ylabel("Predicted")
        ax.set_title(f"XGBoost — {res['label']}\nR²={res['R2_loocv']:.3f}")
        ax.legend(fontsize=8, title="vehID")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_shap(model: xgb.XGBRegressor, X: np.ndarray,
              feature_names: list, output_path: Path) -> None:
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                          show=False, plot_size=None)
        plt.title("SHAP summary — XGBoost with vehID dummy")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("  SHAP plot saved.")
    except ImportError:
        print("  WARNING: 'shap' not installed — SHAP plot skipped.")
        print("  To install: pip install shap")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    y  = df[TARGET].to_numpy()

    results = []
    predictions_out = df[["vehID", "route_file", TARGET]].copy()

    # --- Variant 1: without vehID (baseline for fair comparison) ---
    print("Evaluating XGBoost WITHOUT vehID dummy...")
    X_base, _ = build_feature_matrix(df, include_vehid=False)
    res_base   = evaluate_xgb(X_base, y, label="sem_vehID")
    results.append(res_base)
    predictions_out["pred_xgb_sem_vehid"] = res_base["y_pred"]
    print(f"  → R²={res_base['R2_loocv']:.3f} | MAE={res_base['MAE_loocv']:.4f} | "
          f"RMSE={res_base['RMSE_loocv']:.4f}")
    print(f"  Modal hyperparameters: {res_base['modal_params']}\n")

    # --- Variant 2: with vehID dummy ---
    print("Evaluating XGBoost WITH vehID dummy...")
    X_veh, preprocessor = build_feature_matrix(df, include_vehid=True)
    res_veh = evaluate_xgb(X_veh, y, label="com_vehID_dummy")
    results.append(res_veh)
    predictions_out["pred_xgb_com_vehid"] = res_veh["y_pred"]
    print(f"  → R²={res_veh['R2_loocv']:.3f} | MAE={res_veh['MAE_loocv']:.4f} | "
          f"RMSE={res_veh['RMSE_loocv']:.4f}")
    print(f"  Modal hyperparameters: {res_veh['modal_params']}\n")

    # --- vehID dummy gain ---
    delta_r2 = res_veh["R2_loocv"] - res_base["R2_loocv"]
    print(f"R² gain with vehID dummy: {delta_r2:+.4f}")

    # --- Residuals by vehID (best variant) ---
    best = max(results, key=lambda r: r["R2_loocv"])
    predictions_out["residuo"] = y - best["y_pred"]
    print(f"\n=== Mean residual by vehID — XGBoost {best['label']} ===")
    print(predictions_out.groupby("vehID")["residuo"]
          .agg(["mean", "std", "count"]).round(4).to_string())

    # --- Final training (all data) for SHAP ---
    print("\nTraining final model (all data) for SHAP...")
    feat_names = get_feature_names(preprocessor, df)
    final_model = train_final_model(X_veh, y, res_veh["modal_params"])

    # --- Salvar outputs ---
    comp_df = pd.DataFrame([{k: v for k, v in r.items()
                              if k not in ("y_pred", "modal_params")} for r in results])
    comp_df.to_csv(OUTPUT_DIR / "a2_xgb_results.csv", index=False)
    predictions_out.to_csv(OUTPUT_DIR / "a2_xgb_predictions.csv", index=False)

    plot_residuals(df.reset_index(drop=True), results,
                   OUTPUT_DIR / "a2_xgb_residuals.png")
    plot_predicted_vs_real(df.reset_index(drop=True), results,
                           OUTPUT_DIR / "a2_xgb_predicted_vs_real.png")
    plot_shap(final_model, X_veh, feat_names,
              OUTPUT_DIR / "a2_xgb_shap.png")

    print(f"\nFiles saved in: {OUTPUT_DIR}/")
    print("  a2_xgb_results.csv")
    print("  a2_xgb_predictions.csv")
    print("  a2_xgb_residuals.png")
    print("  a2_xgb_predicted_vs_real.png")
    print("  a2_xgb_shap.png")


if __name__ == "__main__":
    main()
