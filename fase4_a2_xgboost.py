"""
Fase 4, Interpretação A2 v2 — XGBoost com vehID como dummy

Mudanças em relação à versão anterior:
- Modelo: Decision Tree (depth=3) → XGBoost
- vehID incluído como dummy (one-hot encoding) — proxy legítima de
  capacidade nominal do pack, que resolve o viés sistemático por veículo
  que observamos nos resíduos da versão anterior (eUp/i3 subestimados,
  ID4/ID3 superestimados).
- Modelos lineares e interações removidos — foco total no XGBoost.
- Adicionado: busca de hiperparâmetros via GridSearchCV aninhado ao
  LOO-CV, e SHAP values para interpretabilidade.

Por que XGBoost e não Random Forest ou Decision Tree profunda:
- n=105 é pequeno — boosting com árvores rasas (max_depth 2-4) generaliza
  melhor que uma tree profunda ou RF com muitas árvores nesse regime.
- XGBoost tem regularização nativa (lambda, alpha) que controla overfitting
  explicitamente, ao contrário de uma Decision Tree simples.
- SHAP values estão integrados nativamente no XGBoost, facilitando
  interpretabilidade sem custo extra.

Por que vehID como dummy é legítimo aqui (vs. versão anterior):
- Na v1, excluímos vehID porque ele covaría com nominal_capacity_Wh
  (que é redundante com o target). Mas o viés sistemático que observamos
  por veículo prova que existe informação real no vehID além da capacidade
  — estilo aerodinâmico, eficiência do motor, peso, regeneração — que
  o modelo precisa para generalizar entre veículos. Incluir como dummy
  captura isso sem incluir a capacidade numérica diretamente.

Entrada:
    data/processed/ev/ev_trip_features.csv

Saídas (em data/processed/models/):
    a2_xgb_results.csv          — métricas LOO-CV (com e sem vehID dummy)
    a2_xgb_predictions.csv      — previsto vs real por observação
    a2_xgb_residuals.png        — resíduos por vehID
    a2_xgb_shap.png             — SHAP summary plot
    a2_xgb_predicted_vs_real.png

Dependências:
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import LeaveOneOut, GridSearchCV, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb

# ---------------------------------------------------------------------------
# Configuração
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

# Grid de hiperparâmetros — conservador dado n=105
# Árvores rasas (max_depth 2-4) + n_estimators moderado + regularização
PARAM_GRID = {
    "xgb__n_estimators":  [50, 100, 200],
    "xgb__max_depth":     [2, 3, 4],
    "xgb__learning_rate": [0.05, 0.1, 0.2],
    "xgb__subsample":     [0.8, 1.0],
    "xgb__reg_lambda":    [1.0, 2.0],   # L2 — principal controle de overfitting
}


# ---------------------------------------------------------------------------
# Pré-processamento
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame, include_vehid: bool) -> np.ndarray:
    """
    Retorna X com features numéricas + (opcional) dummies de vehID.
    OneHotEncoder com drop='first' para evitar multicolinearidade perfeita.
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
    """Retorna nomes de todas as features após encoding, para o SHAP plot."""
    if preprocessor is None:
        return FEATURES_NUMERIC
    veh_cats = preprocessor.named_transformers_["veh"].categories_[0][1:]  # drop='first'
    return FEATURES_NUMERIC + [f"vehID_{c}" for c in veh_cats]


# ---------------------------------------------------------------------------
# Avaliação via LOO-CV com busca de hiperparâmetros
# ---------------------------------------------------------------------------

def evaluate_xgb(X: np.ndarray, y: np.ndarray, label: str) -> dict:
    """
    LOO-CV externo + GridSearchCV interno (nested CV).
    Para cada fold LOO, o GridSearchCV encontra os melhores hiperparâmetros
    usando os n-1 exemplos de treino — evita vazamento de informação do
    teste para a seleção de hiperparâmetros.
    Nota: com n=105, isso roda 105 × n_grid fits. Pode demorar ~1-2 min.
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
        cv=5,                    # 5-fold dentro de cada fold LOO
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
            print(f"  {label}: {i+1}/{len(y)} folds concluídos...")

    r2   = r2_score(y, y_pred)
    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Hiperparâmetros mais frequentes ao longo dos folds (moda)
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
# Treino final (todos os dados, melhores hiperparâmetros modais)
# para SHAP e análise de importância
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
# Visualizações
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
        ax.set_xlabel("Previsto (DoD_equivalente)")
        ax.set_ylabel("Resíduo (real − previsto)")
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
        ax.plot(lims, lims, "k--", linewidth=1, label="perfeito")
        ax.set_xlabel("Real (DoD_equivalente)")
        ax.set_ylabel("Previsto")
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
        plt.title("SHAP summary — XGBoost com vehID dummy")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("  SHAP plot salvo.")
    except ImportError:
        print("  AVISO: 'shap' não instalado — plot SHAP pulado.")
        print("  Para instalar: pip install shap")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    y  = df[TARGET].to_numpy()

    results = []
    predictions_out = df[["vehID", "route_file", TARGET]].copy()

    # --- Variante 1: sem vehID (baseline para comparação justa) ---
    print("Avaliando XGBoost SEM vehID dummy...")
    X_base, _ = build_feature_matrix(df, include_vehid=False)
    res_base   = evaluate_xgb(X_base, y, label="sem_vehID")
    results.append(res_base)
    predictions_out["pred_xgb_sem_vehid"] = res_base["y_pred"]
    print(f"  → R²={res_base['R2_loocv']:.3f} | MAE={res_base['MAE_loocv']:.4f} | "
          f"RMSE={res_base['RMSE_loocv']:.4f}")
    print(f"  Hiperparâmetros modais: {res_base['modal_params']}\n")

    # --- Variante 2: com vehID dummy ---
    print("Avaliando XGBoost COM vehID dummy...")
    X_veh, preprocessor = build_feature_matrix(df, include_vehid=True)
    res_veh = evaluate_xgb(X_veh, y, label="com_vehID_dummy")
    results.append(res_veh)
    predictions_out["pred_xgb_com_vehid"] = res_veh["y_pred"]
    print(f"  → R²={res_veh['R2_loocv']:.3f} | MAE={res_veh['MAE_loocv']:.4f} | "
          f"RMSE={res_veh['RMSE_loocv']:.4f}")
    print(f"  Hiperparâmetros modais: {res_veh['modal_params']}\n")

    # --- Ganho do vehID dummy ---
    delta_r2 = res_veh["R2_loocv"] - res_base["R2_loocv"]
    print(f"Ganho de R² com vehID dummy: {delta_r2:+.4f}")

    # --- Resíduos por vehID (melhor variante) ---
    best = max(results, key=lambda r: r["R2_loocv"])
    predictions_out["residuo"] = y - best["y_pred"]
    print(f"\n=== Resíduo médio por vehID — XGBoost {best['label']} ===")
    print(predictions_out.groupby("vehID")["residuo"]
          .agg(["mean", "std", "count"]).round(4).to_string())

    # --- Treino final (todos os dados) para SHAP ---
    print("\nTreinando modelo final (todos os dados) para SHAP...")
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

    print(f"\nArquivos salvos em: {OUTPUT_DIR}/")
    print("  a2_xgb_results.csv")
    print("  a2_xgb_predictions.csv")
    print("  a2_xgb_residuals.png")
    print("  a2_xgb_predicted_vs_real.png")
    print("  a2_xgb_shap.png")


if __name__ == "__main__":
    main()
