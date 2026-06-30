"""
Fase 4, Etapa 3 — Interpretação A1: análise de correlação

Objetivo: não é um modelo preditivo — é uma análise descritiva para
entender QUAIS features de condução/rota mais se relacionam com
DoD_equivalente alto (ou seja, com viagens que consomem mais "fração de
bateria", e portanto acumulam EFC mais rápido).

Entrada:
    data/processed/ev/ev_trip_features.csv  (saída da Fase 2)

Saídas (tudo salvo em data/processed/analysis/):
    correlation_matrix.csv       — matriz de correlação completa (Pearson e Spearman)
    correlation_with_target.csv  — ranking de correlação de cada feature com DoD_equivalente
    correlation_heatmap.png      — heatmap visual da matriz de correlação
    scatter_top_features.png     — scatterplots das 4 features mais correlacionadas vs DoD_equivalente
    correlation_by_vehicle.csv   — a mesma correlação, mas quebrada por vehID (para ver se o
                                    padrão muda entre modelos de veículo)

Como rodar:
    python fase4_correlation_analysis.py

Dependências:
    pandas, numpy, matplotlib, seaborn, scipy
    pip install pandas numpy matplotlib seaborn scipy

Notas de interpretação:
- Correlação não é causalidade. Isso é puramente exploratório.
- Pearson mede relação LINEAR; Spearman mede relação MONOTÔNICA (mais
  robusta a outliers e a relações não-lineares, mas não-paramétrica).
  Reportamos os dois — se divergem muito, é sinal de relação não-linear.
- target = DoD_equivalente (é literalmente o EFC por viagem, então essa
  análise responde "o que aumenta o consumo de EFC por viagem").
- Features que são redundantes com o target por construção (ex: DoD_pct,
  net_energy_Wh, C_rate_medio) são mantidas na matriz mas marcadas
  separadamente — elas são derivadas quase diretamente do mesmo cálculo
  e vão sempre correlacionar quase perfeitamente; não são insight, são
  esperado por definição.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
INPUT_FILE = PROCESSED_DIR / "ev" / "ev_trip_features.csv"
OUTPUT_DIR = PROCESSED_DIR / "analysis"

TARGET = "DoD_equivalente"

# Features candidatas para a análise (exclui identificadores e colunas de
# metadado puro, que não são "comportamento de condução" propriamente).
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

# Features redundantes por construção com o target — mantidas na matriz
# completa, mas excluídas do ranking de "insight" para não confundir
# correlação-por-definição com correlação-por-comportamento-de-condução.
DEFINITIONALLY_REDUNDANT = ["DoD_pct", "net_energy_Wh", "nominal_capacity_Wh"]

TOP_N_SCATTER = 4


# ---------------------------------------------------------------------------
# Carga e validação
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_target = TARGET not in df.columns
    if missing_target:
        raise ValueError(f"Coluna target '{TARGET}' não encontrada em {path.name}")

    missing_features = [f for f in CANDIDATE_FEATURES if f not in df.columns]
    if missing_features:
        print(f"AVISO: features não encontradas no CSV, serão ignoradas: {missing_features}")

    return df


# ---------------------------------------------------------------------------
# Cálculo de correlações
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
# Visualizações
# ---------------------------------------------------------------------------

def plot_heatmap(corr_matrix: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1, square=True, cbar_kws={"label": "Correlação de Pearson"},
    )
    plt.title("Matriz de correlação — features de rota (d-EVD)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_top_scatters(df: pd.DataFrame, target_corr: pd.DataFrame, target: str,
                       n: int, output_path: Path) -> None:
    insight_only = target_corr[~target_corr["feature"].isin(DEFINITIONALLY_REDUNDANT)]
    top_features = insight_only.head(n)["feature"].tolist()

    if not top_features:
        print("Nenhuma feature disponível para scatterplot.")
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
# Pipeline principal
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(INPUT_FILE)
    available_features = [f for f in CANDIDATE_FEATURES if f in df.columns]
    all_columns_for_matrix = available_features + DEFINITIONALLY_REDUNDANT + [TARGET]
    all_columns_for_matrix = [c for c in all_columns_for_matrix if c in df.columns]

    # Matriz completa (inclui as redundantes por definição, para contexto)
    corr_matrix = compute_correlation_matrix(df, all_columns_for_matrix)
    corr_matrix.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    # Ranking de correlação com o target (todas as features, incluindo redundantes,
    # mas plot de scatter exclui as redundantes — ver compute_target_correlations)
    target_corr = compute_target_correlations(
        df, available_features + DEFINITIONALLY_REDUNDANT, TARGET
    )
    target_corr.to_csv(OUTPUT_DIR / "correlation_with_target.csv", index=False)

    # Correlação por veículo (insight comportamental — vai além da Fase 4 plano original,
    # mas é direto de computar e relevante: será que a relação muda por modelo de carro?)
    by_vehicle = compute_correlation_by_vehicle(df, available_features, TARGET)
    by_vehicle.to_csv(OUTPUT_DIR / "correlation_by_vehicle.csv", index=False)

    # Visualizações
    plot_heatmap(corr_matrix, OUTPUT_DIR / "correlation_heatmap.png")
    plot_top_scatters(df, target_corr, TARGET, TOP_N_SCATTER, OUTPUT_DIR / "scatter_top_features.png")

    # ---- Print de resumo no terminal ----
    print("=" * 70)
    print(f"Análise de correlação — target = {TARGET}")
    print("=" * 70)
    print(f"\n{len(df)} observações (vehID x rota) carregadas de {INPUT_FILE.name}\n")

    print("--- Ranking de correlação com o target (todas as features) ---")
    print(target_corr.to_string(index=False))
    print(f"\n(*) Features em {DEFINITIONALLY_REDUNDANT} são redundantes por construção")
    print("    com DoD_equivalente — correlação alta nelas é esperada, não é insight.\n")

    print("--- Maior divergência entre Pearson e Spearman (sinal de não-linearidade) ---")
    target_corr["delta_pearson_spearman"] = (target_corr["pearson_r"] - target_corr["spearman_r"]).abs()
    print(target_corr.sort_values("delta_pearson_spearman", ascending=False).head(5)[
        ["feature", "pearson_r", "spearman_r", "delta_pearson_spearman"]
    ].to_string(index=False))

    print(f"\nArquivos salvos em: {OUTPUT_DIR}/")
    print("  - correlation_matrix.csv")
    print("  - correlation_with_target.csv")
    print("  - correlation_by_vehicle.csv")
    print("  - correlation_heatmap.png")
    print("  - scatter_top_features.png")


if __name__ == "__main__":
    main()
