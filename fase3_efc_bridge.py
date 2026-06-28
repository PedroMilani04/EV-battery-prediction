"""
Fase 3 — Ponte (EFC)

Entradas (vindas de data/processed/):
    data/processed/battery/battery_soh_curve_corrigido.csv  (saída da Fase 1)
    data/processed/ev/ev_trip_features.csv                  (saída da Fase 2)

Saída:
    data/processed/bridge/predicted_degradation.csv

Premissas e decisões documentadas (importante ler antes de interpretar o output):

1. ORDEM DAS VIAGENS (item 3.1 do plano): o d-EVD não tem timestamp nem
   sequência real de uso — cada vehID roda as mesmas 21 rotas como
   simulações independentes, não como um histórico cronológico de um
   veículo real. `idx` é só o identificador da rota (1 a 21), não ordem
   de ocorrência. Para construir uma trajetória cumulativa ILUSTRATIVA,
   ordenamos por `idx` de forma arbitrária e reprodutível — isso afeta
   o FORMATO da curva intermediária, mas não o EFC_total final (que é
   uma soma, invariante à ordem).

2. REGRESSÃO SoH(EFC) (item 3.3): ajustamos uma reta forçada pela origem
   (SoH=1 em EFC=0), separadamente por `charge_rate` e também um modelo
   "pooled" com todas as células não descontinuadas. Usamos isso para
   gerar 3 cenários: pessimista (maior taxa de fade observada, células
   3C), otimista (menor taxa, células C/4) e médio (pooled). Como o
   d-EVD não informa o comportamento de carregamento do veículo, não há
   como saber qual cenário é o "certo" — por isso reportamos a faixa.

3. ESCALA DE EFC: a soma de DoD_equivalente ao longo de 21 viagens é
   tipicamente pequena (ordem de poucas unidades de EFC) comparada ao
   range observado no laboratório (até ~245 EFC). Isso é esperado: os
   21 trajetos representam uma fração mínima da vida útil do veículo,
   não a vida inteira. SoH previsto vai ficar perto de 1.0 — não é bug.

4. Células descontinuadas (W3, W7) são excluídas do ajuste da curva
   (comportamento anômalo, não representativo de fade "normal").
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuração
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
    Ajusta SoH = 1 + k*EFC (reta forçada pela origem EFC=0 -> SoH=1).
    k tende a ser negativo (degradação). Retorna k, R² e o range de EFC
    observado (para detectar extrapolação depois).
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
        if len(group) >= 3:  # evita ajuste instável com poucos pontos
            curves[rate] = fit_soh_efc_line(group["EFC_lab"].to_numpy(), group["SoH"].to_numpy())

    # cenário pessimista = maior |k| (degrada mais rápido); otimista = menor |k|
    rate_curves = {k: v for k, v in curves.items() if k != "pooled"}
    if rate_curves:
        pessimista_rate = min(rate_curves, key=lambda r: rate_curves[r]["k"])  # k mais negativo
        otimista_rate = max(rate_curves, key=lambda r: rate_curves[r]["k"])    # k menos negativo
        curves["_pessimista_rate"] = pessimista_rate
        curves["_otimista_rate"] = otimista_rate

    return curves


def print_curve_diagnostics(curves: dict) -> None:
    print("=== Curvas SoH(EFC) ajustadas (reta forçada pela origem) ===")
    for name, c in curves.items():
        if name.startswith("_"):
            continue
        print(f"  {name:>8s}: k={c['k']:.6f}/EFC | R²={c['r2']:.3f} | "
              f"EFC observado=[{c['efc_min']:.1f}, {c['efc_max']:.1f}] | n={c['n_points']}")
    if "_pessimista_rate" in curves:
        print(f"  -> cenário pessimista = taxa de carga {curves['_pessimista_rate']}")
        print(f"  -> cenário otimista   = taxa de carga {curves['_otimista_rate']}")


# ---------------------------------------------------------------------------
# 3.1 + 3.2 — Ordenar viagens (arbitrário) e acumular EFC por veículo
# ---------------------------------------------------------------------------

def build_vehicle_trajectories(ev_df: pd.DataFrame) -> pd.DataFrame:
    df = ev_df.sort_values(["vehID", "idx"]).copy()  # ordem arbitrária, ver docstring
    df["trip_order"] = df.groupby("vehID").cumcount() + 1
    # EFC_cumulative = soma direta de DoD_equivalente entre viagens.
    # O fator 0.6 do protocolo de laboratório é específico do ciclo de
    # envelhecimento UDDS (80%->20% SOC) e não se aplica aqui — cada
    # DoD_equivalente já é a fração real de capacidade usada na viagem.
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
        print(f"AVISO: {n_extrap} ponto(s) com EFC acumulado acima do range "
              "observado no laboratório (extrapolação).")

    print()
    print("=== EFC total e SoH previsto final, por veículo ===")
    summary = result_df.groupby("vehID").agg(
        EFC_total=("EFC_cumulative", "max"),
        SoH_final_medio=("soh_previsto_medio", "min"),
        SoH_final_pessimista=("soh_previsto_pessimista", "min") if "soh_previsto_pessimista" in result_df else ("soh_previsto_medio", "min"),
        SoH_final_otimista=("soh_previsto_otimista", "min") if "soh_previsto_otimista" in result_df else ("soh_previsto_medio", "min"),
    )
    print(summary)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSalvo em: {OUTPUT_FILE}")
    return result_df


if __name__ == "__main__":
    main()
