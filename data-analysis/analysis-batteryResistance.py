import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

csv = './data/processed/Battery/hppc_test_10cells.csv'
output = './hppc_analysis.png'
cells = ['W8', 'W9']
# ── Extração de resistência DC ────────────────────────────────────────────────

def calc_R_pulsos(t, v, c):
    """
    Detecta pulsos curtos de descarga (~0.55C, ~11s) e calcula
    R_dc = ΔV / ΔI para cada um. Retorna lista de valores em Ohms.
    """
    diff_c = np.diff(c)
    starts = np.where(diff_c > 1.5)[0]
    Rs = []
    for s in starts:
        if s + 1 >= len(c):
            continue
        # Filtra apenas pulsos com corrente ~0.55C (2.66A nas células desse dataset)
        if abs(c[s + 1] - 2.66) > 0.5:
            continue
        dI = c[s + 1] - c[s]
        dV = v[s] - v[s + 1]   # queda de tensão no início do pulso
        if dI > 0 and dV > 0:
            Rs.append(dV / dI)
    return Rs


# ── Plot principal ────────────────────────────────────────────────────────────

def plot_hppc(csv_path: str, output_path: str, cells_focus: list = None):
    print(f"Carregando {csv_path} ...")
    df = pd.read_csv(csv_path)

    all_cells = sorted(df['cell'].unique())
    all_diags  = sorted(df['diag_number'].unique())
    n_diags    = len(all_diags)

    if cells_focus is None:
        # Usa as duas células com mais diagnósticos para o zoom e trace
        counts = df.groupby('cell')['diag_number'].nunique().sort_values(ascending=False)
        cells_focus = list(counts.index[:2])
    print(f"Células em destaque (zoom/trace): {cells_focus}")

    # Paletas
    cmap_diag  = plt.cm.plasma
    diag_colors = {d: cmap_diag(i / max(n_diags - 1, 1))
                   for i, d in enumerate(all_diags)}
    cell_colors = plt.cm.tab10(np.linspace(0, 1, len(all_cells)))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#f8f8f6')
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── Plot 1: Zoom num pulso individual ─────────────────────────────────────
    diags_zoom = [all_diags[0], all_diags[len(all_diags) // 2], all_diags[-1]]

    for ci, cell in enumerate(cells_focus[:2]):
        ax = fig.add_subplot(gs[0, ci * 2:(ci + 1) * 2])
        cell_df = df[df['cell'] == cell].copy()

        for diag_n in diags_zoom:
            seg = cell_df[cell_df['diag_number'] == diag_n].copy()
            if len(seg) < 50:
                continue

            t = seg['time_s'].values
            v = seg['voltage_V'].values
            c = seg['current_A'].values
            t_rel = t - t[0]

            # Encontrar 1º pulso curto de descarga
            diff_c = np.diff(c)
            starts = np.where(diff_c > 1.5)[0]
            for s in starts:
                if s + 1 >= len(c):
                    continue
                if abs(c[s + 1] - 2.66) > 0.5:
                    continue
                i0 = max(0, s - 5)
                i1 = min(len(t_rel), s + 65)
                t_win = t_rel[i0:i1] - t_rel[s]
                v_win = v[i0:i1]
                lw = 2.0 if diag_n in [all_diags[0], all_diags[-1]] else 1.2
                ax.plot(t_win, v_win, color=diag_colors[diag_n],
                        lw=lw, label=f'Diag {diag_n}')
                break

        ax.axvline(0, color='gray', lw=0.8, linestyle='--', alpha=0.6)
        ax.set_title(f'Célula {cell} — zoom no pulso de descarga',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Tempo relativo ao pulso (s)', fontsize=9)
        ax.set_ylabel('Tensão (V)', fontsize=9)
        ax.legend(fontsize=8)
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 60)

    # ── Plot 2: Trace completo V vs t ─────────────────────────────────────────
    for ci, cell in enumerate(cells_focus[:2]):
        ax = fig.add_subplot(gs[1, ci * 2:(ci + 1) * 2])
        cell_df = df[df['cell'] == cell].copy()

        for diag_n in all_diags:
            seg = cell_df[cell_df['diag_number'] == diag_n].copy()
            if len(seg) < 50:
                continue
            t_rel = (seg['time_s'].values - seg['time_s'].values[0]) / 3600
            ax.plot(t_rel, seg['voltage_V'].values,
                    color=diag_colors[diag_n], lw=0.7, alpha=0.85)

        ax.set_title(f'Célula {cell} — trace completo HPPC (todos os diags)',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Tempo (horas)', fontsize=9)
        ax.set_ylabel('Tensão (V)', fontsize=9)
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.3)

        sm = plt.cm.ScalarMappable(
            cmap='plasma',
            norm=plt.Normalize(all_diags[0], all_diags[-1])
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('Diag nº', fontsize=8)

    # ── Plot 3: R_dc vs diagnóstico — todas as células ────────────────────────
    ax3 = fig.add_subplot(gs[2, :])

    for ci, cell in enumerate(all_cells):
        cell_df = df[df['cell'] == cell].copy()
        diag_nums = []
        R_means   = []

        for diag_n in all_diags:
            seg = cell_df[cell_df['diag_number'] == diag_n].copy()
            if len(seg) < 50:
                continue
            t   = seg['time_s'].values
            v   = seg['voltage_V'].values
            c   = seg['current_A'].values
            t_r = t - t[0]
            Rs  = calc_R_pulsos(t_r, v, c)
            if Rs:
                diag_nums.append(diag_n)
                R_means.append(np.mean(Rs) * 1000)  # mΩ

        if diag_nums:
            ax3.plot(diag_nums, R_means, 'o-',
                     color=cell_colors[ci],
                     lw=1.8, markersize=5, label=cell)

    ax3.set_title('Resistência interna (R_dc) por diagnóstico — todas as células',
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('Número do diagnóstico', fontsize=10)
    ax3.set_ylabel('Resistência DC média (mΩ)', fontsize=10)
    ax3.legend(ncol=5, fontsize=9, loc='upper left')
    ax3.set_facecolor('white')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(all_diags)

    fig.suptitle(
        'Análise HPPC — degradação da resistência interna ao longo do envelhecimento',
        fontsize=14, fontweight='bold', y=1.01
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"Salvo em: {output_path}")
    plt.close()


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    plot_hppc(csv, output, cells)