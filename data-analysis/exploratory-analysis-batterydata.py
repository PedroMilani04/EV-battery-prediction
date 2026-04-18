import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('./data/processed/Battery/capacity_test_10cells.csv')

cells = sorted(df['cell'].unique())
n_cells = len(cells)

# Build a 2-row grid wide enough to fit all cells
n_cols = 5
n_rows = -(-n_cells // n_cols)  # ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4),
                         sharex=False, sharey=False)
axes = axes.flatten()

for idx, cell in enumerate(cells):
    ax = axes[idx]
    df_cell = df[df['cell'] == cell].copy()

    for diag in sorted(df_cell['diag_number'].unique()):
        seg = df_cell[df_cell['diag_number'] == diag].copy()

        # Relative time within the diagnostic
        seg['time_rel'] = seg['time_s'] - seg['time_s'].iloc[0]

        # Cumulative discharged capacity (Ah) for the X axis
        dt = seg['time_rel'].diff().fillna(0)
        seg['cap_cumulative'] = (np.abs(seg['current_A']) * dt / 3600).cumsum()

        ax.plot(seg['cap_cumulative'], seg['voltage_V'],
                label=f'Diag {diag}', alpha=0.8, linewidth=0.9)

    ax.set_title(f'Cell {cell}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Discharged capacity (Ah)', fontsize=9)
    ax.set_ylabel('Voltage (V)', fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    ax.tick_params(labelsize=8)

# Hide any unused subplots
for idx in range(n_cells, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle('Capacity test — all cells (all diagnostics)', fontsize=14,
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('capacity_all_cells.png', dpi=150, bbox_inches='tight')
plt.show()