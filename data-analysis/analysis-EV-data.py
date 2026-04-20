import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pandas import read_csv

# ── Data ──────────────────────────────────────────────────────────────────────
df = read_csv('./data/processed/EV/1_amurrio_durango_0.35_4_500_0_output.csv')

vehicles  = ['EV1', 'EV4', 'EV7', 'EV10', 'EV13']
colors    = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
ev_data   = {v: df[df['vehID'] == v].sort_values('step') for v in vehicles}

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0F1117',
    'axes.facecolor':   '#1A1D27',
    'axes.edgecolor':   '#2E3147',
    'axes.labelcolor':  '#C8CEDE',
    'axes.titlecolor':  '#E8EAF6',
    'axes.titlesize':   11,
    'axes.labelsize':   9,
    'axes.grid':        True,
    'grid.color':       '#2E3147',
    'grid.linewidth':   0.6,
    'xtick.color':      '#8890A8',
    'ytick.color':      '#8890A8',
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'lines.linewidth':  1.2,
    'legend.facecolor': '#1A1D27',
    'legend.edgecolor': '#2E3147',
    'legend.fontsize':  8,
    'legend.labelcolor':'#C8CEDE',
    'font.family':      'DejaVu Sans',
})

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    'EV Route Analysis  ·  amurrio–durango (speedFactor 0.35 | 500 m slope step)',
    fontsize=14, color='#E8EAF6', y=0.98, fontweight='bold'
)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38,
                       left=0.06, right=0.98, top=0.93, bottom=0.06)

axes = [fig.add_subplot(gs[r, c]) for r, c in [
    (0,0),(0,1),(0,2),
    (1,0),(1,1),(1,2),
    (2,0),(2,1),
]]
ax_legend = fig.add_subplot(gs[2, 2])   # dedicated legend panel

# ── Plot specs ────────────────────────────────────────────────────────────────
plots = [
    ('step', 'SoC(%)',                     'State of Charge (%)',          'Step', 'SoC (%)'),
    ('step', 'speed(m/s)',                  'Speed over Route',             'Step', 'Speed (m/s)'),
    ('step', 'acceleration(m/s\xb2)',       'Acceleration',                 'Step', 'Accel (m/s²)'),
    ('step', 'totalEnergyConsumed(Wh)',     'Cumulative Energy Consumed',   'Step', 'Energy (Wh)'),
    ('step', 'totalEnergyRegenerated(Wh)', 'Cumulative Energy Regenerated','Step', 'Regen (Wh)'),
    ('completedDistance(km)', 'SoC(%)',     'SoC vs Completed Distance',    'Distance (km)', 'SoC (%)'),
    ('step', 'alt',                         'Altitude Profile',             'Step', 'Altitude (m)'),
    ('step', 'remainingRange(km)',          'Remaining Range',              'Step', 'Range (km)'),
]

for ax, (xcol, ycol, title, xlabel, ylabel) in zip(axes, plots):
    for ev, color in zip(vehicles, colors):
        d = ev_data[ev]
        ax.plot(d[xcol], d[ycol], color=color, alpha=0.85, label=ev)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

# ── Legend panel ──────────────────────────────────────────────────────────────
ax_legend.set_axis_off()
handles = [
    plt.Line2D([0], [0], color=c, linewidth=2.5, label=v)
    for v, c in zip(vehicles, colors)
]
ax_legend.legend(handles=handles, loc='center', title='Vehicle ID',
                 title_fontsize=10, fontsize=10, frameon=True)

plt.savefig('./data-analysis/analysis-EV-data.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Plot saved to data-analysis/analysis-EV-data.png")
