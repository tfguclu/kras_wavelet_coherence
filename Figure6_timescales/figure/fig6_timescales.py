import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'analysis')]
os.chdir(HERE)
from pub_style import *
import numpy as np, pandas as pd
from matplotlib.patches import Rectangle
S = pd.read_csv(os.path.join(DATA, 'multiband_summary.csv'))   # 200 controls per band
R = pd.read_csv(os.path.join(DATA, 'multiband_reproducible.csv'))
BANDS = ['P10', 'P20', 'P40', 'P80', 'P160', 'P320']
BL = ['10–15', '20–30', '40–60', '80–120', '160–240', '320–480']
CHANCE = {('chi', 'P10'), ('chi', 'P20')}   # chi at 10-30 ns: controls reach the real recurrence (200 controls)

fig = plt.figure(figsize=(7.1, 4.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], wspace=0.75, hspace=0.55)
gl = gs[:, 0].subgridspec(3, 1, hspace=0.55)
x = np.arange(6)
for r, o in enumerate(['chi', 'sasa', 'hb']):
    ax = fig.add_subplot(gl[r])
    s = S[S.observable == o].set_index('band').reindex(BANDS)
    for k, (col, lo, hi, mk, lab) in enumerate([('real_ge2', 'surr_ge2_min', 'surr_ge2_max', 'o', '≥ 2 replicas'),
                                                  ('real_3', 'surr_r3_min', 'surr_r3_max', 's', '3 replicas')]):
        dx = -0.26 if k == 0 else 0.10; dm = dx + 0.16
        ax.vlines(x + dx, s[lo], s[hi], color='#9e9e9e', lw=3.2, alpha=.55, zorder=1)
        ax.plot(x + dx, s[lo], '_', color='#8a8a8a', ms=6, mew=1.2, zorder=2); ax.plot(x + dx, s[hi], '_', color='#8a8a8a', ms=6, mew=1.2, zorder=2)
        ax.plot(x + dm, s[col], mk, ms=4.2 if k == 0 else 3.6, mfc=OBS[o] if k == 0 else 'white', mec=OBS[o],
                mew=1, zorder=3, label=f'real, {lab}')
    for b in [i for i, bn in enumerate(BANDS) if (o, bn) in CHANCE]:
        pass
    ax.set_yscale('symlog', linthresh=10, linscale=1.2)
    ax.set_ylim(-0.5, 260); ax.set_yticks([0, 5, 10, 50, 100]); ax.set_yticklabels(['0', '5', '10', '50', '100'])
    ax.set_xticks(x); ax.set_xticklabels(BL if r == 2 else [], rotation=45, ha='right', fontsize=7.4); ax.tick_params(axis='y', labelsize=7.4)
    ax.set_xlim(-.55, 5.55)
    ax.text(0.99, 0.93, OBSN[o], transform=ax.transAxes, ha='right', va='top', color=OBS[o], fontsize=8.4, fontweight='semibold')
    if r == 1:
        ax.set_ylabel('recurring coherent pairs', fontsize=8.4)
    if r == 0:
        panel(ax, 'a', x=-0.28, y=1.05); AX0 = ax
        from matplotlib.lines import Line2D
        h = [Line2D([], [], marker='o', ls='', mfc=INK2, mec=INK2, ms=4, label='real, ≥ 2 replicas'),
             Line2D([], [], marker='s', ls='', mfc='white', mec=INK2, ms=3.6, label='real, 3 replicas'),
             Line2D([], [], color='#9e9e9e', lw=3.2, alpha=.55, label='control range (200 controls)')]
        LEG = h
ax.set_xlabel('period band (ns)', fontsize=8.4)
ax.legend(handles=LEG, loc='upper left', bbox_to_anchor=(-0.08, -0.78), ncol=2, fontsize=7.4, handletextpad=.7, handlelength=1.2, columnspacing=2.0)

# b: associations across bands
items = [
    ('Switch II–α3 interface', [('hb', '69-102 | 99-102', 'H-bond  R102: D69 | Q99'), ('sasa', '69-102', 'SASA  D69–R102'),
                      ('chi', '99-102', 'χ  Q99–R102'), ('hb', '73-103 | 73-105', 'H-bond  R73: V103 | D105'),
                      ('sasa', '75-104', 'SASA  G75–K104')]),
    ('α3, α3–α4 and β4–α3', [('hb', '88-91 | 88-92', 'H-bond  K88: E91 | D92'), ('sasa', '88-92', 'SASA  K88–D92'),
                      ('sasa', '94-133', 'SASA  H94–L133'), ('sasa', '80-93', 'SASA  C80–I93')]),
    ('Nucleotide site and P-loop', [('hb', '116-144 | 144-150', 'H-bond  T144: N116 | Q150'), ('sasa', '10-16', 'SASA  G10–K16'),
                      ('sasa', '8-16', 'SASA  V8–K16')]),
    ('α4', [('chi', '128-132', 'χ  K128–D132'), ('chi', '126-128', 'χ  D126–K128')]),
    ('N terminus–β1–β3', [('chi', '3-54', 'χ  E3–D54'), ('chi', '52-54', 'χ  L52–D54'), ('chi', '1-52', 'χ  M1–L52')]),
    ('DMS overlap', [('sasa', '17-32', 'SASA  S17–Y32')]),
]
rows, labels, groups = [], [], []
for g, its in items:
    groups.append((g, len(rows)))
    for o, key, lab in its:
        m = R[(R.observable == o) & (R.association == key)]
        assert len(m) == 1, (o, key)
        rows.append((o, m[BANDS].values[0].astype(float))); labels.append(lab)
V = np.array([v for _, v in rows])
ax = fig.add_subplot(gs[:, 1])
from matplotlib.colors import ListedColormap, BoundaryNorm
cmap = ListedColormap(['#ffffff', '#d4d4d4', '#8f8f8f', '#2b2b2b']); norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4)
ax.imshow(V, cmap=cmap, norm=norm, aspect='auto')
for (r, c), v in np.ndenumerate(V):
    ch = (rows[r][0], BANDS[c]) in CHANCE
    if ch:
        ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, facecolor='#f4f4f4', edgecolor='#c8c8c8', hatch='////', lw=0, zorder=2))
    ax.text(c, r, int(v), ha='center', va='center', fontsize=5.6, zorder=3,
            color=MUTE if ch else ('white' if v >= 2.5 else '#444'))
ax.set_xticks(range(6)); ax.set_xticklabels(BL, rotation=45, ha='right', fontsize=6)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=5.9)
for t, (o, _) in zip(ax.get_yticklabels(), rows):
    t.set_color(OBS[o])
ax.set_xticks(np.arange(-.5, 6), minor=True); ax.set_yticks(np.arange(-.5, len(labels)), minor=True)
ax.grid(which='minor', color='white', lw=.8); ax.tick_params(which='both', length=0)
for s_ in ax.spines.values(): s_.set_visible(False)
for g, i0 in groups[1:]:
    ax.axhline(i0 - .5, color=INK2, lw=.6, zorder=4)
for g, i0 in groups:
    ax.text(5.65, i0 - .35, g, fontsize=5.6, color=MUTE, va='top', ha='left', clip_on=False, style='italic')
ax.set_xlabel('period band (ns)')
ax.text(0, 1.012, 'replicas in which the coherent pair is detected', transform=ax.transAxes, fontsize=6.3, color=INK2)
fig.canvas.draw(); rr = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
xl = min(inv.transform(t.get_window_extent(rr))[0, 0] for t in ax.get_yticklabels())
ya = inv.transform(AX0.transAxes.transform((0, 1.05)))[1]
fig.text(xl, ya, 'b', fontsize=10, fontweight='bold', color=INK, va='bottom', ha='left')
save(fig, 'Figure6.png')
