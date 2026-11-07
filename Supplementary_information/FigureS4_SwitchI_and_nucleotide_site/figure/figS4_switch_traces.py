"""Figure S3: Mg2+, Switch I and Switch II along the HMR runs, and core H-bond pairs by Switch I state,
and (c) K88 position and pocket width by K88 H-bond state (k88_frames.csv from kras_core_geometry.py --dump).

Reads (from the folder given as the first argument, default switch_states_hmr/):
  {namd_3,namd_3_2,namd_3_3}_measures.csv   written by kras_switch_states.py (1 row per ns)
  core_by_switch1/core_hbonds.csv           written by kras_core_by_switch1.py
Writes pub_figS8_states.png in the working folder.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'analysis')]
os.chdir(HERE)
import os, sys
import numpy as np, pandas as pd
from pub_style import *

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, 'switch_states_hmr')
REPS = ['namd_3', 'namd_3_2', 'namd_3_3']
EDGE = (730, 9270)                       # analysed window of the coherence (ns)
ROWS = [('T35OG1_Mg', 'T35 Oγ1–Mg²⁺ (Å)', [2.6]),
        ('sw1_phi', 'Switch I angle φ (rad)', [0.25, 1.23, 3.0]),
        ('Y32OH_gammaO', 'Y32 OH–γ-O (Å)', [3.2]),
        ('G60N_gammaO', 'G60 N–γ-O (Å)', [3.2]),
        ('G12CA_PG', 'G12 Cα–Pγ (Å)', [])]
PAIRS = [('R102 (D69 | Q99)', 'R102 bonds\nD69, Q99'), ('R73 (V103 | D105)', 'R73 bonds\nV103, D105'),
         ('K88 (E91 | D92)', 'K88 bonds\nE91, D92')]
LINE = '#2b2b2b'
SMOOTH = 25                              # running mean (frames = ns) drawn over the raw series


def runs(mask, t):
    """contiguous True runs of a boolean series -> list of (t0, t1)"""
    m = np.r_[False, mask, False].astype(int); d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(t[a], t[b - 1] + (t[1] - t[0])) for a, b in zip(s, e)]


M = {r: pd.read_csv(os.path.join(D, f'{r}_measures.csv')) for r in REPS}
C = pd.read_csv(os.path.join(D, 'core_by_switch1', 'core_hbonds.csv'))

MODE = 'supp'   # this folder draws only the supplementary traces
ALLROWS = ROWS
if MODE == 'main':
    ROWS = []                                     # traces moved to the supplementary figure; panel a is a schematic
    nr = 1
    fig = plt.figure(figsize=(7.1, 4.6))
    gs = fig.add_gridspec(nr + 2, 3, height_ratios=[2.3, .02, 1.35], hspace=.3, wspace=.12, top=.96, bottom=.17)
else:
    ROWS = ALLROWS; nr = len(ROWS)
    fig = plt.figure(figsize=(7.1, 7.2))
    gs = fig.add_gridspec(nr, 3, hspace=.28, wspace=.12, top=.95, bottom=.06)
for c, r in enumerate(REPS):
    m = M[r]; t = m.time_ns.values / 1000; off = m.T35OG1_Mg.values > 2.6
    for k, (col, lab, refs) in enumerate(ROWS):
        ax = fig.add_subplot(gs[k, c])
        offs = pd.Series(off.astype(float)).rolling(SMOOTH, center=True, min_periods=1).mean().values > .5
        for t0, t1 in runs(offs, t):
            ax.axvspan(t0, t1, color='#e9e9e9', lw=0, zorder=0)
        for a_, b_ in ((0, EDGE[0] / 1000), (EDGE[1] / 1000, 10)):
            excl(ax, a_, b_)
        if MODE == 'main' and k == 0:
            on_end = next((a for a, _ in runs(offs, t)), 10)
            if on_end > 1.5:
                ax.text((EDGE[0] / 1000 + on_end) / 2,  1.02, 'T35 on Mg²⁺', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=6.2, color=INK2)
            ax.text((max(on_end, EDGE[0] / 1000) + EDGE[1] / 1000) / 2 if on_end < 9 else 9, 1.02, 'T35 off', transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=6.2, color=INK2, zorder=5)
        y = m[col].values
        ax.plot(t, y, color='#c4c4c4', lw=.3, rasterized=True, zorder=2)
        ys = pd.Series(y).rolling(SMOOTH, center=True, min_periods=1).mean().values
        ax.plot(t, ys, color=LINE, lw=.7, rasterized=True, zorder=3)
        for y in refs:
            ax.axhline(y, color=MUTE, lw=.5, ls='--', zorder=1)
        ax.set_xlim(0, 10)
        lo = np.nanpercentile(np.concatenate([M[q][col].values for q in REPS]), .2)
        hi = np.nanpercentile(np.concatenate([M[q][col].values for q in REPS]), 99.8)
        pad = .06 * (hi - lo); ax.set_ylim(min(lo, *(refs or [lo])) - pad, max(hi, *(refs or [hi])) + pad)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.tick_params(labelbottom=(k == nr - 1))
        if c == 0:
            ax.set_ylabel(lab, fontsize=6.4, labelpad=2)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        if k == 0:
            ax.set_title(f'replica {c + 1}', fontsize=7.4, pad=12 if MODE == 'main' else 3)
            if c == 0 and MODE == 'main':
                panel(ax, 'a', x=-0.34, y=1.08)
        if k == nr - 1 and c == 1:
            ax.set_xlabel('time (µs)')

if MODE != 'main':
    save(fig, 'FigureS4.png'); print('wrote supp'); sys.exit()
exec(open('fig4a_schematic.py').read())
schematic(fig.add_subplot(gs[0, :]))
# (b) one small panel per core H-bond pair: % of frames with both bonds formed, observed vs expected if independent
GREY = {0: '#bdbdbd', 1: '#7d7d7d', 2: '#262626'}
MK = ['o', 's', '^']
PB = PAIRS + [('T144 (N116 | Q150)', 'T144 bonds\nN116, Q150')]
TB = ['R102 bonds D69 and Q99', 'R73 bonds V103 and D105', 'K88 bonds E91 and D92', 'T144 bonds N116 and Q150']
subb = gs[nr + 1, :].subgridspec(1, 4, wspace=.42)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
for p, (key, _) in enumerate(PB):
    ax = fig.add_subplot(subb[0, p])
    if p == 0:
        panel(ax, 'b', x=-0.5, y=1.08)
    ax.set_title(TB[p], fontsize=6.6, pad=4, color=INK)
    top = 0
    for s_, st in enumerate(['T35 on', 'T35 off']):
        obs, exp_ = [], []
        for j_, r in enumerate(REPS):
            q = C[(C.replica == r) & (C.state == st) & (C.pair == key)]
            if not len(q) or q.frames.iloc[0] < 200:
                continue
            o, e = 100 * q.both.iloc[0], 100 * q.occ_first.iloc[0] * q.occ_second.iloc[0]
            obs.append((j_, o)); exp_.append((j_, e))
        for k_, (vals, fc, dx) in enumerate([(obs, '#3a3a3a', -.19), (exp_, '#d6d6d6', .19)]):
            if not vals:
                continue
            x = s_ + dx
            m = np.mean([v for _, v in vals]); top = max(top, max(v for _, v in vals))
            ax.bar(x, m, width=.36, color=fc, ec='none', zorder=2)
            for j_, v in vals:
                ax.plot(x + (j_ - 1) * .09, v, MK[j_], ms=3.2, mfc='white', mec=INK, mew=.6, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['T35 on\nMg²⁺', 'T35 off\nMg²⁺'], fontsize=6.2)
    ax.set_xlim(-.6, 1.6); ax.set_ylim(0, max(top * (1.45 if key.startswith('T144') else 1.25), 1))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4)); ax.tick_params(axis='x', length=0)
    if p == 0:
        ax.set_ylabel('frames with both\nbonds formed (%)', fontsize=6.6)
    if key.startswith('T144'):
        ax.text(.55, .98, 'observed: never\nformed together', transform=ax.transAxes, ha='center', va='top', fontsize=5.9, color=INK2)
h = [Patch(color='#3a3a3a', label='observed'), Patch(color='#d6d6d6', label='expected if the two bonds were independent')]
h += [Line2D([], [], ls='', marker=MK[j_], ms=3.6, mfc='white', mec=INK, label=f'replica {j_ + 1}') for j_ in range(3)]
fig.legend(handles=h, loc='upper center', bbox_to_anchor=(.5, ax.get_position().y0 - .075), ncol=5, fontsize=6.2,
           handlelength=1.2, columnspacing=1.2, frameon=False)

save(fig, 'pub_figS3_states.png')
print('wrote pub_figS3_states.png')
