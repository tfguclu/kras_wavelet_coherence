import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'analysis')]
os.chdir(HERE)
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import os
HERE = os.path.dirname(os.path.abspath(__file__))
INK = '#1a1a1a'; INK2 = '#4d4d4d'; MUTE = '#8c8c8c'; RULE = '#dcdad4'
OBS = {'chi': '#b83b72', 'sasa': '#2a9d8f', 'hb': '#d08c1f'}; OBSN = {'chi': 'χ', 'sasa': 'SASA', 'hb': 'H-bond'}
BAT, BAT_F = '#3b6ea8', '#f1f5fa'
PER, PER_F = '#6b4fa3', '#f5f2fa'
STEP = '#5f5e5a'; ASIDE = '#7a7a7a'; OKC = '#3f7d3a'
plt.rcParams.update({'font.family': 'Inter', 'font.size': 7, 'xtick.labelsize': 6.4, 'ytick.labelsize': 6.4,
                     'axes.edgecolor': INK2, 'axes.linewidth': .6, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'xtick.direction': 'in', 'ytick.direction': 'in'})
SEQ = 'GMTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHK'
lab = lambda i: f'{SEQ[i]}{i}'
cn = pd.read_csv(os.path.join(DATA, 'cnull_summary.csv')); pp = pd.read_csv(os.path.join(DATA, 'ctrl_per_assoc.csv')); pp['k'] = pp.sets_at_or_above_real
REAL2 = {'chi': 14, 'sasa': 25, 'hb': 7}; order = ['chi', 'sasa', 'hb']

W, H = 7.2, 7.9
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
AR = W / H


def box(x, y, w, h, fc, ec, lw=.7, r=1.4, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, mutation_aspect=AR, zorder=z))


def arrow(p, q, col=INK2, lw=.9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=8, color=col, lw=lw, shrinkA=0, shrinkB=0))


def T(x, y, s, size=7, w='regular', c=INK, ha='left', va='top', ls=1.3, z=5):
    ax.text(x, y, s, fontsize=size, fontweight=w, color=c, ha=ha, va=va, linespacing=ls, zorder=z)


def letter(x, y, s): T(x, y, s, 10.5, 'bold', INK)


def badge(x, y, n, col):
    ax.add_patch(matplotlib.patches.Ellipse((x, y), 2.5, 2.5 * AR, fc=col, ec='none', zorder=6))
    T(x, y, str(n), 6.6, 'bold', 'white', ha='center', va='center', z=7)


# ================= a: making the controls =================
letter(.8, 99.3, 'a'); T(4.6, 98.9, 'Making the controls', 8.4, 'semibold')
T(30.5, 98.75, 'one control = every signal shifted, all pairs re-tested; repeated 200 times', 6.6, c=MUTE)
steps = [('Real data', '~81,000', 'signal pairs per replica\n× 3 replicas'),
         ('Shift every signal', '≥ 1 µs', 'own random circular offset;\nvalues kept, timing lost'),
         ('Rerun the analysis', '242,701', 'pair tests per control\n80–120 ns · ≥ 0.75 · ≥ 500 ns'),
         ('Repeat', '× 200', '48.5 million pair tests →\n200 lists of random pairs')]
bw, gap, x0, y0, bh = 21.9, 2.8, 1.6, 81.0, 14.6
for i, (ttl, big, cap) in enumerate(steps):
    x = x0 + i * (bw + gap); last = i == 3
    if last:
        for k in (2, 1): box(x + .8 * k, y0 - 1.0 * k, bw, bh, '#ebe8e2', '#c9c6be', lw=.5)
    box(x, y0, bw, bh, '#f4f2ee' if last else 'white', INK2 if last else '#a9a7a0', lw=.9 if last else .6, z=2)
    badge(x + 2.4, y0 + bh - 2.2, i + 1, STEP)
    T(x + 4.4, y0 + bh - 2.2, ttl, 7.2, 'semibold', INK, va='center')
    T(x + bw / 2, y0 + bh - 5.0, big, 12.5, 'semibold', INK, ha='center')
    T(x + bw / 2, y0 + 5.1, cap, 6.1, c=INK2, ha='center', va='center', ls=1.3)
    if i < 3: arrow((x + bw + .4, y0 + bh / 2), (x + bw + gap - .4, y0 + bh / 2))
xs = x0 + 3 * (bw + gap) + bw / 2 + 1.6
ax.plot([xs, xs], [y0 - 2.0, 76.7], color=INK2, lw=.9)
ax.plot([25.3, xs], [76.7, 76.7], color=INK2, lw=.9)
arrow((25.3, 76.7), (25.3, 74.4)); arrow((xs, 76.7), (xs, 74.4))
box(36, 75.4, 28, 2.6, 'white', 'white', lw=0, r=.5, z=3)
T(50, 76.7, 'the same 200 lists, used two ways', 6.7, 'medium', INK2, ha='center', va='center')

# ================= b: batch =================
bx, by, bW, bH = 1.6, 38.6, 47.4, 35.6
box(bx, by, bW, bH, BAT_F, BAT, lw=.9)
letter(bx + 1.0, by + bH - .9, 'b')
T(bx + 5.2, by + bH - 1.3, 'Batch test: count the pairs', 8.2, 'semibold', BAT)
T(bx + 5.2, by + bH - 4.9, 'Pairs found in ≥ 2 replicas: real count vs the 200 control counts.\nTest: the real count must exceed every control.', 6.3, c=INK2, ls=1.35)
iax = fig.add_axes([(bx + 9.0) / 100, (by + 11.6) / 100, (bW - 11.5) / 100, 13.8 / 100]); iax.set_facecolor('none')
for r, o in enumerate(order):
    y = 2 - r; v = cn[cn.observable == o].n_ge2_reps.values; u, c = np.unique(v, return_counts=True)
    for uu, cc in zip(u, c):
        iax.add_patch(Rectangle((uu - .42, y), .84, .62 * cc / c.max(), fc='#b9b9b9', ec='none'))
    iax.plot([-.5, 28], [y, y], color=RULE, lw=.6, zorder=0)
    iax.plot(REAL2[o], y + .22, marker='D', ms=7, color=OBS[o], mec='white', mew=.7, zorder=3)
    iax.text(REAL2[o] + .9, y + .22, f'{REAL2[o]}', va='center', fontsize=7.4, fontweight='semibold', color=OBS[o])
    iax.text(v.max(), y - .06, f'max {v.max()}', va='top', ha='center', fontsize=5.7, color=MUTE)
    iax.text(-1.3, y + .22, OBSN[o], va='center', ha='right', fontsize=7.2, fontweight='semibold', color=OBS[o])
iax.set_xlim(-.6, 28); iax.set_ylim(-.35, 2.75); iax.set_yticks([]); iax.spines['left'].set_visible(False)
iax.set_xticks([0, 5, 10, 15, 20, 25]); iax.set_xlabel('pairs in ≥ 2 replicas', fontsize=6.5, labelpad=1.5)
iax.text(1.0, 1.0, '■ 200 controls   ◆ real', transform=iax.transAxes, ha='right', va='top', fontsize=6.0, color=MUTE)
box(bx + 2.0, by + 0.8, bW - 4.0, 6.6, 'white', '#c6d3e4', lw=.6, r=.8)
T(bx + 3.4, by + 6.8, '✓ More pairs than random shifting gives, for all three observables;', 6.4, 'medium', INK)
T(bx + 3.4, by + 4.6, '   χ narrowly (14 vs 11). In ≥ 1 replica, controls give 76% / 8% / 44% as many\n   χ / SASA / H-bond pairs → the 304 single-replica pairs are not interpreted.', 5.9, c=INK2, ls=1.3)

# ================= c: per pair =================
px, py, pW, pH = 51.0, 38.6, 47.4, 35.6
box(px, py, pW, pH, PER_F, PER, lw=.9)
letter(px + 1.0, py + pH - .9, 'c')
T(px + 5.2, py + pH - 1.3, 'Per-pair test: search for each pair', 8.2, 'semibold', PER)
T(px + 5.2, py + pH - 4.9, 'For each of the 46 pairs in ≥ 2 replicas, k = controls that find it\nin as many replicas as the real data.  p = (1 + k) / 201', 6.3, c=INK2, ls=1.35)
cell = 1.75; cg = .38; ch = cell * AR
groups = [('chi', 5), ('sasa', 8), ('hb', 3)]
gx = px + 3.6; gtop = py + 24.0
for o, ncol in groups:
    d = pp[pp.observable == o].sort_values(['k', 'real_replicas'], ascending=[True, False]).reset_index(drop=True)
    T(gx, gtop + 1.8, f'{OBSN[o]}  ({len(d)})', 6.6, 'semibold', OBS[o], va='bottom')
    for i, r in d.iterrows():
        cx = gx + (i % ncol) * (cell + cg); cy = gtop - (i // ncol + 1) * (ch + cg * AR)
        if r.k == 0:
            ax.add_patch(Rectangle((cx, cy), cell, ch, fc=OBS[o] if r.real_replicas == 3 else 'white', ec=OBS[o], lw=.8, zorder=3))
        else:
            ax.add_patch(Rectangle((cx, cy), cell, ch, fc=ASIDE, ec=ASIDE, lw=.8, zorder=3))
            T(cx + cell / 2, cy + ch / 2, str(r.k), 5.6, 'bold', 'white', ha='center', va='center')
    gx += ncol * (cell + cg) + 2.6
ly = py + 12.6
for j, (fc, ec, s, dx) in enumerate([(INK2, INK2, 'core', 0), ('white', INK2, 'other reproducible', 7.2), (ASIDE, ASIDE, 'set aside (k inside)', 22.6)]):
    lx = px + 3.6 + dx
    ax.add_patch(Rectangle((lx, ly), 1.5, 1.5 * AR, fc=fc, ec=ec, lw=.8))
    T(lx + 2.1, ly + .7, s, 6.0, c=INK2, va='center')
ok = pp[pp.k == 0]; bad = pp[pp.k > 0].sort_values('k')
oy = py + 1.6
box(px + 2.0, oy, 15.4, 9.0, 'white', '#9fbf98', lw=.7, r=.8)
T(px + 9.7, oy + 7.9, 'k = 0  →  reproducible', 6.8, 'semibold', OKC, ha='center')
T(px + 9.7, oy + 3.6, f'{len(ok)} pairs\n' + '  '.join(f'{OBSN[o]} {(ok.observable == o).sum()}' for o in order), 6.0, c=INK2, ha='center', va='center', ls=1.4)
box(px + 18.6, oy, 26.8, 9.0, 'white', '#b5b3ad', lw=.7, r=.8)
T(px + 32.0, oy + 7.9, 'k ≥ 1  →  set aside (4 χ pairs)', 6.8, 'semibold', INK2, ha='center')
lines = [f'{lab(int(a.split("-")[0]))}–{lab(int(a.split("-")[1]))}   k = {k}' for a, k in zip(bad.association, bad.k)]
T(px + 20.2, oy + 3.6, '\n'.join(lines[:2]), 6.0, c=INK2, va='center', ls=1.4)
T(px + 33.4, oy + 3.6, '\n'.join(lines[2:]), 6.0, c=INK2, va='center', ls=1.4)

# ================= d: variants =================
vx, vy, vW, vH = 1.6, 1.0, 96.8, 35.2
box(vx, vy, vW, vH, '#fafaf8', '#b9b7b0', lw=.7)
letter(vx + 1.0, vy + vH - .9, 'd')
T(vx + 5.2, vy + vH - 1.3, 'Variants: the same tests with other randomization or settings', 8.2, 'semibold')
T(vx + 5.2, vy + vH - 4.9, 'Pairs in ≥ 2 replicas, real (largest in any of the 200 controls); masked schemes give ranges over the 200 masks.', 6.2, c=INK2)
cols = [vx + 5.2, vx + 25.5, vx + 53.0, vx + 63.5, vx + 73.5, vx + 83.5]
hdr = [('variant', None), ('what changes', None), ('χ', 'chi'), ('SASA', 'sasa'), ('H-bond', 'hb'), ('core pairs\nreached', None)]
hy = vy + vH - 8.2
for c, (s, o) in zip(cols, hdr):
    T(c, hy, s, 6.4, 'semibold', OBS[o] if o else INK2, ls=1.15)
rowsV = [('Baseline', 'none', '14 (11)', '25 (1)', '7 (1)', '0'),
         ('Seam masked', 'shift seam removed from episodes', '6–13 (7)', '9–22 (1)', '4–7 (1)', '1 (χ D126–K128)'),
         ('≥ 1 µs relative offset', 'nearly aligned pairs not counted', '8–14 (8)', '9–23 (0)', '2–7 (1)', '0'),
         ('Shared χ offsets', 'χ angles of a residue shifted together', '14 (8)', '–', '–', '0'),
         ('Smoothing 1s', 'own 200 controls', '37 (35)', '46 (2)', '10 (2)', '0'),
         ('Smoothing 3s', 'own 200 controls', '9 (6)', '18 (1)', '7 (1)', '0'),
         ('6 period bands', 'own 200 controls per band', '✓ 40–480 ns', '✓ all 6', '✓ all 6', '–'),
         ('Threshold 0.60–0.90', 'baseline controls', '✓ 0.70–0.85', '✓ all', '✓ all', '–')]
rh = 2.55; ry = hy - 4.2
for n, r in enumerate(rowsV):
    if n % 2 == 0:
        ax.add_patch(Rectangle((vx + 3.6, ry - rh + .65), vW - 6.0, rh, fc='#f0eee9', ec='none', zorder=1))
    for j, (c, s) in enumerate(zip(cols, r)):
        col = INK if j == 0 else INK2
        if j in (2, 3, 4) and s.startswith('✓'): col = OKC
        T(c, ry, s, 6.2, 'medium' if j == 0 else 'regular', col, z=4)
    ry -= rh
T(vx + 5.2, vy + 1.1, 'Real counts exceed every control except for χ: at 10–30 ns, at thresholds 0.60, 0.65 and 0.90, and in 1 of 200 seam-masked comparisons.', 6.2, 'medium', INK, va='bottom')

for f in ['png', 'pdf', 'svg']:
    fig.savefig(f'FigureS1.{f}', dpi=400, facecolor='white')
