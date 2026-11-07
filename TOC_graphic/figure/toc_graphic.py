"""TOC graphic, 3.25 x 1.75 in, 300 dpi TIF/EPS. Schematic only: synthetic signals, no results."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'analysis')]
os.chdir(HERE)
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, gaussian_filter1d
plt.rcParams.update({'font.family': 'Inter', 'font.size': 7, 'mathtext.fontset': 'custom', 'mathtext.it': 'Inter:italic'})
INK, MUTE, CARD = '#1f2430', '#6b7280', '#f4f5f8'
CI, CJ, GOLD = INK, INK, '#e8a317'
S2, A3 = '#3b6fb6', '#7a4fa0'
OC, OS, OH = '#b83b72', '#2a9d8f', '#d08c1f'
fig = plt.figure(figsize=(3.25, 1.75), dpi=300)
# cards
cards = [(.012, .035, .35), (.395, .035, .27), (.698, .035, .29)]
for x, y, w in cards:
    fig.patches.append(FancyBboxPatch((x, y), w, .93, boxstyle='round,pad=0,rounding_size=.035', transform=fig.transFigure,
                                      fc=CARD, ec='none', zorder=-5))
def title(x, s): fig.text(x, .885, s, ha='center', va='center', fontsize=6.8, color=INK, fontweight='semibold')
title(.53, 'wavelet coherence'); title(.843, 'local networks')
for xc in (.3785, .6815):
    fig.patches.append(FancyArrowPatch((xc - .03, .47), (xc + .03, .47), transform=fig.transFigure,
                                       arrowstyle='simple,head_length=1.1,head_width=1.5,tail_width=.55', mutation_scale=6,
                                       fc=INK, ec='white', lw=.8, zorder=10))
# ---- signals
t = np.linspace(0, 10, 1500)
wins = [(1.0, 2.6), (4.5, 6.1), (7.8, 9.4)]
gate = sum(np.exp(-((t - (a + b) / 2) / ((b - a) / 2.1)) ** 8) for a, b in wins)
def drift(seed): x = gaussian_filter1d(np.random.default_rng(seed).normal(0, 1, t.size), 60, mode='wrap'); return x / x.std()
sI = gate * np.sin(2 * np.pi * t / .42) + .38 * (1 - gate) * drift(1)
sJ = gate * np.sin(2 * np.pi * t / .42 + .3) + .38 * (1 - gate) * drift(7)
ax = fig.add_axes([.085, .16, .265, .64]); ax.axis('off'); ax.patch.set_alpha(0); ax.set_xlim(0, 10); ax.set_ylim(-2.55, 2.4)
for a, b in wins: ax.axvspan(a, b, ymin=.04, ymax=.98, color=GOLD, alpha=.22, lw=0)
ax.plot(t, sI + 1.2, color=CI, lw=.7); ax.plot(t, sJ - 1.2, color=CJ, lw=.7)
ax.annotate('', xy=(10, -2.45), xytext=(0, -2.45), arrowprops=dict(arrowstyle='-|>', color=MUTE, lw=.5, mutation_scale=4))
ax.text(5, -2.5, 'time', ha='center', va='top', fontsize=6, color=MUTE)
fig.text(.05, .645, 'residue i', fontsize=6, color=CI, va='center', ha='center', rotation=90, fontweight='medium')
fig.text(.05, .335, 'residue j', fontsize=6, color=CJ, va='center', ha='center', rotation=90, fontweight='medium')
# ---- coherence map
bx = fig.add_axes([.445, .19, .2, .6])
T, P = np.meshgrid(np.linspace(0, 10, 400), np.linspace(0, 1, 160))
M = .08 + .06 * np.random.default_rng(2).random(T.shape)
for a, b in wins: M += np.exp(-((T - (a + b) / 2) / ((b - a) / 2.1)) ** 4) * np.exp(-((P - .55) / .12) ** 2)
M += .35 * np.exp(-((T - 3.5) / .45) ** 2) * np.exp(-((P - .2) / .07) ** 2)
M = gaussian_filter(M, 3)
cmap = LinearSegmentedColormap.from_list('c', ['#1d2340', '#3a2f6b', '#a33f6e', GOLD, '#fff2c6'])
bx.imshow(M, origin='lower', aspect='auto', extent=[0, 10, 0, 1], cmap=cmap, vmin=.08, vmax=.9)
for yy in (.42, .68): bx.axhline(yy, color='white', lw=.5, ls=(0, (2, 1.5)), alpha=.8)
bx.text(9.7, .7, 'band', ha='right', va='bottom', fontsize=5.6, color='white')
bx.set_xticks([]); bx.set_yticks([])
for s in bx.spines.values(): s.set_visible(False)
bx.set_xlabel('time', fontsize=6, labelpad=1.5, color=MUTE); bx.set_ylabel('period', fontsize=6, labelpad=1.5, color=MUTE)
# ---- helices and links
cx = fig.add_axes([.705, .06, .275, .76]); cx.axis('off'); cx.patch.set_alpha(0); cx.set_xlim(0, 1); cx.set_ylim(-.05, 1.05); cx.set_aspect('auto')
def helix(p0, p1, col, n=7):
    p0, p1 = np.array(p0), np.array(p1); s = np.linspace(0, 1, 300); d = p1 - p0; nrm = np.array([-d[1], d[0]]) / np.hypot(*d)
    base = p0 + np.outer(s, d)
    cx.plot(*np.array([p0, p1]).T, color=col, lw=8.5, alpha=.18, solid_capstyle='round', zorder=1)
    w = .045 * np.sin(2 * np.pi * n * s)
    pts = base + np.outer(w, nrm)
    front = np.cos(2 * np.pi * n * s) > 0
    cx.plot(*pts.T, color=col, lw=.6, alpha=.45, zorder=2)
    seg = np.where(front, pts.T, np.nan)
    cx.plot(seg[0], seg[1], color=col, lw=1.3, zorder=3, solid_capstyle='round')
helix((.1, .7), (.52, .93), S2); helix((.3, .08), (.93, .38), A3)
cx.text(.3, .99, 'Switch II', ha='center', fontsize=6.5, color=S2, fontweight='semibold')
cx.text(.86, .14, 'α3', ha='center', fontsize=7, color=A3, fontweight='semibold')
nodes = {'a': (.19, .745), 'b': (.4, .87), 'c': (.5, .18), 'd': (.75, .3)}
for (u, v), col, rad in [(('a', 'c'), OC, -.2), (('b', 'd'), OS, -.22), (('b', 'c'), OH, .12)]:
    cx.annotate('', xy=nodes[v], xytext=nodes[u], arrowprops=dict(arrowstyle='-', color=col, lw=1.4, connectionstyle=f'arc3,rad={rad}', shrinkA=2.5, shrinkB=2.5), zorder=4)
for x, y in nodes.values(): cx.scatter(x, y, s=14, color='white', ec=INK, lw=.6, zorder=5)
r = fig.canvas.get_renderer(); parts = [('χ', OC), (' · ', INK), ('SASA', OS), (' · ', INK), ('H-bonds', OH)]
tx = [fig.text(0, .885, p, fontsize=6.8, color=c, fontweight='semibold', va='center') for p, c in parts]
ws = [x.get_window_extent(r).width / fig.bbox.width for x in tx]; x0 = .187 - sum(ws) / 2
for x, w in zip(tx, ws): x.set_x(x0); x0 += w
fig.savefig('TOC_graphic.tif', dpi=300, pil_kwargs={'compression': 'tiff_lzw'})
fig.savefig('TOC_graphic.png', dpi=1200)
plt.rcParams['ps.fonttype'] = 42
fig.savefig('TOC_graphic.eps')
