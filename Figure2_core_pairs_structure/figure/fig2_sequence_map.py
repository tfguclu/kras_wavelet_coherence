"""Sequence map of the 16 core coherent pairs (panel b of the structure figure).
Secondary structure, regions and core residues along residues 1-167; arcs join the two residues of
each chi or SASA core pair (above the axis) and the shared residue of each core H-bond pair with its
two partners (below; solid, in phase; dashed, antiphase). Writes pub_fig_seqmap.png."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'analysis')]
os.chdir(HERE)
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
HALO = [pe.withStroke(linewidth=2.2, foreground='white')]
from matplotlib.patches import FancyArrow, Rectangle, Arc
from matplotlib.lines import Line2D
from pub_style import *

SEQ = 'MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHK'
aa = lambda i: SEQ[i - 1] + str(i)
CHI = [(128, 132), (126, 128), (1, 52), (3, 54), (52, 54)]
SASA = [(69, 102), (75, 104), (88, 92), (94, 133), (10, 16), (8, 16), (80, 93)]
HB = [(102, (69, 99), 'in'), (73, (103, 105), 'in'), (88, (91, 92), 'anti'), (144, (116, 150), 'anti')]
# region bands on the sequence (alpha3 and alpha4 added for the interface discussed in the text)
BANDS = REG + [('α3', 87, 104)]
BANDC = dict(REGL, **{'α3': '#e6eef6', 'α4': '#eeeeee'})
BANDE = dict(REGC, **{'α3': '#6f9cc8', 'α4': '#9a9a9a'})

core = sorted({r for p in CHI + SASA for r in p} | {s for s, _, _ in HB} | {r for _, pr, _ in HB for r in pr})
obs_of = {}
for o, L in (('chi', CHI), ('sasa', SASA)):
    for p in L:
        for r in p:
            obs_of.setdefault(r, set()).add(o)
for s, pr, _ in HB:
    for r in (s,) + pr:
        obs_of.setdefault(r, set()).add('hb')

fig = plt.figure(figsize=(7.1, 2.7))
ax = fig.add_axes([.03, .04, .94, .9])
ax.set_xlim(-2, 170); ax.set_ylim(-1.0, 1.38); ax.axis('off')
Y = 0.0
# region bands
for nm, a, b in BANDS:
    ax.add_patch(FancyBboxPatch((a - .5, Y - .25), b - a + 1, .5, boxstyle='round,pad=0,rounding_size=0.9',
                                mutation_aspect=.08, fc=BANDC[nm], ec='none', zorder=0))
    if nm != 'α3':
        ax.text((a + b) / 2, Y - .22, nm, ha='center', va='bottom', fontsize=6.3, color=BANDE[nm],
                fontweight='semibold', zorder=6,
                path_effects=[pe.withStroke(linewidth=2.4, foreground=BANDC[nm])])
# secondary structure
ax.plot([.5, 167.5], [Y, Y], color=MUTE, lw=.8, zorder=1)
for nm, a, b in SSEL:
    if nm.startswith('α'):
        ax.add_patch(Rectangle((a - .5, Y - .07), b - a + 1, .14, fc='#bdbdbd', ec='none', zorder=2))
    else:
        ax.add_patch(FancyArrow(a - .5, Y, b - a + .5, 0, width=.09, head_width=.17, head_length=1.4,
                                length_includes_head=True, fc='#bdbdbd', ec='none', zorder=2))
    ax.text((a + b) / 2, Y + .1, nm, ha='center', va='bottom', fontsize=5.8, color=INK2, zorder=6, path_effects=HALO)
ax.text(-.6, Y, 'N', ha='right', va='center', fontsize=6, color=MUTE)
ax.text(168.6, Y, 'C', ha='left', va='center', fontsize=6, color=MUTE)
# core residues
for r in core:
    o = sorted(obs_of[r])
    ax.plot(r, Y, 'o', ms=3.4, mfc='white', mec=INK, mew=.75, zorder=7)

H = lambda span: .18 + .62 * (span / 52) ** .7      # arc height grows with sequence span
def arc(a, b, c, up=True, ls='-', lw=1.2):
    w = abs(b - a); h = H(w)
    ax.add_patch(Arc(((a + b) / 2, Y), w, 2 * h, theta1=0 if up else 180, theta2=180 if up else 360,
                     color=c, lw=lw, ls=ls, zorder=4, alpha=.92, capstyle='round'))
for o, L in (('chi', CHI), ('sasa', SASA)):
    for a, b in L:
        arc(a, b, OBS[o])
for s, pr, ph in HB:
    for p in pr:
        arc(s, p, OBS['hb'], up=False, ls='-' if ph == 'in' else (0, (2.2, 1.4)))

# labels: residues of core pairs, staggered to avoid overlaps
lab_up = sorted({r for p in CHI + SASA for r in p})
lab_dn = sorted({s for s, _, _ in HB} | {r for _, pr, _ in HB for r in pr})
def spread(rs, gap=6.2):
    """1D repel: label x positions at least gap apart, clusters centred on their residues"""
    xs = list(map(float, rs))
    for _ in range(200):
        moved = False
        for i in range(1, len(xs)):
            d = xs[i] - xs[i - 1]
            if d < gap:
                m = (gap - d) / 2; xs[i - 1] -= m; xs[i] += m; moved = True
        if not moved:
            break
    return xs
def place(rs, ylab, yline, va):
    for r, x in zip(rs, spread(rs)):
        ax.text(x, ylab, aa(r), ha='center', va=va, fontsize=5.8, color=INK, rotation=0, zorder=8)
        y1 = ylab - .03 if va == 'bottom' else ylab + .03
        ax.plot([r, r, x], [Y + (.05 if va == 'bottom' else -.05), yline, y1], color='#e3e3e3', lw=.45, zorder=0, solid_capstyle='round')
place(lab_up, 1.02, .9, 'bottom')
place(lab_dn, -.84, -.74, 'top')

h = [Line2D([], [], color=OBS['chi'], lw=1.1, label='χ pair'), Line2D([], [], color=OBS['sasa'], lw=1.1, label='SASA pair'),
     Line2D([], [], color=OBS['hb'], lw=1.1, label='H-bond pair, in phase'),
     Line2D([], [], color=OBS['hb'], lw=1.1, ls=(0, (2.2, 1.4)), label='H-bond pair, antiphase')]
ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=6.3, handlelength=2.2, columnspacing=1.4)
panel(ax, 'b', x=-.03, y=.95)
save(fig, 'fig2_bottom_sequence_map.png')
print('wrote pub_fig_seqmap.png')
