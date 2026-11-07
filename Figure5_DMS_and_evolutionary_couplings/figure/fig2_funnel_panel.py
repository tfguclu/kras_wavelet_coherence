from pub_style import *
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, Circle

KEYS = ['chi', 'sasa', 'hb']
STAGES = [('All coherent pairs', [179, 104, 67], '≥ 1 replica'),
          ('Reproducible', [14, 25, 7], '≥ 2 replicas'),
          ('Kept', [10, 25, 7], 'not matched by controls'),
          ('Core', [5, 7, 4], 'all 3 replicas')]
STEPS = [('Recurrence in ≥ 2', 'set aside: 304 pairs\nfound in 1 replica'),
         ('Per-pair check', 'set aside: 4 χ pairs that\nrecur as often in controls'),
         ('Recurrence in all 3', 'set aside: 26 pairs in 2 replicas\n(reproducible, not interpreted)'),
         ('Sensitivity analysis', 'all 16 found under all\n9 settings')]
XC = [9, 30, 51, 72]          # bar centres
BW = 3.4                          # bar width
MID = 17.5                        # midline
K = 14.0 / np.sqrt(350)           # bar height ∝ √count
BOX = (86.5, 100)                 # final box x-range


def segs(x, nums):
    """stacked segments (top→bottom χ, SASA, H-bond); returns list of (y0, y1)"""
    h = K * np.sqrt(sum(nums)); top = MID + h / 2; out = []
    for n in nums:
        d = h * n / sum(nums); out.append((top - d, top)); top -= d
    return out, h


def draw(ax):
    ax.set_xlim(0, 100); ax.set_ylim(0, 33); ax.set_aspect('equal'); ax.axis('off')
    S = [segs(x, n) for x, (_, n, _) in zip(XC, STAGES)]
    # flows between stages, one band per observable
    targets = [(XC[k + 1] - BW / 2, S[k + 1][0]) for k in range(3)] + [(BOX[0], S[3][0])]
    for k in range(4):
        x0 = XC[k] + BW / 2; x1, s1 = targets[k]; s0 = S[k][0]
        xs = np.linspace(x0, x1, 60); t = (1 - np.cos(np.pi * (xs - x0) / (x1 - x0))) / 2
        for (a0, b0), (a1, b1), key in zip(s0, s1, KEYS):
            lo = a0 + (a1 - a0) * t; hi = b0 + (b1 - b0) * t
            ax.fill_between(xs, lo, hi, color=OBS[key], alpha=.18, lw=0, zorder=1)
    # bars and labels
    for k, ((name, nums, sub), x) in enumerate(zip(STAGES, XC)):
        sg, h = S[k]
        for (y0, y1), key in zip(sg, KEYS):
            ax.add_patch(Rectangle((x - BW / 2, y0), BW, y1 - y0, fc=OBS[key], ec='white', lw=.4, zorder=3))
        ax.text(x, MID + h / 2 + .8, f'{sum(nums)}', ha='center', va='bottom', fontsize=10, fontweight='bold', color=INK)
        ax.text(x, 32.6, name, ha='center', va='top', fontsize=7, fontweight='semibold', color=INK)
        ax.text(x, 30.4, sub, ha='center', va='top', fontsize=6.2, color=INK2)
        yb = MID - h / 2 - .7
        for j, (n, key) in enumerate(zip(nums, KEYS)):
            ax.text(x + (j - 1) * 4.3, yb, str(n), ha='center', va='top', fontsize=6.8, fontweight='semibold', color=OBS[key])
    # final box
    bx0, bx1 = BOX; by0, by1 = MID - 7, MID + 7
    ax.add_patch(FancyBboxPatch((bx0, by0), bx1 - bx0, by1 - by0, boxstyle='round,pad=0,rounding_size=1.2',
                                fc=BAND, ec=INK2, lw=.7, zorder=3))
    cx = (bx0 + bx1) / 2
    ax.text(cx, by1 - 1.4, 'Interpreted', ha='center', va='top', fontsize=7, fontweight='semibold', color=INK, zorder=4)
    ax.text(cx, MID + .3, '16 core pairs\n10 residue groups', ha='center', va='center', fontsize=6.4, color=INK, linespacing=1.45, zorder=4)
    ax.text(cx, by0 + 1.2, '4 near Switch II–α3,\n1 at nucleotide site', ha='center', va='bottom', fontsize=6.2, color=INK2, linespacing=1.25, zorder=4)
    # step markers and descriptions
    for k, (title, desc) in enumerate(STEPS):
        x0 = XC[k] + BW / 2; x1 = targets[k][0]; xm = (x0 + x1) / 2
        ax.add_patch(Circle((xm, MID), 1.25, fc='white', ec=INK2, lw=.7, zorder=5))
        ax.text(xm, MID - .05, str(k + 1), ha='center', va='center', fontsize=6.6, fontweight='bold', color=INK, zorder=6)
        ok = k == 3
        ax.text(xm, 6.4, title, ha='center', va='top', fontsize=6.6, fontweight='semibold', color='#2f7d4f' if ok else INK)
        ax.text(xm, 4.0, desc, ha='center', va='top', fontsize=6.2, color=INK2, linespacing=1.3)
        ax.plot([xm, xm], [MID - 1.6, 7.0], color=RULE, lw=.6, ls=(0, (1.5, 1.5)), zorder=0)
    ax.text(0, 32.6, 'd', fontsize=10, fontweight='bold', va='top', color=INK)


if __name__ == '__main__':
    fig = plt.figure(figsize=(7.1, 2.4)); draw(fig.add_axes([0, 0, 1, 1])); save(fig, 'pub_fig2d_test.png')
