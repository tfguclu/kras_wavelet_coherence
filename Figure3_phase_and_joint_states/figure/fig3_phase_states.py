import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'analysis')]
os.chdir(HERE)
from pub_style import *
from common import lab
import json, numpy as np
from matplotlib.lines import Line2D
rows = json.load(open(os.path.join(DATA, 'phase_table.json')))[1:]
def name(obs, a):
    if obs == 'H-bond':
        return ' · '.join('–'.join(lab(int(r)) for r in b.split('–')) for b in a.split(' / '))
    return a
def num(s): return np.nan if s == '–' else float(s.replace('−', '-'))
wrap = lambda p: p + 360 if p < -90 else p          # plot from −90° to 270°, so antiphase sits at 180° without wrapping
fig = plt.figure(figsize=(7.1, 7.75))
G = fig.add_gridspec(2, 1, height_ratios=[5.4, 1.95], hspace=.4, left=0.08, right=0.97, top=0.955, bottom=0.07)
gs = G[0].subgridspec(1, 2, width_ratios=[1.45, 1], wspace=0.55)
ax = fig.add_subplot(gs[0])
ax.set_position(ax.get_position())
n = len(rows); mk = ['o', 's', '^']
ax.axvspan(-45, 45, color='#f1efe9', zorder=0); ax.axvspan(135, 225, color='#e8eef4', zorder=0)
for x0 in (0, 180): ax.axvline(x0, color='#c9c9c9', lw=.5, zorder=1)
nh = sum(r[0] == 'H-bond' for r in rows)
for k, r in enumerate(rows):
    y = n - 1 - k; obs = r[0]; c = OBS['hb' if obs == 'H-bond' else 'sasa']
    ph = [num(v) for v in r[4].split('/')]
    ax.plot([-90, 270], [y, y], color='#ececec', lw=.4, zorder=1)
    jit = [-.17, 0, .17]
    for i, p in enumerate(ph):
        if not np.isnan(p):
            ax.plot(wrap(p), y + jit[i], mk[i], color=c, ms=3.6, mfc='white' if i == 1 else c, mew=.9, zorder=3, clip_on=False)
ax.set_yticks(range(n)); ax.set_yticklabels([name(r[0], r[1]) for r in rows][::-1], fontsize=7.0)
for t, r in zip(ax.get_yticklabels(), rows[::-1]): t.set_color(OBS['hb' if r[0] == 'H-bond' else 'sasa'])
ax.set_ylim(-.7, n - .3); ax.set_xlim(-90, 270); ax.set_xticks([-90, -45, 0, 45, 90, 135, 180, 225, 270])
ax.set_xlabel('relative phase within coherent episodes (°)', fontsize=7.8); ax.tick_params(axis='x', labelsize=7.0)
ax.axhline(n - nh - .5, color=INK2, lw=.6)
ax.text(0, n - .1, 'in phase', ha='center', va='bottom', fontsize=7.2, color=INK2)
ax.text(180, n - .1, 'antiphase', ha='center', va='bottom', fontsize=7.2, color=INK2)
ax.text(268, n - nh / 2 - .5, 'H-bond pairs', ha='right', va='center', fontsize=7.2, color=OBS['hb'], rotation=90)
ax.text(268, (n - nh) / 2 - .5, 'SASA pairs', ha='right', va='center', fontsize=7.2, color=OBS['sasa'], rotation=90)
h = [Line2D([], [], color=INK2, marker=m, mfc=('white' if i == 1 else INK2), ls='', ms=3.6, label=f'replica {i+1}') for i, m in enumerate(mk)]
ax.legend(handles=h, loc='upper center', bbox_to_anchor=(.5, -.07), ncol=3, fontsize=7.2, handlelength=1, columnspacing=1.2, frameon=False)


# ---------------- b: joint states of the two bonds of each core H-bond pair
import pandas as pd
from matplotlib.patches import Patch
J = pd.read_csv(os.path.join(DATA, 'hbond_joint_states_summary.csv'))
J = J[J.window == 'all']
PAIRS = [('73-103|73-105', 'R73', 'V103', 'D105'), ('69-102|99-102', 'R102', 'D69', 'Q99'),
         ('88-91|88-92', 'K88', 'E91', 'D92'), ('116-144|144-150', 'T144', 'N116', 'Q150')]
SC = [('p11', '#7a4d0c', 'both'), ('p10', '#d08c1f', 'first only'), ('p01', '#f0cf98', 'second only'), ('p00', '#ececec', 'neither')]
bx = fig.add_subplot(gs[1])
y, yt, yl = 0, [], []
for hk, c, a1, a2 in PAIRS:
    for k, rep in enumerate(['namd_3', 'namd_3_2', 'namd_3_3']):
        r = J[(J.hyperedge == hk) & (J.rep == rep)].iloc[0]; x0 = 0
        for col, colr, _ in SC:
            bx.barh(y, r[col], left=x0, height=.78, color=colr, edgecolor='white', lw=.4); x0 += r[col]
        yt.append(y); yl.append(f'rep. {k + 1}'); y += 1
    bx.text(-0.02, y - 3 - .75, f'{c}: {a1} | {a2}', ha='right', va='bottom', fontsize=7.9, color=OBS['hb'], fontweight='semibold',
            transform=bx.get_yaxis_transform())
    y += 1.0
bx.set_yticks(yt); bx.set_yticklabels(yl, fontsize=7.4); bx.invert_yaxis()
bx.set_xlim(0, 1); bx.set_xticks([0, .5, 1]); bx.set_xticklabels(['0', '0.5', '1'])
bx.set_xlabel('fraction of frames (whole trajectory)', fontsize=7.8); bx.tick_params(axis='x', labelsize=7.0)
bx.spines['left'].set_visible(False); bx.tick_params(axis='y', length=0)
bx.legend(handles=[Patch(color=cl, label=lb) for _, cl, lb in SC], loc='upper center', bbox_to_anchor=(.45, -.07), ncol=2,
          fontsize=7.2, handlelength=1, columnspacing=1.0, frameon=False)


# ---------------- c: observed vs expected co-formation, by Switch I state
import sys
from matplotlib.lines import Line2D
C = pd.read_csv(os.path.join(DATA, 'core_hbonds_by_switch1.csv'))
REPS = ['namd_3', 'namd_3_2', 'namd_3_3']
PB = [('R73 (V103 | D105)', 'R73: V103 | D105'), ('R102 (D69 | Q99)', 'R102: D69 | Q99'),
      ('K88 (E91 | D92)', 'K88: E91 | D92'), ('T144 (N116 | Q150)', 'T144: N116 | Q150')]
OBC, EXC = '#7a4d0c', '#dcdcdc'
subc = G[1].subgridspec(1, 4, wspace=.42)
for p, (key, tit) in enumerate(PB):
    cx = fig.add_subplot(subc[0, p])
    if p == 0:
        CX0 = cx
    cx.set_title(tit, fontsize=8.2, pad=6, color=OBS['hb'], fontweight='semibold')
    top = 0; ratios = []
    for s_, st in enumerate(['T35 on', 'T35 off']):
        obs, exp_ = [], []
        for j_, r in enumerate(REPS):
            q = C[(C.replica == r) & (C.state == st) & (C.pair == key)]
            if not len(q) or q.frames.iloc[0] < 200:
                continue
            obs.append((j_, 100 * q.both.iloc[0])); exp_.append((j_, 100 * q.occ_first.iloc[0] * q.occ_second.iloc[0]))
        ms_ = []
        for vals, fc, dx in [(obs, OBC, -.2), (exp_, EXC, .2)]:
            x = s_ + dx; m = np.mean([v for _, v in vals]); ms_.append(m); top = max(top, max(v for _, v in vals))
            cx.bar(x, m, width=.38, color=fc, ec='none', zorder=2)
            for j_, v in vals:
                cx.plot(x + (j_ - 1) * .1, v, mk[j_], ms=3.2, mfc='white' if j_ == 1 else INK, mec=INK, mew=.6, zorder=3)
        ratios.append((s_, ms_[0] / ms_[1]))
    ylim = top * 1.12
    cx.set_xticks([0, 1]); cx.set_xticklabels(['T35 on\nMg²⁺', 'T35 off\nMg²⁺'], fontsize=7.4)
    cx.set_xlim(-.55, 1.55); cx.set_ylim(0, ylim)
    cx.yaxis.set_major_locator(plt.MaxNLocator(4, integer=True)); cx.tick_params(axis='x', length=0); cx.tick_params(axis='y', labelsize=7.2)
    cx.grid(axis='y', color='#eeeeee', lw=.6, zorder=0); cx.set_axisbelow(True)
    if p == 0:
        cx.set_ylabel('frames with both\nbonds formed (%)', fontsize=7.8)
h = [Patch(color=OBC, label='observed'), Patch(color=EXC, label='expected for independent bonds')]
h += [Line2D([], [], ls='', marker=mk[j_], ms=3.4, mfc='white' if j_ == 1 else INK, mec=INK, label=f'replica {j_ + 1}') for j_ in range(3)]
fig.legend(handles=h, loc='upper center', bbox_to_anchor=(.53, cx.get_position().y0 - .04), ncol=5, fontsize=7.2,
           handlelength=1.2, columnspacing=1.2, frameon=False)

# ---------------- d: schematic of the two patterns (drawn, not data)
LX = 0.005
def L(x, y, t): fig.text(x, y, t, fontsize=12, fontweight='bold', color=INK, ha='left', va='bottom')
ytop = ax.get_position().y1 + 0.022
L(LX, ytop, 'a'); L(0.585, ytop, 'b')
L(LX, CX0.get_position().y1 + 0.022, 'c')
save(fig, 'Figure3.png')
