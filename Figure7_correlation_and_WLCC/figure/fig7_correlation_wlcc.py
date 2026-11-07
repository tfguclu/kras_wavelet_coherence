import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'analysis')]
os.chdir(HERE)
import sys
from pub_style import *
import numpy as np, pandas as pd, os
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
U = DATA + os.sep
FILES = {'chi': ('pairs_r2_chi.csv', ('resid_i', 'resid_j')), 'sasa': ('pairs_r2_sasa.csv', ('resid_i', 'resid_j')),
         'hb': ('pairs_r2_hb.csv', ('hb1', 'hb2'))}
REPS = ['namd_3', 'namd_3_2', 'namd_3_3']
AS = pd.read_csv(os.path.join(DATA, 'avg_associations.csv'))
SET_ASIDE = {'100-112', '85-97', '158-167', '100-161'}   # chi pairs set aside by the per-pair test (Section 2.1)
CUT = 0.5                                                 # |Pearson r|; for chi, sqrt of the RV coefficient
LG, DG = '#d0d0d0', '#8a8a8a'
rng = np.random.default_rng(0)
def load(o):
    f, key = FILES[o]; d = pd.read_csv(U + f)
    d['k'] = (d[key[0]].astype(str) + ' | ' + d[key[1]].astype(str)) if o == 'hb' else d[key[0]].astype(str) + '-' + d[key[1]].astype(str)
    d['a'] = np.sqrt(d.r2)                                # |r| (chi: sqrt RV)
    return d
plt.rcParams.update({'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5})
fig = plt.figure(figsize=(7.4, 3.9))
row = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.3], wspace=0.5, bottom=0.31, top=0.8, left=0.075, right=0.995)
top = [row[0], row[1], row[2]]
SUM = {}
for c, o in enumerate(['chi', 'sasa', 'hb']):
    d = load(o); g = AS[AS.observable == o]
    rep = set(g[g.replicas >= 2].association) - SET_ASIDE; core = set(g[g.replicas == 3].association) - SET_ASIDE
    above = d[d.a >= CUT]; cnt = above.k.value_counts(); sel = set(cnt[cnt >= 2].index)
    SUM[o] = dict(sel=len(sel), rep=len(rep), rec=len(sel & rep), core=len(core), corec=len(sel & core), only=sorted(sel - rep))
    ax = fig.add_subplot(top[c])
    for ri, r in enumerate(REPS):
        dr = d[d.replica == r]
        show = dr[(dr.a >= CUT) | dr.k.isin(rep)]
        col = np.where(show.k.isin(rep & sel), INK, np.where(show.k.isin(rep), OBS[o], np.where(show.k.isin(sel), DG, LG)))
        z = np.where(show.k.isin(rep), 4, 3)
        x = ri + rng.uniform(-.22, .22, len(show))
        for zz in (3, 4):
            m = z == zz; ax.scatter(x[m], show.a.values[m], s=9, c=col[m], lw=0, zorder=zz)
    ax.axhline(CUT, color=INK2, lw=.7, ls='--', zorder=2)
    ax.set_xticks(range(3)); ax.set_xticklabels(['1', '2', '3'], fontsize=8.5); ax.set_xlabel('replica', fontsize=9)
    ax.set_ylim(0, 1); ax.set_xlim(-.5, 2.5)
    ax.set_ylabel('whole-trajectory √RV' if o == 'chi' else 'whole-trajectory |r|', fontsize=9)
    ax.set_title(f'{len(sel & rep)}/{len(rep)} reproducible\n{len(sel - rep)} correlation only', fontsize=7.8, color=INK2, pad=4, linespacing=1.1)
    ax.text(.5, 1.2, (r'$\chi$ dihedral' if o == 'chi' else OBSN[o]), transform=ax.transAxes, ha='center', va='bottom', fontsize=10, color=OBS[o], fontweight='semibold')
    panel(ax, 'abc'[c], x=-0.38, y=1.2)
# ---- d: WLCC recovery
WL = os.environ.get('WLCC_UNITS', os.path.join(DATA, 'wlcc_broad_units_results.csv'))
u = pd.read_csv(WL)
cu = u[(u.set == 'coherent') & ~u.label.isin(list(SET_ASIDE))]
ax = fig.add_subplot(row[3])
for i, o in enumerate(['chi', 'sasa', 'hb']):
    rp = cu[(cu.obs == o) & (cu.coh_reps >= 2)]; co = cu[(cu.obs == o) & (cu.coh_reps == 3)]
    fr, fc = (rp.reps >= 2).sum(), (co.reps >= 2).sum()
    ax.bar(i - .2, 100 * fr / len(rp), .36, color=OBS[o], alpha=.45, lw=0)
    ax.bar(i + .2, 100 * fc / len(co), .36, color=OBS[o], lw=0)
    ax.text(i - .2, 100 * fr / len(rp) - 3, f'{fr}/{len(rp)}', ha='center', va='top', fontsize=7.6, color=INK, rotation=90)
    ax.text(i + .2, 100 * fc / len(co) - 3, f'{fc}/{len(co)}', ha='center', va='top', fontsize=7.6, color='white', rotation=90)
ax.set_xticks(range(3)); ax.set_xticklabels(['χ', 'SASA', 'H-bond'], fontsize=8.5); ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel('found by WLCC in ≥ 2 replicas (%)', fontsize=8.5)
ax.legend(handles=[Patch(color='#9a9a9a', alpha=.45, label='reproducible'), Patch(color='#5a5a5a', label='core')], loc='upper left', fontsize=8, ncol=2, frameon=False, bbox_to_anchor=(-.02, 1.06), handlelength=1.2, columnspacing=.8)
panel(ax, 'd', x=-0.38, y=1.2)
ax.text(.5, 1.2, 'WLCC', transform=ax.transAxes, ha='center', va='bottom', fontsize=10, color=INK, fontweight='semibold')
# legend for a-c
lx = fig
mk = lambda c_, l_: Line2D([], [], marker='o', ls='', mfc=c_, mec=c_, ms=4, label=l_)
h = [mk(INK, 'reproducible coherent pair, |r| ≥ 0.5 in ≥ 2 replicas'),
     (mk(OBS['chi'], ''), mk(OBS['sasa'], ''), mk(OBS['hb'], '')),
     mk(DG, '|r| ≥ 0.5 in ≥ 2 replicas, not a reproducible coherent pair'),
     mk(LG, '|r| ≥ 0.5 in one replica only'),
     Line2D([], [], color=INK2, lw=.8, ls='--')]
from matplotlib.legend_handler import HandlerTuple
lx.legend(handles=h, labels=[h[0].get_label(), 'reproducible coherent pair, not selected by correlation', h[2].get_label(), h[3].get_label(), 'cutoff, |r| = 0.5'],
          handler_map={tuple: HandlerTuple(ndivide=None, pad=.2)}, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=8.2, frameon=False, handlelength=2.0, columnspacing=1.2)
save(fig, 'Figure7.png')
import json; print(json.dumps({o: {k: v for k, v in s.items()} for o, s in SUM.items()}))
