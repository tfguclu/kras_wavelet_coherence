"""Figure S6: per-signal properties with HMR (4 fs) vs without HMR (2 fs), first 1.5 us of three replicas each.
Row a: level of each signal; row b: autocorrelation at 10 ns. One point per signal, mean over the three replicas
of each protocol. Input: ../data/per_signal_replica.csv; writes hmr_corr_summary.csv"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'analysis')]
os.chdir(HERE)
import os, sys
import numpy as np, pandas as pd
from pub_style import *
plt.rcParams.update({'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8})
P = pd.read_csv(os.path.join(DATA, 'per_signal_replica.csv'))
OB = ['chi', 'sasa', 'hb']
keep = set()
for o, d in P.groupby('obs'):
    g = d.groupby('key')
    ok = (g['sd'].min() >= 1.0) if o == 'sasa' else \
         (g['level'].mean().between(.05, .95) & (g['rep'].nunique() == 6)) if o == 'hb' else (g['level'].min() < .98)
    keep |= {(o, k) for k in ok[ok].index}
P = P[[(o, k) in keep for o, k in zip(P.obs, P.key)]]
HR = ['namd_3', 'namd_3_2', 'namd_3_3']; VR = ['vanilla_1', 'vanilla_2', 'vanilla_3']
LEVEL = {'chi': ('angle concentration', (0, 1), [0, .5, 1]), 'sasa': ('mean SASA (Å²)', (0, 200), [0, 100, 200]),
         'hb': ('occupancy', (0, 1), [0, .5, 1])}
def rr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1]
fig, axs = plt.subplots(2, 3, figsize=(7.1, 5.0), gridspec_kw=dict(hspace=.5, wspace=.38, top=.9, bottom=.1, left=.13, right=.98))
rows = []
for j, metric in enumerate(['level', 'acf10']):
    for i, o in enumerate(OB):
        piv = P[P.obs == o].pivot_table(index='key', columns='rep', values=metric)
        h, v = piv[HR].mean(axis=1).values, piv[VR].mean(axis=1).values
        r = rr(h, v)
        same = np.mean([rr(piv[a].values, piv[b].values) for g in (HR, VR) for k, a in enumerate(g) for b in g[k + 1:]])
        acr = np.mean([rr(piv[a].values, piv[b].values) for a in HR for b in VR])
        rows.append(dict(obs=o, metric=metric, n=len(piv), r_protocol_means=round(r, 3),
                         r_replicas_same_protocol=round(same, 3), r_replicas_across_protocols=round(acr, 3)))
        ax = axs[j, i]
        name, lim, ticks = LEVEL[o] if metric == 'level' else ('autocorrelation at 10 ns', (-.1, 1), [0, .5, 1])
        ax.plot(lim, lim, color=MUTE, lw=.8, ls=(0, (4, 2)), zorder=1)
        ax.scatter(h, v, s=11, color=OBS[o], alpha=.6, lw=.3, edgecolor='white', zorder=2, rasterized=True)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal'); ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_title(name, fontsize=8.5, color=INK2, pad=4)
        ax.text(.05, .95, f'r = {r:.2f}', transform=ax.transAxes, va='top', fontsize=9.5, fontweight='bold', color=OBS[o])
        ax.text(.05, .82, f'replicas\nsame {same:.2f}\nacross {acr:.2f}', transform=ax.transAxes, va='top',
                fontsize=6.9, color=INK2, linespacing=1.25)
        ax.set_xlabel('with HMR, 4 fs')
        if i == 0: ax.set_ylabel('without HMR, 2 fs')
        if j == 0: fig.text(ax.get_position().x0 + ax.get_position().width / 2, .975, OBSN[o], ha='center', va='top',
                            fontsize=10, fontweight='bold', color=OBS[o])
    pos = axs[j, 0].get_position()
    fig.text(.012, pos.y0 + pos.height / 2, ['signal level', 'signal dynamics'][j], rotation=90, ha='left', va='center',
             fontsize=9.5, fontweight='bold', color=INK)
    fig.text(.012, pos.y1 + .045, 'ab'[j], fontsize=11, fontweight='bold', color=INK, va='bottom')
save(fig, 'FigureS6.png')
pd.DataFrame(rows).to_csv('hmr_corr_summary.csv', index=False); print(pd.DataFrame(rows).to_string(index=False))
