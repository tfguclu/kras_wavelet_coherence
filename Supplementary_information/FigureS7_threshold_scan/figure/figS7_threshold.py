import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'analysis')]
os.chdir(HERE)
from pub_style import *
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
A = DATA + os.sep
S = pd.read_csv(A + 'threshold_scan.csv'); P = pd.read_csv(A + 'pair_rates.csv')   # 200 controls
q75 = P[(P.threshold == 0.75) & (P.set > 0)]
print(q75.groupby('observable').pairs_with_episode.agg(['min', 'max', 'mean']))
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.45), gridspec_kw=dict(wspace=0.42))
for o in ['chi', 'sasa', 'hb']:
    s = S[S.observable == o].sort_values('threshold'); t = s.threshold.values
    for a_, r, lo, hi, m in [(ax[0], 'real_ge2', 'surr_ge2_min', 'surr_ge2_max', 'surr_ge2_mean'),
                             (ax[1], 'real_3', 'surr_r3_min', 'surr_r3_max', 'surr_r3_mean')]:
        a_.fill_between(t, s[lo], s[hi], color=OBS[o], alpha=.18, lw=0)
        a_.plot(t, s[m], ':', color=OBS[o], lw=1)
        a_.plot(t, s[r], '-o', color=OBS[o], ms=3, lw=1.1)
    # c: per-pair episode fraction, mean over replicas and sets; spread over sets x replicas
    q = P[(P.observable == o) & (P.set > 0)].groupby('threshold').rate
    lo_, hi_, mu = q.min(), q.max(), q.mean()
    mu_ = mu.values.copy(); z = mu_ <= 0; mu_[z] = np.nan
    ax[2].plot(t, mu_, ':', color=OBS[o], lw=1)
    ax[2].plot(t[z], np.full(z.sum(), 1.3e-6), 'v', color=OBS[o], ms=3.2, mfc='white', clip_on=False)
    ax[2].plot(t, s.real_pair_rate, '-o', color=OBS[o], ms=3, lw=1.1)
for a_ in ax[:2]:
    a_.set_yscale('symlog', linthresh=1, linscale=.5); a_.set_ylim(0, 600)
    a_.set_yticks([0, 1, 10, 100]); a_.set_yticklabels(['0', '1', '10', '100'])
ax[0].set_ylabel('coherent pairs in ≥ 2 replicas'); ax[1].set_ylabel('coherent pairs in 3 replicas')
ax[2].set_yscale('log'); ax[2].set_ylim(1e-6, 5e-2)
ax[2].set_ylabel('signal pairs with ≥ 1 episode (fraction)')
for a_, l in zip(ax, 'abc'):
    a_.axvline(0.75, color=MUTE, lw=.6, ls='--', zorder=0); a_.set_xlabel('coherence threshold')
    a_.set_xticks([.6, .7, .8, .9]); a_.set_xlim(.585, .915); panel(a_, l, x=-0.3, y=1.03)
h = [Line2D([], [], color=OBS[o], lw=2.5, label=OBSN[o]) for o in ['chi', 'sasa', 'hb']]
h += [Line2D([], [], color=INK2, marker='o', ms=3, lw=1.1, label='real'),
      Line2D([], [], color=INK2, ls=':', lw=1, label='control mean'),
      Line2D([], [], color=INK2, alpha=.25, lw=5, label='control range (200 controls)'),
      Line2D([], [], color=INK2, marker='v', mfc='white', ls='', ms=3.2, label='control = 0')]
fig.legend(handles=h, loc='lower center', bbox_to_anchor=(.5, -.13), ncol=7, fontsize=6, handlelength=1.6, columnspacing=1)
save(fig, 'FigureS7.png')
