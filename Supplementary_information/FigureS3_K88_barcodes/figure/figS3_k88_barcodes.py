import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'analysis')]
os.chdir(HERE)
# K88 bond barcodes: E91, D92 side chain, D92 backbone, E62 H-bond, E62 salt bridge (+ distance), per replica
from pub_style import *
import numpy as np, pandas as pd
from matplotlib.colors import ListedColormap
z = np.load(os.path.join(DATA, 'k88_signals.npz'), allow_pickle=True)
F = pd.read_csv(os.path.join(DATA, 'k88_frames.csv'))
REPS = ['namd_3', 'namd_3_2', 'namd_3_3']; OPEN = [6.6, 7.1, 0.78]
SB = 4.0   # salt bridge: K88 Nζ to nearest E62 carboxylate O within 4 Å
ROWS = ['K88 SC–E91 SC', 'K88 SC–D92 SC', 'K88 BB–D92 BB', 'K88 SC–E62 SC', 'Switch II open']
COL = ['#d08c1f', '#7a4d0c', '#9a9a9a', '#2f7d6d', '#8e44ad']
MEAS = {'namd_3': 'namd_3_measures.csv', 'namd_3_2': 'namd_3_2_measures.csv', 'namd_3_3': 'namd_3_3_measures.csv'}
AD = DATA + os.sep
out, cond = [], []
fig = plt.figure(figsize=(7.1, 5.8))
G = fig.add_gridspec(3, 1, hspace=0.55, left=0.2, right=0.97, top=0.94, bottom=0.11)
for k, r in enumerate(REPS):
    f = F[F.replica == r]; t = f.time_ns.astype(int).values; tu = t / 1000
    pair = z[f'88-91|hb|88-91 | 88-92|{r}|raw']
    sig = [pair[0][t], z[f'88-92|hbsplit|E91 | D92sc|{r}|raw'][1][t], z[f'88-92|hbsplit|E91 | D92bb|{r}|raw'][1][t],
           z[f'88-92|hb|62-88 | 88-92|{r}|raw'][0][t]]
    me = pd.read_csv(AD + MEAS[r]); g60 = me.set_index(me.time_ns.round().astype(int)).G60N_gammaO.reindex(t).values
    sig.append((g60 > 3.2).astype(int))
    dist = f.SB_E62_K88.values
    gs = G[k].subgridspec(2, 1, height_ratios=[5, 1.0], hspace=0.06)
    ax = fig.add_subplot(gs[0]); ad = fig.add_subplot(gs[1], sharex=ax)
    for i, s in enumerate(sig):
        cm = ListedColormap(['white', COL[i]])
        ax.imshow(s[None, :], aspect='auto', cmap=cm, vmin=0, vmax=1, interpolation='nearest',
                  extent=[tu[0], tu[-1], i + .5 - .42, i + .5 + .42], origin='lower')
        out.append(dict(replica=k + 1, signal=ROWS[i], occupancy=s.mean()))
    ax.set_ylim(len(sig) + .05, -.05); ax.set_yticks([i + .5 for i in range(len(sig))])
    ax.set_yticklabels(ROWS if True else [], fontsize=6.6)
    for tl, c in zip(ax.get_yticklabels(), COL): tl.set_color(c)
    ax.set_xlim(0, 10); ax.tick_params(axis='y', length=0); ax.tick_params(labelsize=6.8)
    ax.tick_params(axis='x', labelbottom=False, length=0)
    sp = me.set_index(me.time_ns.round().astype(int)).CA_62_95.reindex(t).values
    im = ad.imshow(sp[None, :], aspect='auto', cmap='magma_r', vmin=8, vmax=17, interpolation='nearest', extent=[tu[0], tu[-1], 0, 1])
    excl(ad, 0, tu[0], zorder=0); excl(ad, tu[-1], 10, zorder=0)
    ad.set_yticks([.5]); ad.set_yticklabels(['Switch II–α3 (Å)'], fontsize=6.6); ad.tick_params(axis='y', length=0)
    for s_ in ad.spines.values(): s_.set_visible(False)
    ad.tick_params(labelsize=6.8); ad.set_xlim(0, 10)
    if k == 2: ad.set_xlabel('time (µs)', fontsize=7.4)
    else: ad.tick_params(axis='x', labelbottom=False)
    for s_ in ax.spines.values(): s_.set_visible(False)
    excl(ax, 0, tu[0], zorder=0); excl(ax, tu[-1], 10, zorder=0)
    ax.axvline(OPEN[k], color=INK, lw=.7, ls=(0, (2, 1.5)), zorder=5)
    ax.text(OPEN[k], -0.15, 'Switch I opens', fontsize=6, ha='center', va='bottom', color=INK2)
    ax.text(-0.02, 1.08, f'replica {k + 1}', transform=ax.transAxes, fontsize=7.6, fontweight='semibold', ha='right', va='bottom')
cax = fig.add_axes([0.77, 0.035, 0.2, 0.012]); cb = fig.colorbar(im, cax=cax, orientation='horizontal'); cb.ax.tick_params(labelsize=6.2); cb.set_label('Switch II–α3 distance (Å)', fontsize=6.4)
save(fig, 'FigureS3.png')
pd.DataFrame(out).pivot(index='signal', columns='replica', values='occupancy').round(3).to_csv('k88_barcode_occupancy.csv')
print(pd.DataFrame(out).pivot(index='signal', columns='replica', values='occupancy').round(3))

