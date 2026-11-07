"""Figure S1: time-shifted controls and sensitivity analysis, combined.
a  real vs control network sizes
b  replicas in which each pair is found under each setting change, smoothing and trajectory half
c  controls (of 200) reaching the real number of replicas, per reproducible chi pair, under each control scheme
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'analysis')]
os.chdir(HERE)
import re, os
from pub_style import *
from common import *
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle

chi, sasa, h = load_chi(), load_sasa(), load_hb()
real = {'chi': chi.n.value_counts(), 'sasa': sasa.n.value_counts(), 'hb': h.n_reps.value_counts()}
ns = pd.read_csv(os.path.join(DATA, 'cnull_summary.csv'))
SMOOTH = os.path.join(DATA, 'smooth_items.csv')                 # kras_hmr_smoothing.py items.csv (columns 1.0, 2.0, 3.0)
SMOOTH_CTRL = os.path.join(DATA, 'smooth_per_association.csv')  # kras_hmr_smoothing.py per_association.csv
GREY = ListedColormap(['#ffffff', '#d4d4d4', '#8f8f8f', '#2b2b2b']); GNORM = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4)
RED = LinearSegmentedColormap.from_list('r', ['#f6d9d2', '#a8322d'])
EMPTY = '#f5f5f5'


def cell(ax, x, y, fc, txt, tc, ec='white'):
    ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, fc=fc, ec=ec, lw=.8))
    ax.text(x, y, txt, ha='center', va='center', fontsize=6.3, color=tc)


def strip(ax, rows, x0):
    """observable colour strip left of the heat map"""
    for y, o in enumerate(rows):
        ax.add_patch(Rectangle((x0, y - .45), .3, .9, fc=OBS[o], ec='none', clip_on=False))


def sections(ax, spans, xr, x_end, x_start=-.85):
    for k, (name, a_, b_) in enumerate(spans):
        if k:
            ax.plot([x_start, x_end + .5], [a_ - .5, a_ - .5], color=INK2 if k == 1 else MUTE, lw=.5,
                    ls='-' if k == 1 else (0, (2, 1.5)), clip_on=False)
        ax.text(xr, (a_ + b_) / 2, name, ha='left', va='center', fontsize=6.3, color=INK2, rotation=-90, clip_on=False)


def tidy(ax, n, xpos, labels):
    ax.set_ylim(n - .5, -.5); ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=55, ha='right', rotation_mode='anchor', fontsize=6.4)
    ax.tick_params(which='both', length=0)
    for s in ax.spines.values(): s.set_visible(False)


fig = plt.figure(figsize=(7.1, 6.4))
gs = fig.add_gridspec(2, 2, width_ratios=[0.86, 1.14], height_ratios=[1.0, .82], wspace=.55, hspace=.5)

# ---------------- a: network sizes ----------------
ax = fig.add_subplot(gs[0, 0])
obs = ['chi', 'sasa', 'hb']; w = .36
rng = lambda a_, b_: f'{a_}–{b_}' if b_ > a_ else f'{b_}'
for k, o in enumerate(obs):
    r = [real[o].get(c, 0) for c in (1, 2, 3)]
    s = ns[ns.observable == o]; s1 = (s.n_associations - s.n_ge2_reps).mean(); s23 = s.n_ge2_reps.mean()
    ax.bar(k - w / 2, r[0], w, color=OBS[o], alpha=.35, lw=0); ax.bar(k - w / 2, r[1] + r[2], w, bottom=r[0], color=OBS[o], lw=0)
    ax.bar(k + w / 2, s1, w, color='#c8c8c8', lw=0); ax.bar(k + w / 2, s23, w, bottom=s1, color='#5c5c5c', lw=0)
    ax.text(k - w / 2, sum(r) + 4, f'{r[1] + r[2]}\n({r[2]})', ha='center', fontsize=6.4, color=OBS[o], fontweight='semibold', linespacing=1.0)
    lo, hi = int(s.n_ge2_reps.min()), int(s.n_ge2_reps.max()); l3, h3 = int(s.n_3_reps.min()), int(s.n_3_reps.max())
    ax.text(k + w / 2 + .03, s.n_associations.max() + 5, f'{rng(lo, hi)}\n({rng(l3, h3)})', ha='center', fontsize=6.4, color=INK2, linespacing=1.0)
    m = s.n_associations.mean()
    ax.errorbar(k + w / 2, m, yerr=[[m - s.n_associations.min()], [s.n_associations.max() - m]], color=INK2, lw=.7, capsize=1.5)
ax.set_xticks(range(3)); ax.set_xticklabels([OBSN[o] for o in obs]); ax.set_ylabel('coherent pairs')
ax.set_ylim(0, 215); ax.set_xlim(-.6, 2.6)
ax.legend(handles=[Patch(color='#c8c8c8', label='1 replica'), Patch(color='#5c5c5c', label='≥ 2 replicas')],
          loc='upper right', fontsize=6.4, handlelength=1, bbox_to_anchor=(1.02, 1.0))
ax.text(.98, .80, 'numbers: ≥ 2 replicas\n(all 3 replicas)', transform=ax.transAxes, fontsize=6.3, color=MUTE, va='top', ha='right', linespacing=1.15)
ax.text(0, 1.04, 'real (colour) vs 200 time-shifted controls (grey)', transform=ax.transAxes, fontsize=6.6, color=INK2)
ax.text(-.26, 1.04, 'a', transform=ax.transAxes, fontsize=10, fontweight='bold')

# ---------------- b: replicas under every setting ----------------
AA3 = {'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I', 'Leu': 'L',
       'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P', 'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'}
one = lambda t: re.sub('(' + '|'.join(AA3) + r')(\d+)', lambda m: AA3[m.group(1)] + m.group(2), t)
def parse(s):
    """item -> (observable, short label)"""
    if s.startswith('DMS'):
        o = {'χ': 'chi', 'SASA': 'sasa', 'HB': 'hb'}[s.split()[1]]; p = s.split()[-1].split('–')
        return o, f'{lab(int(p[0]))}–{lab(int(p[1]))}'
    if s.startswith('HB  '):
        a_, b_ = s[4:].split(': '); return 'hb', one(f'{a_}: ' + b_.replace(' / ', ' | '))
    o, rest = s.split('  ', 1)
    return {'χ': 'chi', 'SASA': 'sasa'}[o], one(rest)

it = pd.read_csv(os.path.join(DATA, 'sens_items.csv')).set_index('item')
keep = ['HB  Arg102: Asp69 / Gln99', 'HB  Arg73: Val103 / Asp105', 'HB  Thr144: Asn116 / Gln150', 'HB  Lys88: Glu91 / Asp92',
        'SASA  Asp69–Arg102', 'SASA  Gly75–Lys104', 'SASA  Lys88–Asp92', 'SASA  His94–Leu133', 'SASA  Val8–Lys16', 'SASA  Gly10–Lys16', 'SASA  Cys80–Ile93',
        'χ  Lys128–Asp132', 'χ  Asp126–Lys128', 'χ  Glu3–Asp54', 'χ  Leu52–Asp54', 'χ  Met1–Leu52',
        'DMS SASA  17–32']
P = [parse(s) for s in keep]
M = it.loc[keep, ['base', 'band_lo', 'band_hi', 'thr_lo', 'thr_hi', 'ep_lo', 'ep_hi']].astype(float).values
SM = np.full((len(keep), 2), np.nan)
if os.path.exists(SMOOTH):
    S = pd.read_csv(SMOOTH).set_index('item'); c1, c3 = [c for c in S.columns if float(c) in (1.0, 3.0)]
    for i, k in enumerate(keep):
        if k in S.index: SM[i] = [S.loc[k, c1], S.loc[k, c3]]
B = pd.read_csv(os.path.join(DATA, 'blocks_summary.csv'))
def bkey(p):
    o, rest = p.split(' ', 1)
    return {'chi': 'chi', 'SASA': 'sasa', 'HB': 'hb'}[o], rest.replace('-', '–')
bh = {bkey(r.pair): (r.reps_first_half, r.reps_second_half) for r in B.itertuples()}
HV = np.array([bh.get(p, (np.nan, np.nan)) for p in P], float)
X = np.column_stack([M, SM, HV])

cols = ['baseline', '60–100 ns', '100–140 ns', '0.70', '0.80', '400 ns', '600 ns', '1 × s', '3 × s', 'first', 'second']
groups = [('band', 1, 2), ('threshold', 3, 4), ('episode', 5, 6), ('smoothing', 7, 8), ('half', 9, 10)]
xpos = np.array([0, 1.35, 2.35, 3.7, 4.7, 6.05, 7.05, 8.4, 9.4, 10.75, 11.75])

ax = fig.add_subplot(gs[:, 1])
for (r, c), v in np.ndenumerate(X):
    if np.isnan(v): cell(ax, xpos[c], r, EMPTY, '', MUTE)
    else: cell(ax, xpos[c], r, GREY(GNORM(v)), int(v), 'white' if v >= 2.5 else '#444')
strip(ax, [o for o, _ in P], -.92)
ax.set_xlim(-1.0, xpos[-1] + .6); tidy(ax, len(keep), xpos, cols)
ax.set_yticks(range(len(keep))); ax.set_yticklabels([l for _, l in P], fontsize=6.4)
ax.tick_params(axis='y', pad=2)
sections(ax, [('core', 0, 15), ('DMS', 16, 16)], xpos[-1] + .75, xpos[-1])
for g, a_, b_ in groups:
    ax.text((xpos[a_] + xpos[b_]) / 2, -1.05, g, ha='center', va='bottom', fontsize=6.4, color=INK2)
    ax.plot([xpos[a_] - .45, xpos[b_] + .45], [-.85, -.85], color=MUTE, lw=.5, clip_on=False)
ax.text(0, 1.06, 'replicas (of 3) in which the pair is found', transform=ax.transAxes, fontsize=6.6, color=INK2)
ax.text(-.38, 1.06, 'b', transform=ax.transAxes, fontsize=10, fontweight='bold')

# ---------------- c: controls reaching the real count, chi ----------------
order = ['128-132', '126-128', '3-54', '52-54', '1-52', '6-53', '41-54', '84-123', '127-143', '99-102',
         '100-112', '100-161', '85-97', '158-167']
def get(f, col):
    d = pd.read_csv(f); d = d[d.observable == 'chi'] if 'observable' in d else d
    return d.set_index('association')[col].to_dict()
C = [get(os.path.join(DATA, 'ctrl_per_assoc.csv'), 'sets_at_or_above_real'),
     get(os.path.join(DATA, 'ctrl_offset_per_assoc.csv'), 'sets_at_or_above_real_filtered'),
     get(os.path.join(DATA, 'ctrl_seam_per_assoc.csv'), 'controls_at_or_above_real_masked'),
     get(os.path.join(DATA, 'ctrl_reswise_per_assoc.csv'), 'controls_at_or_above_real')]
if os.path.exists(SMOOTH_CTRL):
    d = pd.read_csv(SMOOTH_CTRL); d = d[d.observable == 'chi']
    C += [d[d.tsmooth == t].set_index('association').controls_at_or_above_real.to_dict() for t in (1.0, 3.0)]
else:
    C += [{}, {}]
Y = np.array([[c.get(a, np.nan) for c in C] for a in order], float)
# SASA and H-bond: number of reproducible pairs (of 25 and 7) reached by any control; per residue applies to chi only
def reached(f, col, o):
    d = pd.read_csv(f); d = d[(d.observable == o) & (d.real_replicas >= 2)]
    return float((d[col] > 0).sum()), len(d)
SUM = []
for o in ['sasa', 'hb']:
    row = [reached(os.path.join(DATA, 'ctrl_per_assoc.csv'), 'sets_at_or_above_real', o),
           reached(os.path.join(DATA, 'ctrl_offset_per_assoc.csv'), 'sets_at_or_above_real_filtered', o),
           reached(os.path.join(DATA, 'ctrl_seam_per_assoc.csv'), 'controls_at_or_above_real_masked', o)]
    row = row[:3] + [(np.nan, 0)]
    if os.path.exists(SMOOTH_CTRL):
        d = pd.read_csv(SMOOTH_CTRL); d = d[(d.observable == o) & (d.real_replicas >= 2)]
        row += [(float((d[d.tsmooth == t].controls_at_or_above_real > 0).sum()), int((d.tsmooth == t).sum())) for t in (1.0, 3.0)]
    else:
        row += [(np.nan, 0), (np.nan, 0)]
    SUM.append((o, row))
Y = np.vstack([Y] + [[v for v, _ in r] for _, r in SUM])
TXT = {(len(order) + k, c): f'{int(v)}/{n}' for k, (_, r) in enumerate(SUM) for c, (v, n) in enumerate(r) if n}
rowobs = ['chi'] * len(order) + [o for o, _ in SUM]
NA = {(len(order) + k, 3) for k in range(len(SUM))}          # per residue: not applicable
ccols = ['baseline', '≥ 1 µs apart', 'seam masked', 'per residue', '1 × s', '3 × s']
cpos = np.array([0, 1, 2, 3, 4.35, 5.35])
ax = fig.add_subplot(gs[1, 0])
for (r, c), v in np.ndenumerate(Y):
    if (r, c) in NA: cell(ax, cpos[c], r, '#ffffff', '–', '#b0b0b0', ec='#ebebeb')
    elif np.isnan(v): cell(ax, cpos[c], r, EMPTY, '·' if c >= 4 else '', '#9a9a9a')
    elif v == 0: cell(ax, cpos[c], r, '#ffffff', TXT.get((r, c), '0'), '#9a9a9a', ec='#ebebeb')
    else: cell(ax, cpos[c], r, RED(min(np.log(v) / np.log(13), 1)), TXT.get((r, c), int(v)), 'white' if v >= 6 else '#444')
strip(ax, rowobs, -.92)
ax.set_xlim(-1.0, cpos[-1] + .6); tidy(ax, len(Y), cpos, ccols)
ax.set_yticks(range(len(Y)))
ax.set_yticklabels([f'{lab(int(a.split("-")[0]))}–{lab(int(a.split("-")[1]))}' for a in order]
                   + ['SASA, all' if o == 'sasa' else 'H-bond, all' for o, _ in SUM], fontsize=6.4)
ax.tick_params(axis='y', pad=2)
sections(ax, [('core', 0, 4), ('reproducible', 5, 9), ('set aside', 10, 13), ('', 14, 15)], cpos[-1] + .75, cpos[-1], -.92)
ax.plot([-.92, cpos[-1] + .5], [13.5, 13.5], color=INK2, lw=.6, clip_on=False)
ax.text((cpos[4] + cpos[5]) / 2, -1.05, 'smoothing', ha='center', va='bottom', fontsize=6.4, color=INK2)
ax.plot([cpos[4] - .45, cpos[5] + .45], [-.85, -.85], color=MUTE, lw=.5, clip_on=False)
ax.text(0, 1.12, 'controls (of 200) reaching the real replicas', transform=ax.transAxes, fontsize=6.6, color=INK2)
ax.text(-.36, 1.12, 'c', transform=ax.transAxes, fontsize=10, fontweight='bold')

save(fig, 'FigureS2.png')
