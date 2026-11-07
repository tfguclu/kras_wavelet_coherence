"""Kept coherence network with the episode-length distribution of every kept pair drawn on its edge.

Network as in fig_network_kept.py (42 kept pairs: >= 2 replicas, not matched by time-shifted controls).
Each chi or SASA edge carries a small panel with the probability density of its episode lengths
(three replicas pooled; chi episodes of all chi-angle pairs merged into their union per residue pair);
each H-bond hyperedge carries one below its outline. Number = episodes. Densities: Gaussian kernel,
bandwidth BW, x from 0 to XMAX; each curve scaled to its own maximum (shape, not height, is compared).
Inputs (../data): chi_network_edges.csv, sasa_network_edges.csv, hbond_hyperedges.csv (via common),
episodes_all_reps_rotamers.csv, episodes_all_reps_SASA.csv. Writes fig_network_episodes.png/.pdf/.svg.
"""
import os, sys, itertools
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'analysis')]
os.chdir(HERE)
from pub_style import *
from common import load_chi, load_sasa, load_hb, lab
DATA = os.path.join(HERE, '..', 'data')
import numpy as np, networkx as nx, pandas as pd
from matplotlib.patches import FancyArrowPatch, Circle, Polygon, FancyBboxPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.path import Path
from scipy.spatial import ConvexHull

MINREP = 2
SETASIDE = {(100, 112), (85, 97), (158, 167), (100, 161)}   # chi pairs matched by controls (Section 2.1)
REPS = ['namd_3', 'namd_3_2', 'namd_3_3']
BW, XMAX = 200.0, 3500.0                                     # kernel sd (ns); x range of the small panels (ns)
LOC = REG + [('α3', 87, 104), ('α4', 127, 137)]
NC = {n: (REGL[n], REGC[n]) for n, *_ in REG}
NC.update({'α3': ('#e6eef6', '#6f9cc8'), 'α4': ('#f4e7dc', '#b5866a'), 'other': ('white', '#a6a6a6')})
loc = lambda r: next((n for n, a, b in LOC if a <= r <= b), 'other')

# ---------------------------------------------------------------- network
chi, sasa, h = load_chi(), load_sasa(), load_hb()
E = [(r.i, r.j, 'chi', r.n) for r in chi[chi.n >= MINREP].itertuples()
     if (min(r.i, r.j), max(r.i, r.j)) not in SETASIDE] + \
    [(r.i, r.j, 'sasa', r.n) for r in sasa[sasa.n >= MINREP].itertuples()]
HYP = [([tuple(map(int, x.split('-'))) for x in (r.hb1, r.hb2)], int(r.n_reps))
       for r in h[h.n_reps >= MINREP].itertuples()]

# ---------------------------------------------------------------- episodes
parse = lambda x: [tuple(map(float, a.split('-'))) for a in str(x).split(';')] if isinstance(x, str) and x else []
def union(iv):
    out = []
    for s0, e0 in sorted(iv):
        if out and s0 <= out[-1][1]: out[-1][1] = max(out[-1][1], e0)
        else: out.append([s0, e0])
    return out
er = pd.read_csv(os.path.join(DATA, 'episodes_all_reps_rotamers.csv'))
es = pd.read_csv(os.path.join(DATA, 'episodes_all_reps_SASA.csv'))
EP = {}
for i, j, k, _ in E:
    a, b = min(i, j), max(i, j)
    if k == 'chi':
        g = er[((er.resid_i == a) & (er.resid_j == b)) | ((er.resid_i == b) & (er.resid_j == a))]
        L = [e - s for rep in REPS for s, e in union([iv for x in g['episodes_ns_' + rep] for iv in parse(x)])]
    else:
        g = es[((es.resid_i == a) & (es.resid_j == b)) | ((es.resid_i == b) & (es.resid_j == a))]
        L = [e - s for rep in REPS for x in g['episodes_ns_' + rep] for s, e in parse(x)]
    EP[(k, (a, b))] = np.array(L)
for r, (b, _) in zip(h[h.n_reps >= MINREP].itertuples(), HYP):
    EP[('hb', b[0] + b[1])] = np.array([e - s for rep in REPS for s, e in parse(getattr(r, 'episodes_ns_' + rep))])
assert len(EP) == 42 and all(len(v) for v in EP.values())   # all kept pairs

# ---------------------------------------------------------------- layout (data units)
G = nx.Graph(); G.add_edges_from((u, v) for u, v, *_ in E)
for b, _ in HYP: G.add_edges_from(b); G.add_edge(b[0][0], b[1][0])
G.remove_edges_from(list(nx.selfloop_edges(G)))
comps = [frozenset(c) for c in nx.connected_components(G)]
comp = lambda r: next(c for c in comps if r in c)
hypres = [set(b[0]) | set(b[1]) for b, _ in HYP]
PANELS_ALL = [('Switch II–α3 interface', [69, 73, 75, 68]), ('α3 and α3–α4', [88, 101, 80, 94]),
          ('Nucleotide site', [116, 28, 17]), ('α4 and β5–α4 loop', [128, 123]),
          ('P-loop, β1 and β3', [16]), ('N terminus, β2 and β3', [52]),
          ('Core and α5', [161, 53, 51, 19, 76, 149])]
# two figures so that each fits a page at full size: python fig_network_episodes.py --part A | B
PART = sys.argv[sys.argv.index('--part') + 1] if '--part' in sys.argv else 'A'
PARTS = {'A': ['Switch II–α3 interface', 'α3 and α3–α4', 'Nucleotide site', 'α4 and β5–α4 loop'],
         'B': ['P-loop, β1 and β3', 'N terminus, β2 and β3', 'Core and α5']}
assert set().union(*[comp(r) for _, rs in PANELS_ALL for r in rs]) == set(G)
PANELS = [p for p in PANELS_ALL if p[0] in PARTS[PART]]
KEEP = set().union(*[comp(r) for _, rs in PANELS for r in rs])        # residues drawn in this part
E = [e for e in E if e[0] in KEEP]; HYP = [(b, n) for b, n in HYP if b[0][0] in KEEP]
EP = {k: v for k, v in EP.items() if k[1][0] in KEEP}
G = G.subgraph(KEEP).copy(); hypres = [set(b[0]) | set(b[1]) for b, _ in HYP]

R, EMIN = 0.56, 4.2                  # node radius; centre distance of linked nodes (long enough to see the line style)
SW, SH = 1.95, 0.84                  # small panel width, height
PW, PAD, HEAD, GAP = 13.2, 0.6, 1.15, 0.6
X0IN = PAD + 0.6
CSCALE = 1.1                         # custom layouts were drawn for the smaller panels
CUSTOM = {69: {70: (0, 0), 66: (2.0, 0), 65: (4.0, 0), 69: (6.0, 0), 102: (6.0, -3.7), 99: (9.7, -3.7)},
          16: {60: (0, 0), 10: (2.2, -3.0), 16: (4.4, 0), 57: (6.6, -3.0), 20: (8.8, 0), 55: (8.8, 3.9), 8: (4.4, 3.9)},
          128: {128: (0, 0), 132: (4.8, 0), 126: (2.4, -3.9)},
          123: {127: (0, 0), 143: (3.7, 0), 123: (6.4, 0), 125: (10.1, 0), 84: (6.4, 3.4)},
          52: {1: (0, 0), 52: (3.7, 0), 54: (7.4, 0), 3: (9.2, 3.2), 41: (9.2, -3.2)}}
def comp_layout(cc):
    ks = sorted(cc)
    for key, lay in CUSTOM.items():
        if key in cc:
            a = np.array([lay[k] for k in ks], float) * CSCALE; a[:, 1] *= -1; a -= a.min(0); return dict(zip(ks, a))
    if len(cc) == 2: return {ks[0]: np.array([0., 0]), ks[1]: np.array([EMIN, 0])}
    if len(cc) == 3 and nx.is_path_graph(G.subgraph(cc)) if hasattr(nx, 'is_path_graph') else False: pass
    sg = G.subgraph(cc); order = list(nx.dfs_preorder_nodes(sg, [n for n in cc if sg.degree(n) == 1][0]))
    a = np.array([[EMIN * order.index(k), 0.] for k in ks]); return dict(zip(ks, a))   # chains of 3
def build(rs):
    x = X0IN; y = 0; rowh = 0; rowhyp = False; pos = {}
    for r in rs:
        cc = comp(r); lp = comp_layout(cc); bw, bh = np.array(list(lp.values())).max(0)
        hy = any(cc & s for s in hypres)
        if x > X0IN and x + bw > PW - X0IN:
            y += rowh + (3.0 if rowhyp else 2.0); x = X0IN; rowh = 0; rowhyp = False
        for k, a in lp.items(): pos[k] = np.array([x + a[0], -(y + a[1])])
        x += bw + 2.2; rowh = max(rowh, bh); rowhyp |= hy
    sh = [k for k in SHIFT if k in pos]
    if sh:                                                   # move a component up into free space of its panel
        for key in sh:
            for k in comp(key): pos[k] = pos[k] + np.array([SHIFT[key][0], SHIFT[key][1]])
        bot = max(-p[1] for p in pos.values())
        return pos, bot + (2.5 if any(comp(key) & s for key in sh for s in hypres) else 0.7)
    return pos, y + rowh + (1.9 if rowhyp else 0.7)
SHIFT = {84: (0.0, 3.6)}
HEADX = {'α4 and β5–α4 loop': 1.0}             # room above arcs that bow upwards
built = [(t, *build(rs)) for t, rs in PANELS]
best = min((asg for asg in itertools.product((0, 1), repeat=len(built)) if asg[0] == 0),
           key=lambda asg: abs(sum((H + HEAD + HEADX.get(t, 0) + PAD) * (1 if a == 0 else -1) for (t, _, H), a in zip(built, asg))))
pos, boxes, ybot = {}, [], []
for k in (0, 1):
    Y = 0; X0 = k * (PW + GAP)
    for (t, p, H), a in zip(built, best):
        if a != k: continue
        hx = HEADX.get(t, 0); inner = Y - HEAD - hx - PAD * .4
        for n, q in p.items(): pos[n] = np.array([X0 + q[0], inner + q[1] - R - .1])
        hh = HEAD + hx + PAD * .4 + H + R + .1; boxes.append((t, X0, Y, hh)); Y -= hh + .5
    ybot.append(Y)
def hull(pts, r=0.75, n=48):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    P = np.concatenate([p + r * np.c_[np.cos(th), np.sin(th)] for p in pts]); return P[ConvexHull(P).vertices]

# ---------------------------------------------------------------- draw
Wd = 2 * PW + GAP; Hd = -min(ybot) + (0.3 if max(ybot) - min(ybot) > 9.5 else 5.6)
fig = plt.figure(figsize=(7.1, 7.1 * Hd / Wd)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-.05, Wd + .05); ax.set_ylim(-Hd, .05); ax.set_aspect('equal'); ax.axis('off')
DASH = (0, (2.5, 1.6)); X = np.linspace(0, XMAX, 200); SPARK = []; SPINFO = []
def spark(c, L, col, z=6, label=None, nrep=3):
    """small density panel centred at c"""
    x0, y0 = c[0] - SW / 2, c[1] - SH / 2
    ax.add_patch(FancyBboxPatch((x0, y0), SW, SH, boxstyle='round,pad=0,rounding_size=0.12', fc='white', ec=col,
                                lw=.9 if nrep == 3 else .8, ls='-' if nrep == 3 else (0, (2.2, 1.6)), zorder=z))   # border = replicas
    f = np.exp(-.5 * ((X[:, None] - L[None]) / BW) ** 2).sum(1); f /= f.max()
    px0, pw, pb, ph = x0 + .42, SW - .52, y0 + .1, SH - .22
    xx, yy = px0 + X / XMAX * pw, pb + f * ph
    ax.fill_between(xx, pb, yy, color=col, alpha=.25, lw=0, zorder=z + .1); ax.plot(xx, yy, color=col, lw=.8, zorder=z + .2)
    ax.plot([px0, px0 + pw], [pb, pb], color=INK2, lw=.4, zorder=z + .2)
    ax.text(x0 + .22, c[1], label or str(len(L)), ha='center', va='center', fontsize=8.0, color=col, fontweight='bold', zorder=z + .3)
    if (L > XMAX).any(): ax.text(px0 + pw, pb + ph, '›', ha='right', va='top', fontsize=8.5, color=col, zorder=z + .3)
    SPARK.append((x0, y0, SW, SH)); SPINFO.append((label, len(L), nrep, col))
for t, X0, top, hh in boxes:
    ax.add_patch(FancyBboxPatch((X0, top - hh), PW, hh, boxstyle='round,pad=0,rounding_size=0.3', fc=BAND, ec='none', zorder=0))
    ax.text(X0 + .45, top - .52, t, fontsize=9.4, fontweight='semibold', va='center', color=INK)
HULLS = []; NREP_HB = {b[0] + b[1]: n for b, n in HYP}
for b, n in HYP:
    P = hull(np.array([pos[k] for k in sorted(set(b[0]) | set(b[1]))]))
    ax.add_patch(Polygon(P, fc=OBS['hb'], ec='none', alpha=.14, zorder=.5))
    ax.add_patch(Polygon(P, fc='none', ec=OBS['hb'], lw=.9, ls='-' if n == 3 else (0, (4, 2.5)), zorder=.6))
    for u, v in b:
        ax.plot(*zip(pos[u], pos[v]), color=OBS['hb'], lw=2, ls='-' if n == 3 else DASH, zorder=1.2, solid_capstyle='round', dash_capstyle='round')
    HULLS.append((b, P))
pairs = defaultdict(list)
for e in E: pairs[(min(e[:2]), max(e[:2]))].append(e)
hbp = {tuple(sorted(x)) for b, _ in HYP for x in b}
for key, es_ in pairs.items():
    es_ = sorted(es_, key=lambda e: e[2]); off = key in hbp; m = len(es_) + off
    p0, p1 = pos[key[0]], pos[key[1]]
    slots = ([0.] if m == 1 else [-.5, .5]) if not off else ([.75] if m == 2 else [-.75, .75])   # H-bond stays straight
    for (u, v, k, n), sl in zip(es_, slots):
        rad = sl
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-', connectionstyle=f'arc3,rad={rad}', color=OBS[k], lw=2,
                                     ls='-' if n == 3 else DASH, zorder=1, shrinkA=9, shrinkB=9, capstyle='round'))
        d = p1 - p0; perp = np.array([d[1], -d[0]])                  # matplotlib arc3 control-point direction
        mid = (p0 + p1) / 2 + .5 * rad * perp                        # midpoint of the quadratic Bezier
        spark(mid, EP[(k, key)], OBS[k], nrep=n)
def ov(a, b, pad=.04): return not (a[0] + a[2] + pad <= b[0] or b[0] + b[2] + pad <= a[0] or a[1] + a[3] + pad <= b[1] or b[1] + b[3] + pad <= a[1])
NODEBOX = [(p[0] - R, p[1] - R, 2 * R, 2 * R) for p in pos.values()]
for b, P in HULLS:                                                  # H-bond pair panel: first free spot on the outline
    xs = np.array([pos[k][0] for k in set(b[0]) | set(b[1])]); ys = np.array([pos[k][1] for k in set(b[0]) | set(b[1])])
    cands = [(xs.mean(), P[:, 1].min() + .12 - SH / 2), (xs.mean(), P[:, 1].max() - .12 + SH / 2),
             (P[:, 0].max() - .1 + SW / 2, ys.mean()), (P[:, 0].min() + .1 - SW / 2, ys.mean()),
             (xs.min() + SW / 2, P[:, 1].min() + .12 - SH / 2), (xs.max() - SW / 2, P[:, 1].min() + .12 - SH / 2)]
    cen = P.mean(0)                                                 # then a ring of spots just outside the outline
    for th in np.radians(np.arange(-90, 270, 15)):
        dvec = np.array([np.cos(th), np.sin(th)]); q = P[np.argmax(P @ dvec)]
        cands.append((q[0] + dvec[0] * (SW / 2 - .1), q[1] + dvec[1] * (SH / 2 - .1)))
    box = next((t, X0, top, hh) for t, X0, top, hh in boxes if X0 <= xs.mean() <= X0 + PW and top - hh <= ys.mean() <= top)
    def free(c):
        r_ = (c[0] - SW / 2, c[1] - SH / 2, SW, SH)
        inside = box[1] + .1 <= r_[0] and r_[0] + SW <= box[1] + PW - .1 and box[2] - box[3] + .1 <= r_[1] and r_[1] + SH <= box[2] - HEAD + .3
        return inside and not any(ov(r_, q) for q in SPARK + NODEBOX)
    spark(next((c for c in cands if free(c)), cands[0]), EP[('hb', b[0] + b[1])], OBS['hb'], nrep=NREP_HB[b[0] + b[1]])
for k, p in pos.items():
    f, e = NC[loc(k)]
    ax.add_patch(Circle(p, R, fc=f, ec=e, lw=1.1, zorder=8))
    ax.text(*p, lab(k), ha='center', va='center', fontsize=8.0, color=INK, fontweight='medium', zorder=9)
# legend: in the shorter column if it has room, else below both columns
free = max(ybot) - min(ybot); stack = free > 9.5
col_free = int(np.argmax(ybot))
if stack: lx, ly = col_free * (PW + GAP) + .3, ybot[col_free] - .2
else: lx, ly = .3, min(ybot) - .3
h1 = [Line2D([], [], color=OBS['chi'], lw=2, label='χ dihedral'), Line2D([], [], color=OBS['sasa'], lw=2, label='SASA'),
      Line2D([], [], color=OBS['hb'], lw=2, label='H-bond'), Patch(fc=OBS['hb'], alpha=.25, ec=OBS['hb'], lw=.8, label='coupled H-bond pair'),
      Line2D([], [], color=INK2, lw=2, label='3 replicas'), Line2D([], [], color=INK2, lw=2, ls=DASH, label='2 replicas')]
h2 = [Line2D([], [], marker='o', ls='', ms=7, mfc=NC[k][0], mec=NC[k][1], mew=1.1, label=k) for k in NC]
l1 = ax.legend(handles=h1, loc='upper left', bbox_to_anchor=(lx, ly), bbox_transform=ax.transData, ncol=3, fontsize=8.0, handlelength=1.6 if stack else 2,
               columnspacing=.8 if stack else 1.2, handletextpad=.5, title='Coherent pairs (80–120 ns)', title_fontproperties={'weight': 'semibold', 'size': 8.6}, alignment='left')
ax.add_artist(l1)
ax.legend(handles=h2, loc='upper left', bbox_to_anchor=(lx, ly - (3.0 if stack else 3.3)), bbox_transform=ax.transData, ncol=3 if stack else 6, fontsize=8.0,
          handletextpad=.3, columnspacing=1.0, title='Residue location', title_fontproperties={'weight': 'semibold', 'size': 8.6}, alignment='left')
kx, ky = (lx + .7, ly - 5.75) if stack else (16.0, ly - .2)
ax.text(kx - .7, ky, 'Episode lengths on each edge', fontsize=8.6, fontweight='semibold', color=INK, va='top')
demo = np.concatenate([EP[k] for k in EP]); cdemo = (kx + SW / 2 - .3, ky - 1.35); spark(cdemo, demo, INK2, label='n')
ax.text(cdemo[0] - SW / 2 + .42, cdemo[1] - SH / 2 - .08, '0', ha='center', va='top', fontsize=7.6, color=INK2)
ax.text(cdemo[0] + SW / 2 - .1, cdemo[1] - SH / 2 - .08, '3.5 µs', ha='center', va='top', fontsize=7.6, color=INK2)
ax.text(cdemo[0] + SW / 2 + .8, ky - .62, 'n = episodes (3 replicas pooled)\ncurve = density of episode length,\nscaled to its own maximum\n› = one episode longer than 3.5 µs\nborder: solid, 3 replicas; dashed, 2',
        fontsize=7.6, color=INK2, va='top', linespacing=1.3)
SPARK.pop(); SPINFO.pop()   # key panel is not checked

# ---------------------------------------------------------------- checks
bad = [(i, j) for i, j in itertools.combinations(range(len(SPARK)), 2) if ov(SPARK[i], SPARK[j])]
nod = [(i, k) for i, s in enumerate(SPARK) for k, p in pos.items() if ov(s, (p[0] - R, p[1] - R, 2 * R, 2 * R), 0)]
print(f'{len(SPARK)} small panels; panel-panel overlaps: {len(bad)}; panel-node overlaps: {len(nod)}')
for i, j in bad: print('  overlap', np.round(SPARK[i][:2], 1), np.round(SPARK[j][:2], 1))
for i, k in nod: print('  panel', SPARK[i][:2], 'covers node', lab(k))
short = []
for key, es_ in pairs.items():
    d = np.linalg.norm(pos[key[0]] - pos[key[1]]); vis = (d - 2 * R - SW) / 2   # straight-line part left visible on each side
    if len(es_) + (key in hbp) == 1 and vis < .45: short.append((lab(key[0]), lab(key[1]), round(vis, 2)))
print('straight edges with < 0.45 visible on each side of their panel:', short)
nset = {(k, key): n for key, es_ in pairs.items() for (_, _, k, n) in es_}
assert sorted(i[1] for i in SPINFO) == sorted(len(v) for v in EP.values()), 'n mismatch'
print('part', PART, ':', len(SPINFO), 'pairs')
print('replica styles: 3 ->', sum(i[2] == 3 for i in SPINFO), '; 2 ->', sum(i[2] == 2 for i in SPINFO))
save(fig, {'A': 'Figure4.png', 'B': 'FigureS5.png'}[PART])
