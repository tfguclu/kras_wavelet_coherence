#!/usr/bin/env python
"""
kras_hmr_hbond_states.py -- joint occupancy of coupled H-bond pairs (hyperedges).

For each selected hyperedge (two H-bonds, A = res1-res2 and B = res3-res4) and
each replica, the script recomputes atom-level H-bonds with the same criteria
as kras_hmr_hbond.py and reports:

  1. joint states per frame: 00 neither, 10 A only, 01 B only, 11 both
     (fractions overall, inside the coherent episodes, and outside them)
  2. co-occurrence: observed P(11) vs P(A)P(B) (ratio), phi coefficient,
     P(B|A) and P(B|not A)
  3. transitions between states (4 x 4 counts) and "swaps": an A-only run
     followed by a B-only run (or the reverse), directly or through <= --gap-ns
     of 00, with no 11 in between
  4. mean dwell time in each state
  5. the donor/acceptor atoms behind each H-bond, backbone vs side chain, and
     for the shared residue (3-residue hyperedges): in 11 frames, whether both
     bonds use the same atom of the shared residue (bifurcated) or different atoms

Input : wco_hmr/{rep}_wrapped.pdb + .dcd   (from kras_hmr_prep.py)
        HB_.../hyperedges.csv               (from kras_hmr_hbond_wco.py)
        optional wco_hmr/{rep}_hbond.npz    (consistency check of residue-pair signals)
Output: <outdir>/   (default wco_hmr/HB_.../states/)
        <outdir>/states_summary.csv      one row per hyperedge x replica x window
        <outdir>/transitions.csv         4 x 4 transition counts per hyperedge x replica x window
        <outdir>/atom_pairs.csv          donor/acceptor atoms per H-bond and replica
        <outdir>/shared_atom_use.csv     shared-residue atom use in 11 frames
        <outdir>/states_<hb1>_<hb2>.png  state raster per replica with coherent episodes
        <outdir>/params.json

The label column is a heuristic summary with explicit thresholds (see --help);
use the numbers, not the label, in the paper.

  python kras_hmr_hbond_states.py                                     # finds wco_hmr/HB_*/hyperedges.csv; all 3-replica hyperedges
  python kras_hmr_hbond_states.py --hyperedges ... --pairs 69-102,99-102 88-91,88-92
  python kras_hmr_hbond_states.py --selftest
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

DONORS = "(protein or resname GTP) and (name N* or name O* or name SG)"
HYDROGENS = "(protein or resname GTP) and name H*"
ACCEPTORS = ("(protein or resname GTP) and (name O* or (resname HSD and name NE2) "
             "or (resname HSE and name ND1) or (resname GTP and name N3 N7))")
BACKBONE = {"N", "O", "OT1", "OT2", "OXT", "HN", "H"}
STATES = ["00", "10", "01", "11"]          # code = 2*A + B -> index: 0=00, 1=01, 2=10, 3=11
CODE2LAB = {0: "00", 1: "01", 2: "10", 3: "11"}


# ----------------------------------------------------------------------------- helpers
def parse_pair(s):
    a, b = (int(x) for x in s.split("-"))
    return (min(a, b), max(a, b))


def parse_episodes(s):
    """'730-1330;1470-3820' -> [(730,1330),(1470,3820)]; NaN/'' -> []"""
    if not isinstance(s, str) or not s.strip():
        return []
    out = []
    for tok in s.split(";"):
        a, b = tok.split("-")
        out.append((float(a), float(b)))
    return out


def runs(x):
    """run-length encoding of an int array -> (values, starts, lengths)"""
    x = np.asarray(x)
    if x.size == 0:
        return x, np.array([], int), np.array([], int)
    edges = np.flatnonzero(np.diff(x)) + 1
    starts = np.r_[0, edges]
    lengths = np.diff(np.r_[starts, x.size])
    return x[starts], starts, lengths


def count_swaps(code, gap):
    """A-only (10, code 2) run followed by B-only (01, code 1) run or the reverse,
    separated by nothing or by a 00 run of <= gap frames; an 11 in between breaks it."""
    v, s, l = runs(code)
    a2b = b2a = 0
    for k in range(len(v) - 1):
        if v[k] not in (1, 2):
            continue
        j = k + 1
        if v[j] == 0 and l[j] <= gap and j + 1 < len(v):
            j += 1
        if v[k] == 2 and v[j] == 1:
            a2b += 1
        elif v[k] == 1 and v[j] == 2:
            b2a += 1
    return a2b, b2a


def state_metrics(A, B, gap):
    A = A.astype(bool); B = B.astype(bool)
    n = A.size
    if n == 0:
        return None, None
    code = 2 * A.astype(int) + B.astype(int)
    p = np.bincount(code, minlength=4) / n                 # index 0=00,1=01,2=10,3=11
    pA, pB = A.mean(), B.mean()
    exp11 = pA * pB
    den = np.sqrt(pA * (1 - pA) * pB * (1 - pB))
    phi = (p[3] - exp11) / den if den > 0 else np.nan
    v, s, l = runs(code)
    dwell = {CODE2LAB[c]: (l[v == c].mean() if np.any(v == c) else 0.0) for c in range(4)}
    T = np.zeros((4, 4), int)
    np.add.at(T, (code[:-1], code[1:]), 1)
    a2b, b2a = count_swaps(code, gap)
    any_ = 1 - p[0]
    m = dict(n_frames=n, p00=p[0], p10=p[2], p01=p[1], p11=p[3], pA=pA, pB=pB,
             exp11_indep=exp11, ratio11_obs_exp=(p[3] / exp11 if exp11 > 0 else np.nan),
             phi=phi, pB_given_A=(p[3] / pA if pA > 0 else np.nan),
             pB_given_notA=(p[1] / (1 - pA) if pA < 1 else np.nan),
             exclusivity=((p[1] + p[2]) / any_ if any_ > 0 else np.nan),
             swaps_A_to_B=a2b, swaps_B_to_A=b2a,
             dwell00_ns=dwell["00"], dwell10_ns=dwell["10"], dwell01_ns=dwell["01"], dwell11_ns=dwell["11"],
             n_runs_11=int(np.sum(v == 3)))
    return m, T


def label(m, a):
    """heuristic label; thresholds from args"""
    if m is None or m["n_frames"] == 0:
        return "no data"
    if m["p11"] >= a.min_p11 and m["ratio11_obs_exp"] >= a.min_ratio:
        return "co-occupied (both bonds together)"
    if m["p11"] <= a.max_p11_excl and (m["swaps_A_to_B"] + m["swaps_B_to_A"]) >= a.min_swaps:
        return "exclusive with partner exchange"
    if m["p11"] <= a.max_p11_excl:
        return "exclusive, few direct swaps"
    return "mixed"


# ----------------------------------------------------------------------------- per replica
def analyse_rep(rep, edges_sel, a):
    import MDAnalysis as mda
    from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
    t0 = time.time()
    u = mda.Universe(os.path.join(a.indir, f"{rep}_wrapped.pdb"), os.path.join(a.indir, f"{rep}_wrapped.dcd"))
    res = sorted({r for e in edges_sel for p in (e["A"], e["B"]) for r in p})
    rsel = "resid " + " ".join(map(str, res))
    hba = HydrogenBondAnalysis(universe=u, donors_sel=f"{DONORS} and {rsel}",
                               hydrogens_sel=f"{HYDROGENS} and {rsel}",
                               acceptors_sel=f"{ACCEPTORS} and {rsel}",
                               d_h_cutoff=1.2, d_a_cutoff=a.d_a, d_h_a_angle_cutoff=a.angle,
                               update_selections=False)
    hba.run(step=a.stride)
    hb = hba.results.hbonds
    nF = len(range(0, u.trajectory.n_frames, a.stride))
    t = np.arange(nF) * a.dt_ns * a.stride
    frames_run = np.arange(0, u.trajectory.n_frames, a.stride)
    fr = np.searchsorted(frames_run, hb[:, 0].astype(int)) if len(hb) else np.array([], int)
    d_idx = hb[:, 1].astype(int) if len(hb) else np.array([], int)
    h_idx = hb[:, 2].astype(int) if len(hb) else np.array([], int)     # donating hydrogen
    a_idx = hb[:, 3].astype(int) if len(hb) else np.array([], int)
    rd = u.atoms.resids[d_idx]; ra = u.atoms.resids[a_idx]
    lo, hi = np.minimum(rd, ra), np.maximum(rd, ra)
    atoms = u.atoms
    print(f"[{rep}] {nF} frames, {len(hb)} H-bonds among {len(res)} residues, {time.time() - t0:.0f} s", flush=True)

    # residue-pair signals and atom-pair detail for every bond we need
    need = sorted({p for e in edges_sel for p in (e["A"], e["B"])})
    sig, atom_rows, bond_atoms, bond_h = {}, [], {}, {}
    for p in need:
        m = (lo == p[0]) & (hi == p[1])
        s = np.zeros(nF, np.uint8); s[fr[m]] = 1
        sig[p] = s
        per_frame, per_frame_h = {}, {}
        df = pd.DataFrame(dict(frame=fr[m], d=d_idx[m], h=h_idx[m], a=a_idx[m]))
        for (d, h, ac), g in df.groupby(["d", "h", "a"]):
            nfr = g.frame.nunique()
            atom_rows.append(dict(rep=rep, bond=f"{p[0]}-{p[1]}",
                                  donor=f"{atoms[d].resname}{atoms[d].resid}:{atoms[d].name}",
                                  hydrogen=atoms[h].name,
                                  acceptor=f"{atoms[ac].resname}{atoms[ac].resid}:{atoms[ac].name}",
                                  donor_class="backbone" if atoms[d].name in BACKBONE else "side chain",
                                  acceptor_class="backbone" if atoms[ac].name in BACKBONE else "side chain",
                                  frames=nfr, occupancy=nfr / nF,
                                  share_of_bond_frames=nfr / max(1, int(s.sum()))))
            for f_ in g.frame.unique():
                per_frame.setdefault(int(f_), set()).update({int(d), int(ac)})
                per_frame_h.setdefault(int(f_), set()).add(int(h))
        bond_atoms[p] = per_frame
        bond_h[p] = per_frame_h

    # consistency with the residue-pair signals used for coherence
    npz = os.path.join(a.indir, f"{rep}_hbond.npz")
    check = {}
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        key = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(z["res_i"], z["res_j"]))}
        for p in need:
            if p in key and z["hb"].shape[0] == nF:
                check[p] = float(np.mean(z["hb"][:, key[p]] != sig[p]))
    return dict(rep=rep, t=t, sig=sig, atom_rows=atom_rows, bond_atoms=bond_atoms, bond_h=bond_h,
                check=check, atoms=atoms)


# ----------------------------------------------------------------------------- main analysis
def analyse(a):
    he = pd.read_csv(a.hyperedges)
    he["A"] = he.hb1.apply(parse_pair); he["B"] = he.hb2.apply(parse_pair)
    if a.pairs:
        want = [tuple(sorted((parse_pair(x.split(",")[0]), parse_pair(x.split(",")[1])))) for x in a.pairs]
        rows = []
        for w in want:
            m = he[[tuple(sorted((r.A, r.B))) == w for r in he.itertuples()]]
            if len(m):
                rows.append(m.iloc[0].to_dict())
            else:
                rows.append(dict(hb1=f"{w[0][0]}-{w[0][1]}", hb2=f"{w[1][0]}-{w[1][1]}", A=w[0], B=w[1], n_reps=np.nan))
        sel = rows
    else:
        sel = he[he.n_reps >= a.min_reps].to_dict("records")
    if not sel:
        sys.exit("no hyperedges selected")
    print("hyperedges:", ", ".join(f"{e['hb1']}|{e['hb2']}" for e in sel))
    os.makedirs(a.outdir, exist_ok=True)
    gap = int(round(a.gap_ns / (a.dt_ns * a.stride)))

    summ, trans, atom_rows, shared_rows = [], [], [], []
    rasters = {}
    for rep in a.reps:
        R = analyse_rep(rep, sel, a)
        atom_rows += R["atom_rows"]
        for p, f in R["check"].items():
            if f > 0:
                print(f"  [{rep}] WARNING bond {p}: {100 * f:.2f}% frames differ from {rep}_hbond.npz")
        t = R["t"]; T_end = t[-1] + a.dt_ns * a.stride
        win_all = (t >= a.edge_ns) & (t < T_end - a.edge_ns)
        for e in sel:
            A, B = R["sig"][e["A"]], R["sig"][e["B"]]
            eps = parse_episodes(e.get(f"episodes_ns_{rep}", ""))
            inside = np.zeros_like(win_all)
            for s0, s1 in eps:
                inside |= (t >= s0) & (t < s1)
            inside &= win_all
            tag = f"{e['hb1']}|{e['hb2']}"
            rasters.setdefault(tag, {})[rep] = (t, 2 * A.astype(int) + B.astype(int), eps, win_all)
            for wname, w in [("all", win_all), ("coherent", inside), ("outside", win_all & ~inside)]:
                m, Tm = state_metrics(A[w], B[w], gap)
                if m is None:
                    continue
                row = dict(hyperedge=tag, bond_A=e["hb1"], bond_B=e["hb2"], rep=rep, window=wname,
                           n_reps_coherent=e.get("n_reps", np.nan), **m)
                row["label_heuristic"] = label(m, a)
                summ.append(row)
                for i in range(4):
                    for j in range(4):
                        trans.append(dict(hyperedge=tag, rep=rep, window=wname,
                                          from_state=CODE2LAB[i], to_state=CODE2LAB[j], count=int(Tm[i, j])))
            # shared residue: which of its atoms carry each bond in 11 frames
            shared = set(e["A"]) & set(e["B"])
            if shared:
                sr = shared.pop()
                atoms = R["atoms"]
                both = np.flatnonzero((A == 1) & (B == 1) & win_all)
                same = diff = 0
                usage = {}
                for f_ in both:
                    sa = {atoms[i].name for i in R["bond_atoms"][e["A"]].get(int(f_), set()) if atoms[i].resid == sr}
                    sb = {atoms[i].name for i in R["bond_atoms"][e["B"]].get(int(f_), set()) if atoms[i].resid == sr}
                    if sa & sb:
                        same += 1
                    else:
                        diff += 1
                    key = f"{'/'.join(sorted(sa))} | {'/'.join(sorted(sb))}"
                    usage[key] = usage.get(key, 0) + 1
                top = sorted(usage.items(), key=lambda kv: -kv[1])[:5]
                # donating hydrogens of the shared residue in 11 frames (when it is the donor in both bonds)
                h_same = h_diff = h_n = 0
                h_usage = {}
                for f_ in both:
                    ha = {atoms[i].name for i in R["bond_h"][e["A"]].get(int(f_), set()) if atoms[i].resid == sr}
                    hb_ = {atoms[i].name for i in R["bond_h"][e["B"]].get(int(f_), set()) if atoms[i].resid == sr}
                    if not ha or not hb_:
                        continue                      # shared residue is not the donor in one of the bonds
                    h_n += 1
                    if ha & hb_:
                        h_same += 1                   # one hydrogen donates to both acceptors (bifurcated)
                    else:
                        h_diff += 1                   # two different hydrogens
                    k = f"{'/'.join(sorted(ha))} | {'/'.join(sorted(hb_))}"
                    h_usage[k] = h_usage.get(k, 0) + 1
                h_top = sorted(h_usage.items(), key=lambda kv: -kv[1])[:5]
                shared_rows.append(dict(hyperedge=tag, rep=rep, shared_residue=sr, frames_11=len(both),
                                        same_atom_frac=(same / len(both) if len(both) else np.nan),
                                        different_atoms_frac=(diff / len(both) if len(both) else np.nan),
                                        top_atom_use_A_B="; ".join(f"{k} ({v})" for k, v in top),
                                        frames_11_shared_donor_both=h_n,
                                        same_hydrogen_frac=(h_same / h_n if h_n else np.nan),
                                        different_hydrogens_frac=(h_diff / h_n if h_n else np.nan),
                                        top_hydrogen_use_A_B="; ".join(f"{k} ({v})" for k, v in h_top)))

    pd.DataFrame(summ).to_csv(os.path.join(a.outdir, "states_summary.csv"), index=False, float_format="%.4g")
    pd.DataFrame(trans).to_csv(os.path.join(a.outdir, "transitions.csv"), index=False)
    pd.DataFrame(atom_rows).sort_values(["bond", "rep", "frames"], ascending=[True, True, False]) \
        .to_csv(os.path.join(a.outdir, "atom_pairs.csv"), index=False, float_format="%.4g")
    pd.DataFrame(shared_rows).to_csv(os.path.join(a.outdir, "shared_atom_use.csv"), index=False, float_format="%.4g")
    json.dump(dict(vars(a), donors=DONORS, hydrogens=HYDROGENS, acceptors=ACCEPTORS, gap_frames=gap,
                   state_coding="A=hb1, B=hb2; 00 neither, 10 A only, 01 B only, 11 both"),
              open(os.path.join(a.outdir, "params.json"), "w"), indent=2, default=str)
    if not a.no_plot:
        plot(rasters, a)
    S = pd.DataFrame(summ)
    cols = ["hyperedge", "rep", "window", "p10", "p01", "p11", "ratio11_obs_exp", "phi", "swaps_A_to_B", "swaps_B_to_A", "label_heuristic"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(S[S.window == "coherent"][cols].round(3).to_string(index=False))
    print(f"wrote {a.outdir}/")


def plot(rasters, a):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    cmap = ListedColormap(["#f2f2f2", "#d08c1f", "#2b6cb0", "#2b2b2b"])   # 00, 01(B only), 10(A only), 11
    for tag, d in rasters.items():
        reps = [r for r in a.reps if r in d]
        fig, axs = plt.subplots(len(reps), 1, figsize=(7.1, .55 * len(reps) + .6), sharex=True, squeeze=False)
        for ax, rep in zip(axs[:, 0], reps):
            t, code, eps, win = d[rep]
            ax.imshow(code[None, :], aspect="auto", cmap=cmap, vmin=-.5, vmax=3.5, interpolation="nearest",
                      extent=[t[0] / 1000, (t[-1] + a.dt_ns) / 1000, 0, 1])
            for s0, s1 in eps:
                ax.plot([s0 / 1000, s1 / 1000], [1.12, 1.12], color="#c0504d", lw=2.2, solid_capstyle="butt", clip_on=False)
            ax.axvspan(t[0] / 1000, a.edge_ns / 1000, color="white", alpha=.6, lw=0)
            ax.axvspan((t[-1] - a.edge_ns) / 1000, (t[-1] + a.dt_ns) / 1000, color="white", alpha=.6, lw=0)
            ax.set_yticks([]); ax.set_ylabel(rep, rotation=0, ha="right", va="center", fontsize=7)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
        A, B = tag.split("|")
        axs[-1, 0].set_xlabel("time (µs)")
        fig.legend(handles=[Patch(color="#f2f2f2", label="neither"), Patch(color="#2b6cb0", label=f"{A} only"),
                            Patch(color="#d08c1f", label=f"{B} only"), Patch(color="#2b2b2b", label="both"),
                            Patch(color="#c0504d", label="coherent episode")],
                   loc="upper center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(.5, 1.02))
        plt.tight_layout(rect=(0, 0, 1, .9))
        fig.savefig(os.path.join(a.outdir, f"states_{A}_{B}.png"), dpi=300)
        plt.close(fig)


# ----------------------------------------------------------------------------- self-test
def selftest():
    rng = np.random.default_rng(0)
    n = 4000
    # co-occupied: B follows A
    A = (np.convolve(rng.normal(size=n), np.ones(30) / 30, "same") > 0).astype(int)
    B = A.copy(); flip = rng.random(n) < .05; B[flip] = 1 - B[flip]
    m, _ = state_metrics(A, B, 5)
    assert m["ratio11_obs_exp"] > 1.5 and m["phi"] > .7, m
    # exclusive: B = not A when either present
    on = (np.convolve(rng.normal(size=n), np.ones(30) / 30, "same") > -.3).astype(int)
    side = (np.convolve(rng.normal(size=n), np.ones(60) / 60, "same") > 0).astype(int)
    A2, B2 = on * side, on * (1 - side)
    m2, T2 = state_metrics(A2, B2, 5)
    assert m2["p11"] == 0 and m2["phi"] < 0 and (m2["swaps_A_to_B"] + m2["swaps_B_to_A"]) > 0, m2
    assert T2.sum() == n - 1
    # swap counter
    code = np.array([2, 2, 0, 0, 1, 1, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1])
    assert count_swaps(code, 2) == (1, 1), count_swaps(code, 2)     # 10->(00x2)->01 and 01->10; long 00 gap breaks
    assert parse_episodes("730-1330;1470-3820") == [(730., 1330.), (1470., 3820.)]
    assert parse_episodes(float("nan")) == []
    print("selftest OK")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="wco_hmr")
    p.add_argument("--reps", nargs="+", default=["namd_3", "namd_3_2", "namd_3_3"])
    p.add_argument("--hyperedges", help="hyperedges.csv (default: newest wco_hmr/HB_*/hyperedges.csv)")
    p.add_argument("--pairs", nargs="*", help="hyperedges as 'a-b,c-d' (default: all with >= --min-reps)")
    p.add_argument("--min-reps", type=int, default=3)
    p.add_argument("--outdir", default=None, help="default: <hyperedges folder>/states")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--dt-ns", type=float, default=1.0)
    p.add_argument("--d-a", type=float, default=3.2)
    p.add_argument("--angle", type=float, default=150.0)
    p.add_argument("--edge-ns", type=float, default=722.0, help="excluded at each trajectory end (as in the coherence analysis)")
    p.add_argument("--gap-ns", type=float, default=5.0, help="max 00 interval allowed inside a swap")
    p.add_argument("--min-p11", type=float, default=0.10, help="label: co-occupied if p11 >= this ...")
    p.add_argument("--min-ratio", type=float, default=1.5, help="... and p11/(pA pB) >= this")
    p.add_argument("--max-p11-excl", type=float, default=0.02, help="label: exclusive if p11 <= this")
    p.add_argument("--min-swaps", type=int, default=5, help="... with partner exchange if swaps >= this")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest(); return
    if not a.hyperedges:
        import glob
        found = sorted(glob.glob(os.path.join(a.indir, "HB_*", "hyperedges.csv")), key=os.path.getmtime)
        if not found:
            p.error(f"no {a.indir}/HB_*/hyperedges.csv found; give --hyperedges")
        a.hyperedges = found[-1]
        print(f"using {a.hyperedges}")
    if a.outdir is None:
        a.outdir = os.path.join(os.path.dirname(a.hyperedges), "states")
    analyse(a)


if __name__ == "__main__":
    main()
