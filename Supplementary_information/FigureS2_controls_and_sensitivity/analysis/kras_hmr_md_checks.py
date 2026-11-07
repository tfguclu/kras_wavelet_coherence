#!/usr/bin/env python
"""
kras_hmr_md_checks.py -- simulation and robustness diagnostics requested in review.

  structure  per replica, every --stride-ns along the dry wrapped trajectory:
             CA RMSD of the core (all residues except Switch I 30-38, Switch II 59-76 and
             residues < 2 / > 164) after fitting on the core, to the first frame and to
             --ref (if given); CA RMSD of Switch I and Switch II after the core fit;
             Mg2+ first shell from protein/GTP oxygens (< 2.6 A; S17 OG, T35 OG1, beta and
             gamma phosphate O); GTP heavy-atom RMSD after the core fit (retention);
             T35 OG1-Mg and G60 N-GTP gamma-O distances (Switch I/II contacts with GTP).
             With --ref (e.g. 6VJJ chain A), also: start model vs reference (core, Switch I,
             Switch II, nucleotide atoms with matching names, Mg position).
  blocks     do the 16 core pairs (and the EXTRA_PAIRS) have episodes in both halves of the analysed trajectory?
             (episodes from the baseline runs; per replica, first and second half)
  jumps      for every core SASA pair: largest change of the 50 ns running mean over 100 ns
             within each episode, divided by the fast fluctuation (SD around the 50 ns
             running mean). Synthetic tests: a single shared step needs >= ~27x to make an
             episode (Methods); values of a few x mean no single jump drives the episode.
  volume     box volume per frame from NAMD .xst files (--xst "namd_3/*.xst" ...), if kept.

Run from the analysis folder (needs MDAnalysis for 'structure'):
  python kras_hmr_md_checks.py structure --ref 6vjj_chainA.pdb
  python kras_hmr_md_checks.py blocks jumps --accept-legacy-real
  python kras_hmr_md_checks.py volume --xst "../namd_3/step5_production*.xst" ...
Output: <indir>/md_checks/*.csv, *.png
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SW1, SW2 = (30, 38), (59, 76)
CORE = "name CA and resid 2:164 and not resid 30:38 and not resid 59:76"
CORE_PAIRS = [  # (label, observable, key as in the run outputs)
    ("HB R102: D69 | Q99", "hb", ("69-102", "99-102")), ("HB R73: V103 | D105", "hb", ("73-103", "73-105")),
    ("HB K88: E91 | D92", "hb", ("88-91", "88-92")), ("HB T144: N116 | Q150", "hb", ("116-144", "144-150")),
    ("SASA D69-R102", "sasa", (69, 102)), ("SASA G75-K104", "sasa", (75, 104)), ("SASA K88-D92", "sasa", (88, 92)),
    ("SASA H94-L133", "sasa", (94, 133)), ("SASA V8-K16", "sasa", (8, 16)), ("SASA G10-K16", "sasa", (10, 16)),
    ("SASA C80-I93", "sasa", (80, 93)),
    ("chi K128-D132", "chi", (128, 132)), ("chi D126-K128", "chi", (126, 128)), ("chi E3-D54", "chi", (3, 54)),
    ("chi L52-D54", "chi", (52, 54)), ("chi M1-L52", "chi", (1, 52)),
]
# also tracked in 'blocks': the non-robust chi/SASA pairs and the ten direct DMS overlaps (Figure S1b rows).
# "hbres": an H-bond residue pair, matched to every hyperedge containing both residues (as in the sensitivity scan)
EXTRA_PAIRS = [
    ("chi Q99-R102", "chi", (99, 102)), ("chi D69-R102", "chi", (69, 102)), ("chi R73-K104", "chi", (73, 104)),
    ("SASA R73-K104", "sasa", (73, 104)),
    ("SASA S17-Y32", "sasa", (17, 32)), ("SASA G12-G60", "sasa", (12, 60)), ("SASA I21-Y40", "sasa", (21, 40)),
    ("chi V8-Y40", "chi", (8, 40)), ("chi I21-T35", "chi", (21, 35)), ("chi T74-V112", "chi", (74, 112)),
    ("chi L133-Y137", "chi", (133, 137)),
    ("HB E31-Y40", "hbres", (31, 40)), ("HB Y32-Y40", "hbres", (32, 40)), ("HB E37-T58", "hbres", (37, 58)),
]
BASE = dict(pmin=80, pmax=120, thr=0.75, ep=500, tdec=10)


# ----------------------------------------------------------------------------- structure
def structure(a, outdir):
    import MDAnalysis as mda
    from MDAnalysis.analysis import align
    from MDAnalysis.lib.distances import distance_array
    ref_u = mda.Universe(a.ref) if a.ref else None
    rows, startref = [], []
    for rep in a.reps:
        top = os.path.join(a.indir, f"{rep}_wrapped.pdb"); trj = os.path.join(a.indir, f"{rep}_wrapped.dcd")
        u = mda.Universe(top, trj)
        first = mda.Universe(top, trj); first.trajectory[0]
        core = u.select_atoms(CORE); core0 = first.select_atoms(CORE)
        sw1 = u.select_atoms(f"name CA and resid {SW1[0]}:{SW1[1]}"); sw2 = u.select_atoms(f"name CA and resid {SW2[0]}:{SW2[1]}")
        sw1_0 = first.select_atoms(f"name CA and resid {SW1[0]}:{SW1[1]}"); sw2_0 = first.select_atoms(f"name CA and resid {SW2[0]}:{SW2[1]}")
        gtp = u.select_atoms("resname GTP and not name H*"); gtp0 = first.select_atoms("resname GTP and not name H*")
        mg = u.select_atoms("name MG")
        lig = u.select_atoms("(protein or resname GTP) and name O* and not name H*")
        t35 = u.select_atoms("resid 35 and name OG1"); g60 = u.select_atoms("resid 60 and name N")
        og = u.select_atoms("resname GTP and name O1G O2G O3G")
        dt = a.dt_ns                                                     # ns per frame of the dry dcd
        step = max(1, int(round(a.stride_ns / dt)))
        x0 = core0.positions - core0.center_of_mass()
        for ts in u.trajectory[::step]:
            R, rmsd_core = align.rotation_matrix(core.positions - core.center_of_mass(), x0)
            com = core.center_of_mass()
            fit = lambda X: (X - com) @ R.T + first.select_atoms(CORE).center_of_mass()
            r1 = np.sqrt(((fit(sw1.positions) - sw1_0.positions) ** 2).sum(1).mean())
            r2 = np.sqrt(((fit(sw2.positions) - sw2_0.positions) ** 2).sum(1).mean())
            rg = np.sqrt(((fit(gtp.positions) - gtp0.positions) ** 2).sum(1).mean())
            d = distance_array(mg.positions, lig.positions, box=ts.dimensions)[0]
            shell = lig[d < 2.6]
            names = sorted(f"{at.resname}{at.resid}:{at.name}" for at in shell)
            rows.append(dict(replica=rep, time_ns=ts.frame * dt, rmsd_core=rmsd_core, rmsd_sw1=r1, rmsd_sw2=r2,
                             rmsd_gtp=rg, mg_n_protein_gtp_O=len(shell), mg_shell=";".join(names),
                             t35_mg=float(distance_array(t35.positions, mg.positions, box=ts.dimensions).min()),
                             g60N_gammaO=float(distance_array(g60.positions, og.positions, box=ts.dimensions).min())))
        if ref_u is not None:
            startref.append(dict(replica=rep, **compare_ref(first, ref_u)))
    D = pd.DataFrame(rows); D.to_csv(os.path.join(outdir, "structure_timeseries.csv"), index=False)
    S = D.groupby("replica").agg(rmsd_core_mean=("rmsd_core", "mean"), rmsd_core_max=("rmsd_core", "max"),
                                 rmsd_sw1_mean=("rmsd_sw1", "mean"), rmsd_sw2_mean=("rmsd_sw2", "mean"),
                                 rmsd_gtp_max=("rmsd_gtp", "max"),
                                 mg_shell_mode=("mg_shell", lambda v: v.mode().iat[0]),
                                 mg_shell_mode_frac=("mg_shell", lambda v: (v == v.mode().iat[0]).mean()),
                                 t35_mg_frac_lt_2p6=("t35_mg", lambda v: (v < 2.6).mean()),
                                 g60_gamma_frac_lt_3p5=("g60N_gammaO", lambda v: (v < 3.5).mean()))
    # halves: does the mean structure drift?
    D["half"] = np.where(D.time_ns < D.groupby("replica").time_ns.transform("max") / 2, "first", "second")
    H = D.groupby(["replica", "half"])[["rmsd_core", "rmsd_sw1", "rmsd_sw2"]].mean().unstack()
    S.to_csv(os.path.join(outdir, "structure_summary.csv")); H.to_csv(os.path.join(outdir, "structure_halves.csv"))
    if startref:
        pd.DataFrame(startref).to_csv(os.path.join(outdir, "start_vs_reference.csv"), index=False)
        print(pd.DataFrame(startref).to_string(index=False))
    print(S.to_string()); print(H.round(2).to_string())
    plot_structure(D, os.path.join(outdir, "structure_timeseries.png"))


def compare_ref(u, ref):
    from MDAnalysis.analysis import align
    common = sorted(set(u.select_atoms(CORE).resids) & set(ref.select_atoms(CORE).resids))
    sel = lambda U, extra="": U.select_atoms(f"name CA and resid {' '.join(map(str, common))}{extra}")
    A, B = sel(u), sel(ref)
    R, rc_ = align.rotation_matrix(A.positions - A.center_of_mass(), B.positions - B.center_of_mass())
    fit = lambda X: (X - A.center_of_mass()) @ R.T + B.center_of_mass()
    out = dict(core_ca_rmsd=rc_)
    for nm, (lo, hi) in [("sw1", SW1), ("sw2", SW2)]:
        a_ = u.select_atoms(f"name CA and resid {lo}:{hi}"); b_ = ref.select_atoms(f"name CA and resid {lo}:{hi}")
        ids = sorted(set(a_.resids) & set(b_.resids))
        a_ = u.select_atoms(f"name CA and resid {' '.join(map(str, ids))}"); b_ = ref.select_atoms(f"name CA and resid {' '.join(map(str, ids))}")
        out[f"{nm}_ca_rmsd"] = float(np.sqrt(((fit(a_.positions) - b_.positions) ** 2).sum(1).mean()))
    nuc_ref = ref.select_atoms("resname GNP GTP GCP GSP and not name H*")
    nuc = u.select_atoms("resname GTP and not name H*")
    names = sorted(set(nuc.names) & set(nuc_ref.names))
    if names:
        p = np.array([nuc.select_atoms(f"name {n}").positions[0] for n in names])
        q = np.array([nuc_ref.select_atoms(f"name {n}").positions[0] for n in names])
        out.update(nucleotide_rmsd=float(np.sqrt(((fit(p) - q) ** 2).sum(1).mean())), nucleotide_atoms=len(names))
    mg, mg_r = u.select_atoms("name MG"), ref.select_atoms("name MG")
    if len(mg) and len(mg_r):
        out["mg_shift_A"] = float(np.linalg.norm(fit(mg.positions)[0] - mg_r.positions[0]))
    return out


def plot_structure(D, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols = ["rmsd_core", "rmsd_sw1", "rmsd_sw2", "rmsd_gtp", "mg_n_protein_gtp_O"]
    fig, ax = plt.subplots(len(cols), 1, figsize=(7, 1.4 * len(cols)), sharex=True)
    for c, x in zip(cols, ax):
        for rep, g in D.groupby("replica"):
            x.plot(g.time_ns / 1000, g[c], lw=.6, label=rep)
        x.set_ylabel(c, fontsize=7)
    ax[0].legend(fontsize=6, ncol=3); ax[-1].set_xlabel("time (µs)")
    fig.tight_layout(); fig.savefig(path, dpi=200)


# ----------------------------------------------------------------------------- blocks / jumps
def baseline_dirs(a):
    import kras_hmr_runcheck as rc
    out = {}
    for o in ["chi", "sasa", "hb"]:
        h = rc.input_hashes(a.indir, o, a.reps)
        out[o] = rc.find_valid(o, BASE, a.indir, a.reps, rc.expectation(o, BASE, a.reps, h),
                               accept_legacy=a.accept_legacy_real, verbose=False)
    return out


def pair_episodes(o, d, key, reps):
    from kras_hmr_controls_review import parse_eps
    if o == "hbres":
        h = pd.read_csv(os.path.join(d, "hyperedges.csv"))
        res = lambda r: {int(x) for x in r.hb1.split("-")} | {int(x) for x in r.hb2.split("-")}
        m = h[[set(key) <= res(r) for r in h.itertuples()]]
        return {r: sum((parse_eps(x) for x in m[f"episodes_ns_{r}"]), []) for r in reps}
    if o == "hb":
        h = pd.read_csv(os.path.join(d, "hyperedges.csv"))
        m = h[((h.hb1 == key[0]) & (h.hb2 == key[1])) | ((h.hb1 == key[1]) & (h.hb2 == key[0]))]
        return {r: sum((parse_eps(x) for x in m[f"episodes_ns_{r}"]), []) for r in reps}
    e = pd.read_csv(os.path.join(d, "episodes_all_reps.csv"))
    m = e[(e.resid_i == key[0]) & (e.resid_j == key[1])]
    return {r: sum((parse_eps(x) for x in m[f"episodes_ns_{r}"]), []) for r in reps}


def blocks(a, outdir):
    import json
    dirs = baseline_dirs(a); rows = []
    for lab, o, key in CORE_PAIRS + EXTRA_PAIRS:
        d = dirs["hb" if o == "hbres" else o]
        if not d:
            continue
        P = json.load(open(os.path.join(d, "params.json"))); M = P.get("meta", {})
        eps = pair_episodes(o, d, key, a.reps)
        for r in a.reps:
            edge = float(M.get(r, {}).get("coi_edge_ns", 722.0)); t1 = 10000.0 - edge
            mid = (edge + t1) / 2
            cov = lambda lo, hi: sum(max(0.0, min(e, hi) - max(s, lo)) for s, e in eps[r])
            rows.append(dict(pair=lab, replica=r, first_half_ns=cov(edge, mid), second_half_ns=cov(mid, t1)))
    B = pd.DataFrame(rows); B.to_csv(os.path.join(outdir, "blocks.csv"), index=False)
    S = B.assign(f=B.first_half_ns > 0, s=B.second_half_ns > 0).groupby("pair").agg(
        reps_first_half=("f", "sum"), reps_second_half=("s", "sum"))
    S = S.reindex([l for l, *_ in CORE_PAIRS + EXTRA_PAIRS]); S.to_csv(os.path.join(outdir, "blocks_summary.csv"))
    print(S.to_string())


def jumps(a, outdir):
    dirs = baseline_dirs(a); rows = []
    rm = lambda x, w=50: np.convolve(x, np.ones(w) / w, mode="same")
    for r in a.reps:
        z = np.load(os.path.join(a.indir, f"{r}_sasa.npz"), allow_pickle=True)
        X, res, t = z["sasa"], [int(v) for v in z["resid"]], np.asarray(z["time_ns"], float)
        col = {v: i for i, v in enumerate(res)}
        for lab, o, key in CORE_PAIRS:
            if o != "sasa" or not dirs["sasa"]:
                continue
            for s, e in pair_episodes("sasa", dirs["sasa"], key, [r])[r]:
                m = (t >= s) & (t < e); rr = []
                for k in key:
                    x = X[:, col[k]].astype(float); fast = np.std((x - rm(x))[100:-100])
                    mm = rm(x)[m]; jump = np.max(np.abs(mm[100:] - mm[:-100])) if len(mm) > 100 else np.nan
                    rr.append(jump / fast)
                rows.append(dict(pair=lab, replica=r, start_ns=s, end_ns=e, ratio_res1=rr[0], ratio_res2=rr[1]))
    J = pd.DataFrame(rows); J.to_csv(os.path.join(outdir, "sasa_jumps.csv"), index=False, float_format="%.2f")
    if len(J):
        print(J.groupby("pair")[["ratio_res1", "ratio_res2"]].max().round(1).to_string())
        print("largest ratio over all core SASA episodes:", round(float(J[["ratio_res1", "ratio_res2"]].max().max()), 1))


def volume(a, outdir):
    rows = []
    for pat in a.xst:
        for f in sorted(glob.glob(pat)):
            x = np.loadtxt(f, comments="#")
            x = x[None] if x.ndim == 1 else x
            V = np.abs(np.einsum("ij,ij->i", x[:, 1:4], np.cross(x[:, 4:7], x[:, 7:10])))
            rows += [dict(file=f, step=int(s), volume_A3=v) for s, v in zip(x[:, 0], V)]
    D = pd.DataFrame(rows); D.to_csv(os.path.join(outdir, "volume.csv"), index=False)
    if len(D):
        D["dir"] = D.file.map(os.path.dirname)
        print(D.groupby("dir").volume_A3.agg(["mean", "std", "min", "max"]).to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", nargs="+", choices=["structure", "blocks", "jumps", "volume"])
    p.add_argument("--indir", default="wco_hmr")
    p.add_argument("--reps", nargs="+", default=["namd_3", "namd_3_2", "namd_3_3"])
    p.add_argument("--ref", default=None, help="reference PDB (e.g. 6VJJ chain A, canonical numbering)")
    p.add_argument("--stride-ns", type=float, default=10.0)
    p.add_argument("--dt-ns", type=float, default=1.0, help="time between frames of the wrapped dcd")
    p.add_argument("--xst", nargs="*", default=[], help="glob(s) of NAMD .xst files")
    p.add_argument("--accept-legacy-real", action="store_true")
    a = p.parse_args()
    a.indir = os.path.abspath(a.indir)
    outdir = os.path.join(a.indir, "md_checks"); os.makedirs(outdir, exist_ok=True)
    for w in a.what:
        print(f"=== {w}")
        dict(structure=structure, blocks=blocks, jumps=jumps, volume=volume)[w](a, outdir)
    print(f"wrote {outdir}/")


if __name__ == "__main__":
    main()
