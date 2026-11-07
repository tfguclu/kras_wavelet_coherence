#!/usr/bin/env python
"""
kras_k88_split.py -- is the K88 alternation carried by the side chain or by the helix backbone?

The residue-pair signal K88-D92 is 1 when K88 and D92 share any H-bond. It mixes two bonds:
the alpha3 backbone H-bond (D92 N-H ... K88 O, i -> i+4) and side-chain contacts (salt bridge
K88 NZ ... D92 OD1/OD2, or any bond with a side-chain atom). This script splits it.

Per replica, atom-level H-bonds among K88, E91 and D92 are recomputed with the criteria of
kras_hmr_hbond.py (donor-acceptor <= 3.2 A, D-H...A >= 150 deg), giving four 0/1 signals:
  E91      K88-E91, any H-bond
  D92      K88-D92, any H-bond (should equal the signal used for coherence)
  D92bb    K88-D92, backbone-backbone H-bond only
  D92sc    K88-D92, any H-bond involving a side-chain atom
Then, with the unchanged coherence function of kras_hmr_hbond_wco.py (80-120 ns, 0.75, 500 ns,
smoothing 2s, power floor 0.05, same edge exclusion), the pairs E91|D92, E91|D92bb, E91|D92sc and
D92bb|D92sc are analysed in each replica, and compared with --n-ctrl time-shifted controls
(each signal circularly shifted by its own random offset of at least --min-shift-ns).
Signals failing the H-bond signal filter (occupancy 0.05-0.95, >= 10 formations) are reported
and not analysed, as in the main analysis.

Reports per pair and replica: coherent coverage, episodes, mean phase within episodes
(~0 together, ~180 alternating), joint occupancy ratio P(both)/(P(A)P(B)); per pair the number of
replicas with an episode and k = controls reaching that number (p = (1+k)/(1+n)).

Run from the analysis folder (needs wco_hmr/{rep}_wrapped.pdb/.dcd and {rep}_hbond.npz):
  python kras_k88_split.py                 # 200 controls
  python kras_k88_split.py --n-ctrl 20     # quick
Writes wco_hmr/k88_split/: signals_summary.csv, pairs.csv, controls.csv, report.txt
Test without trajectories: python kras_k88_split.py --selftest
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPS = ["namd_3", "namd_3_2", "namd_3_3"]
SIGNALS = ["E91", "D92", "D92bb", "D92sc"]
PAIRS = [("E91", "D92"), ("E91", "D92bb"), ("E91", "D92sc"), ("D92bb", "D92sc")]
BACKBONE = {"N", "O", "OT1", "OT2", "OXT", "HN", "H"}


def atom_signals(rep, a):
    import MDAnalysis as mda
    from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
    from kras_hmr_hbond_states import DONORS, HYDROGENS, ACCEPTORS
    u = mda.Universe(os.path.join(a.indir, f"{rep}_wrapped.pdb"), os.path.join(a.indir, f"{rep}_wrapped.dcd"))
    rsel = "resid 88 91 92"
    hba = HydrogenBondAnalysis(universe=u, donors_sel=f"{DONORS} and {rsel}", hydrogens_sel=f"{HYDROGENS} and {rsel}",
                               acceptors_sel=f"{ACCEPTORS} and {rsel}", d_h_cutoff=1.2, d_a_cutoff=a.d_a,
                               d_h_a_angle_cutoff=a.angle, update_selections=False)
    hba.run()
    hb = hba.results.hbonds
    nF = u.trajectory.n_frames
    S = {k: np.zeros(nF, np.uint8) for k in SIGNALS}
    at = u.atoms
    for fr, d, h, ac in hb[:, :4].astype(int):
        rd, ra = at[d].resid, at[ac].resid
        pair = tuple(sorted((rd, ra)))
        bb = at[d].name in BACKBONE and at[ac].name in BACKBONE
        if pair == (88, 91):
            S["E91"][fr] = 1
        elif pair == (88, 92):
            S["D92"][fr] = 1
            S["D92bb" if bb else "D92sc"][fr] = 1
    return S, nF


def coh_args(a, dt, t0):
    return argparse.Namespace(pmin=a.pmin, pmax=a.pmax, thr=a.thr, min_ep=a.min_ep, tsmooth=a.tsmooth,
                              tdec=a.tdec, dj=a.dj, dt_ns=dt, pow_floor=a.pow_floor, chunk=64, t0_ns=t0)


def run_pairs(S, names, ca):
    """coherence for all pairs among the given signals -> {(n1,n2): row}"""
    from kras_hmr_hbond_wco import analyse
    M = np.stack([S[n] for n in names], axis=1)
    labels = [(i + 1, 1000 + i) for i in range(len(names))]          # unique dummy labels, order kept
    rows, meta = analyse(M, labels, ca, log=lambda *x: None)
    out = {}
    for r in rows:
        i = int(r["hb1"].split("-")[0]) - 1; j = int(r["hb2"].split("-")[0]) - 1
        out[(names[i], names[j])] = r
    return out, meta


def passes(x, a):
    occ = x.mean(); tr = int(((x[1:] == 1) & (x[:-1] == 0)).sum())
    return (a.occ_min <= occ <= a.occ_max) and tr >= a.min_trans, occ, tr


def joint_ratio(A, B, valid):
    A, B = A[valid].astype(bool), B[valid].astype(bool)
    e = A.mean() * B.mean()
    return (A & B).mean() / e if e > 0 else np.nan, (A & B).mean()


def mean_phase(r):
    ph = str(r.get("ep_phase_deg", "") or "")
    if not ph or ph == "nan":
        return np.nan
    v = np.radians([float(x) for x in ph.split(";") if x])
    return float(np.degrees(np.angle(np.exp(1j * v).mean())))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="wco_hmr")
    p.add_argument("--reps", nargs="+", default=REPS)
    p.add_argument("--n-ctrl", type=int, default=200)
    p.add_argument("--min-shift-ns", type=float, default=1000.0)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--pmin", type=float, default=80.0); p.add_argument("--pmax", type=float, default=120.0)
    p.add_argument("--thr", type=float, default=0.75); p.add_argument("--min-ep", type=float, default=500.0)
    p.add_argument("--tsmooth", type=float, default=2.0); p.add_argument("--tdec", type=int, default=10)
    p.add_argument("--dj", type=float, default=1 / 12); p.add_argument("--pow-floor", type=float, default=0.05)
    p.add_argument("--occ-min", type=float, default=0.05); p.add_argument("--occ-max", type=float, default=0.95)
    p.add_argument("--min-trans", type=int, default=10)
    p.add_argument("--d-a", type=float, default=3.2); p.add_argument("--angle", type=float, default=150.0)
    p.add_argument("--edge-ns", type=float, default=722.0, help="excluded ends for the joint ratios")
    p.add_argument("--signals-npz", default=None, help="testing: npz with E91,D92,D92bb,D92sc per replica")
    p.add_argument("--outdir", default=None)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest(a)
    out = a.outdir or os.path.join(a.indir, "k88_split"); os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    sig_rows, pair_rows, data = [], [], {}
    for rep in a.reps:
        t0c = time.time()
        if a.signals_npz:
            z = np.load(a.signals_npz); S = {k: z[f"{rep}_{k}"] for k in SIGNALS}; nF = len(S["E91"]); t0, dt = 0.0, 1.0
        else:
            S, nF = atom_signals(rep, a)
            z = np.load(os.path.join(a.indir, f"{rep}_hbond.npz"), allow_pickle=True)
            t = z["time_ns"].astype(float); dt, t0 = float(np.diff(t)[0]), float(t[0])
            key = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(z["res_i"], z["res_j"]))}
            if (88, 92) in key and z["hb"].shape[0] == nF:
                mism = float(np.mean(z["hb"][:, key[(88, 92)]] != S["D92"]))
                print(f"[{rep}] K88-D92 vs residue-pair signal: {100 * mism:.2f}% frames differ")
        ok = {}
        for k in SIGNALS:
            good, occ, tr = passes(S[k], a); ok[k] = good
            sig_rows.append(dict(replica=rep, signal=k, occupancy=occ, formations=tr, passes_filter=good))
        valid = (np.arange(nF) * dt >= a.edge_ns) & (np.arange(nF) * dt <= (nF - 1) * dt - a.edge_ns)
        names = [k for k in SIGNALS if ok[k]]
        ca = coh_args(a, dt, t0)
        res, meta = run_pairs(S, names, ca) if len(names) >= 2 else ({}, {})
        data[rep] = (S, names, ca, nF, dt)
        for A, B in PAIRS:
            r = res.get((A, B)) or res.get((B, A))
            ratio, p11 = joint_ratio(S[A], S[B], valid)
            pair_rows.append(dict(replica=rep, pair=f"{A}|{B}", analysed=r is not None,
                                  n_episodes=(r["n_episodes"] if r else np.nan),
                                  coverage=(r["total_ep_ns"] / meta["valid_ns"] if r else np.nan),
                                  mean_phase_deg=(mean_phase(r) if r else np.nan),
                                  joint_ratio=ratio, p_both=p11))
        print(f"[{rep}] signals {names}; {time.time() - t0c:.0f} s")
    SG = pd.DataFrame(sig_rows); PR = pd.DataFrame(pair_rows)
    SG.to_csv(os.path.join(out, "signals_summary.csv"), index=False, float_format="%.4g")
    PR.to_csv(os.path.join(out, "pairs.csv"), index=False, float_format="%.4g")
    real_n = {f"{A}|{B}": int((PR[(PR.pair == f"{A}|{B}")].n_episodes.fillna(0) > 0).sum()) for A, B in PAIRS}

    # controls: each signal shifted circularly by its own offset, separately per replica
    crow = []
    for c in range(1, a.n_ctrl + 1):
        n = {k: 0 for k in real_n}
        for rep in a.reps:
            S, names, ca, nF, dt = data[rep]
            if len(names) < 2:
                continue
            lo, hi = int(a.min_shift_ns / dt), nF - int(a.min_shift_ns / dt)
            Sc = {k: np.roll(S[k], int(rng.integers(lo, hi))) for k in names}
            res, _ = run_pairs(Sc, names, ca)
            for A, B in PAIRS:
                r = res.get((A, B)) or res.get((B, A))
                if r and r["n_episodes"] > 0:
                    n[f"{A}|{B}"] += 1
        crow.append(dict(control=c, **n))
        if c % 20 == 0:
            print(f"  controls {c}/{a.n_ctrl}")
    CT = pd.DataFrame(crow); CT.to_csv(os.path.join(out, "controls.csv"), index=False)

    L = ["K88 split: E91 = K88-E91; D92 = K88-D92 any; D92bb = backbone-backbone; D92sc = with a side-chain atom",
         "", "Signals (occupancy, formations, passes filter) per replica:"]
    for x in SG.itertuples():
        L.append(f"  {x.replica:9s} {x.signal:6s} occ {x.occupancy:.3f}  formations {x.formations:5d}  {'ok' if x.passes_filter else 'FAILS filter'}")
    L += ["", "Pairs per replica: episodes, coverage, mean phase (deg), joint ratio P(both)/P(A)P(B):"]
    for x in PR.itertuples():
        L.append(f"  {x.replica:9s} {x.pair:12s} " + (f"ep {int(x.n_episodes)}  cov {x.coverage:.2f}  phase {x.mean_phase_deg:6.0f}"
                 if x.analysed else "not analysed (signal fails filter)") + f"  ratio {x.joint_ratio:.2f}")
    L += ["", f"Recurrence vs {a.n_ctrl} time-shifted controls:"]
    for k, v in real_n.items():
        kk = int((CT[k] >= v).sum()) if v > 0 else a.n_ctrl
        L.append(f"  {k:12s} real {v} replicas; controls reaching it {kk}; p = {(1 + kk) / (1 + a.n_ctrl):.3f}" if v
                 else f"  {k:12s} real 0 replicas")
    L += ["", "Reading: antiphase (~180 deg) and recurrence of E91|D92sc -> the alternation is carried by the side chain;",
          "only E91|D92bb -> helix backbone fluctuation."]
    open(os.path.join(out, "report.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    json.dump(vars(a), open(os.path.join(out, "params.json"), "w"), indent=1, default=str)


def selftest(a):
    """synthetic: E91 alternates with a side-chain bond in 100 ns blocks; backbone bond independent noise"""
    rng = np.random.default_rng(0); T = 10000
    d = {}
    for rep in REPS:
        blocks = (np.sin(2 * np.pi * np.arange(T) / 100 + rng.uniform(0, 6)) > 0).astype(np.uint8)
        flip = rng.random(T) < .05
        e91 = np.where(flip, 1 - blocks, blocks); sc = np.where(rng.random(T) < .05, blocks, 1 - blocks)
        bb = (rng.random(T) < .6).astype(np.uint8)
        d.update({f"{rep}_E91": e91, f"{rep}_D92sc": sc, f"{rep}_D92bb": bb, f"{rep}_D92": (sc | bb)})
    f = "/tmp/k88_selftest.npz"; np.savez(f, **d)
    sys.argv = [sys.argv[0], "--signals-npz", f, "--n-ctrl", "5", "--outdir", "/tmp/k88_selftest"]
    main()


if __name__ == "__main__":
    main()
