#!/usr/bin/env python
"""
kras_vanilla_check.py -- do HMR and the 4 fs time step change the dynamics this study measures?

Compares three runs without HMR (2 fs time step, 1 frame per ns) with the HMR runs, both
cut to the first 1.5 us (1500 frames), using the pipeline's own signal and coherence code.

STEPS (run from the analysis folder that holds wco_hmr/ and the kras_hmr_*.py scripts)
  python kras_vanilla_check.py signals     # build the signals of both data sets
  python kras_vanilla_check.py kinetics    # timescales and populations, signal by signal
  python kras_vanilla_check.py coherence   # coherence at 20-30 and 40-60 ns, core pairs
  python kras_vanilla_check.py all         # the three in order

signals
  wco_vanilla_first1500ns/vanilla_{1,2,3}_*: first --frames frames of the DCDs numbered 1, 2, 3 in
     vanilla_MD_results/,
     wrapped as in kras_hmr_prep.py (protein, GTP, Mg made whole; protein centred);
     chi angles with kras_hmr_prep.build_chi_index (same angles, same order);
     SASA and H-bonds with kras_hmr_sasa.py and kras_hmr_hbond.py (unchanged settings).
  wco_hmr_first1500ns/namd_3*_{chi,sasa,hbond}.npz: copies of wco_hmr/ signal files cut to the
     same first --frames frames (wco_hmr/ itself is only read).
  The chi angles, SASA residues and their order must match between the two; this is checked.

kinetics  (vanilla_vs_hmr_first1500ns/)
  Per signal and replica, on the 1500 frames:
    integrated autocorrelation time  dt*(1/2 + sum of the normalised autocorrelation up to its first
                        negative value, at most 500 ns); chi: of exp(i chi), symmetric groups doubled
    autocorrelation at 10 ns and at 100 ns
    chi                 circular mean resultant length (how spread the angle is)
    SASA                mean and SD
    H-bond              occupancy and formations per us (0 -> 1 transitions)
  The two methods are compared signal by signal. Every difference between methods is set
  against the difference between replicas of the same method, so that "different" means
  more different than replicas of one method are from each other.
  Core H-bond pairs (R102, R73, T144, K88): occupancies, joint occupancy against independent
  bonds, and switches per us (one bond to the other with <= 5 ns of neither).
  Normalised power spectra (Welch) per observable, averaged over signals and replicas.

coherence  (coherence runs inside the two signal folders; reports in vanilla_vs_hmr_first1500ns/)
  kras_hmr_wco.py, kras_hmr_sasa_wco.py and kras_hmr_hbond_wco.py with the manuscript's
  band-scan settings at 20-30 ns and 40-60 ns. The 80-120 ns band is not possible: its
  excluded ends (722 ns each) leave 56 ns of a 1.5 us run. Reports the numbers of coherent
  pairs in 1, 2 and 3 replicas, their overlap between methods, and the 16 core pairs.

OUTPUTS (folder names carry the length, so another --frames never overwrites them; the script
refuses to write into wco_hmr/, wco_hmr_null*/, wlcc_*/, vanilla_MD_results/ or KRAS_unbound_mg_gtp_1/)
  wco_vanilla_first1500ns/   vanilla_{1,2,3}_wrapped.pdb/.dcd, _chi.npz, _sasa.npz, _hbond.npz,
                             and the coherence folders P20-30_*, SASA_P20-30_*, HB_P20-30_*, P40-60_* ...
  wco_hmr_first1500ns/       namd_3*_chi/_sasa/_hbond.npz and the same coherence folders
  vanilla_vs_hmr_first1500ns/
      per_signal_replica.csv, kinetics_summary.csv, core_hbonds.csv, spectra.csv, kinetics.png,
      kinetics_report.txt, coherence_counts.csv, coherence_core_pairs.csv, coherence_report.txt

Requires the pipeline's environment (MDAnalysis, freesasa, numpy, scipy, pandas, matplotlib).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HMR_REPS = ["namd_3", "namd_3_2", "namd_3_3"]
VAN_REPS = ["vanilla_1", "vanilla_2", "vanilla_3"]
SYMMETRIC = {("ASP", 2), ("GLU", 3), ("PHE", 2), ("TYR", 2)}
KINDS = ("chi", "sasa", "hbond")
BANDS = {"P20": dict(pmin=20, pmax=30, ep=125, tdec=4), "P40": dict(pmin=40, pmax=60, ep=250, tdec=8)}
CORE = {
    "chi": [(1, 52), (3, 54), (52, 54), (128, 132), (126, 128)],
    "sasa": [(69, 102), (75, 104), (88, 92), (94, 133), (10, 16), (8, 16), (80, 93)],
    "hb": [((69, 102), (99, 102)), ((73, 103), (73, 105)), ((116, 144), (144, 150)), ((88, 91), (88, 92))],
}
HB_PAIRS = {"R102 (D69 | Q99)": ((69, 102), (99, 102)), "R73 (V103 | D105)": ((73, 103), (73, 105)),
            "T144 (N116 | Q150)": ((116, 144), (144, 150)), "K88 (E91 | D92)": ((88, 91), (88, 92))}


def run(cmd):
    print("   " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode:
        sys.exit(f"failed: {' '.join(cmd)}")


# ============================================================================ signals
def build_vanilla(a):
    import warnings
    warnings.filterwarnings("ignore")
    import MDAnalysis as mda
    from MDAnalysis import transformations as trans
    from MDAnalysis.lib.distances import calc_dihedrals
    sys.path.insert(0, HERE)
    import kras_hmr_prep as prep

    import re
    psf = os.path.join(a.vanilla_dir, a.psf)
    found = {}
    for f in glob.glob(os.path.join(a.vanilla_dir, a.dcd_glob)):
        m = re.search(r"_(\d+)_wrapped", os.path.basename(f))
        if m:
            found[int(m.group(1))] = f
    ready = all(os.path.exists(os.path.join(a.out_vanilla, f"vanilla_{k}{s}")) for k in (1, 2, 3)
                for s in ("_chi.npz", "_wrapped.dcd", "_wrapped.pdb"))
    if ready and not a.force:
        # deposited wrapped trajectories: chi angles from kras_hmr_prep.py --from-wrapped, then SASA and H-bonds
        found = {k: None for k in (1, 2, 3)}
    elif sorted(found) != [1, 2, 3] or not os.path.exists(psf):
        sys.exit(f"need {a.psf} and DCDs numbered 1, 2, 3 in {a.vanilla_dir} "
                 f"(found {sorted(found)}, PSF present: {os.path.exists(psf)})")
    os.makedirs(a.out_vanilla, exist_ok=True)
    for rep, dcd in ((f"vanilla_{k}", found[k]) for k in (1, 2, 3)):
        stem = os.path.join(a.out_vanilla, rep)
        if os.path.exists(stem + "_chi.npz") and os.path.exists(stem + "_wrapped.dcd") and not a.force:
            print(f"[{rep}] exists, skipped")
            continue
        t0 = time.time()
        u = mda.Universe(psf, dcd)
        n_all = u.trajectory.n_frames
        if n_all < a.frames:
            sys.exit(f"{dcd}: {n_all} frames < --frames {a.frames}")
        print(f"[{rep}] {os.path.basename(dcd)}: {n_all} frames, header dt {u.trajectory.dt:g} ps; "
              f"using the first {a.frames} at {a.dt_ns:g} ns", flush=True)
        protein = u.select_atoms("protein")
        keep = u.select_atoms(a.keep)
        if u.dimensions is not None:
            wf = [trans.unwrap(keep), trans.center_in_box(protein, center="mass")]
            nonprot = keep - protein
            if nonprot.n_atoms:
                wf.append(trans.wrap(nonprot, compound="fragments"))
            u.trajectory.add_transformations(*wf)
        idx, labels = prep.build_chi_index(protein)
        i0, i1, i2, i3 = (idx[:, k] for k in range(4))
        keep.write(stem + "_wrapped.pdb", frames=[0])
        ang = np.empty((a.frames, len(labels)), np.float32)
        with mda.Writer(stem + "_wrapped.dcd", keep.n_atoms, dt=1000.0 * a.dt_ns) as w:
            for i, ts in enumerate(u.trajectory[:a.frames]):
                x = u.atoms.positions
                ang[i] = np.degrees(calc_dihedrals(x[i0], x[i1], x[i2], x[i3]))
                w.write(keep)
                if i and i % 300 == 0:
                    print(f"  [{rep}] {i}/{a.frames}", flush=True)
        np.savez_compressed(stem + "_chi.npz", angles=ang,
                            resid=np.array([l[0] for l in labels], np.int32),
                            resname=np.array([l[1] for l in labels]),
                            chi=np.array([l[2] for l in labels], np.int8),
                            time_ns=np.arange(a.frames, dtype=np.float64) * a.dt_ns,
                            segment_start=np.array([0], np.int64),
                            segment_name=np.array([os.path.basename(dcd)]))
        print(f"[{rep}] chi angles and wrapped trajectory written, {time.time() - t0:.0f} s", flush=True)
    py = sys.executable
    if a.force or not all(os.path.exists(os.path.join(a.out_vanilla, f"{r}_sasa.npz")) for r in VAN_REPS):
        run([py, os.path.join(HERE, "kras_hmr_sasa.py"), "--indir", a.out_vanilla, "--reps", *VAN_REPS,
             "--dt-ns", str(a.dt_ns), "--nproc", str(a.nproc)])
    if a.force or not all(os.path.exists(os.path.join(a.out_vanilla, f"{r}_hbond.npz")) for r in VAN_REPS):
        run([py, os.path.join(HERE, "kras_hmr_hbond.py"), "--indir", a.out_vanilla, "--reps", *VAN_REPS,
             "--dt-ns", str(a.dt_ns), "--nproc", str(a.nproc)])


def cut_hmr(a):
    os.makedirs(a.out_hmr15, exist_ok=True)
    for rep in HMR_REPS:
        for kind in KINDS:
            src = os.path.join(a.hmr_dir, f"{rep}_{kind}.npz")
            dst = os.path.join(a.out_hmr15, f"{rep}_{kind}.npz")
            z = np.load(src, allow_pickle=True)
            T = len(z["time_ns"])
            if T < a.frames:
                sys.exit(f"{src}: {T} frames < {a.frames}")
            out = {}
            for k in z.files:
                v = z[k]
                out[k] = v[:a.frames] if (v.ndim >= 1 and v.shape[0] == T) else v
            if kind == "hbond":
                out["occupancy"] = out["hb"].mean(axis=0).astype(np.float32)
            np.savez_compressed(dst, **out)
        print(f"[{rep}] HMR signals cut to {a.frames} frames -> {a.out_hmr15}", flush=True)


def check_match(a):
    for kind, keys in (("chi", ("resid", "chi")), ("sasa", ("resid",))):
        h = np.load(os.path.join(a.out_hmr15, f"{HMR_REPS[0]}_{kind}.npz"), allow_pickle=True)
        v = np.load(os.path.join(a.out_vanilla, f"{VAN_REPS[0]}_{kind}.npz"), allow_pickle=True)
        for k in keys:
            if not np.array_equal(h[k], v[k]):
                sys.exit(f"{kind}: '{k}' differs between HMR and vanilla signals (PSF numbering or names)")
    print("chi angles and SASA residues match between the two data sets", flush=True)


def cmd_signals(a):
    build_vanilla(a)
    cut_hmr(a)
    check_match(a)


# ============================================================================ kinetics
def acf_stats(Z, dt, cap_ns):
    """normalised autocorrelation of each column (complex input allowed): integrated
    autocorrelation time dt*(1/2 + sum rho(k)) summed until rho first drops below 0 (at most
    cap_ns), and rho at 10 ns and 100 ns. Unlike a 1/e crossing, these do not stop at the
    1 ns frame spacing."""
    Z = Z - Z.mean(axis=0)
    n = Z.shape[0]
    F = np.fft.fft(Z, n=2 * n, axis=0)
    ac = (np.fft.ifft(F * np.conj(F), axis=0)[:n].real) / np.arange(n, 0, -1)[:, None]
    ac = ac / np.where(ac[0] > 0, ac[0], np.nan)
    m = min(n - 1, int(cap_ns / dt))
    r = ac[1:m + 1]
    neg = r < 0
    stop = np.where(neg.any(axis=0), neg.argmax(axis=0), m)           # lags summed before the first negative
    csum = np.vstack([np.zeros((1, r.shape[1])), np.cumsum(r, axis=0)])
    tau = dt * (0.5 + csum[stop, np.arange(r.shape[1])])
    k10, k100 = int(round(10 / dt)), int(round(100 / dt))
    return tau, ac[k10], ac[k100]


def load_set(d, reps):
    S = {}
    for rep in reps:
        S[rep] = {k: np.load(os.path.join(d, f"{rep}_{k}.npz"), allow_pickle=True) for k in KINDS}
    return S


def per_signal(S, reps, dt, cap):
    rows = []
    for rep in reps:
        c = S[rep]["chi"]
        th = np.radians(c["angles"].astype(float))
        dbl = np.array([(r, int(k)) in SYMMETRIC for r, k in zip(c["resname"].astype(str), c["chi"])])
        th[:, dbl] *= 2
        z = np.exp(1j * th)
        tau, r10, r100 = acf_stats(z, dt, cap)
        R = np.abs(z.mean(axis=0))
        for j in range(th.shape[1]):
            rows.append(dict(obs="chi", key=f"{int(c['resid'][j])}:chi{int(c['chi'][j])}", rep=rep,
                             tau_ns=tau[j], acf10=r10[j], acf100=r100[j], level=R[j], sd=np.nan, rate=np.nan))
        s = S[rep]["sasa"]
        X = s["sasa"].astype(float)
        tau, r10, r100 = acf_stats(X, dt, cap)
        for j in range(X.shape[1]):
            rows.append(dict(obs="sasa", key=str(int(s["resid"][j])), rep=rep, tau_ns=tau[j], acf10=r10[j],
                             acf100=r100[j], level=X[:, j].mean(), sd=X[:, j].std(), rate=np.nan))
        h = S[rep]["hbond"]
        M = h["hb"].astype(float)
        tau, r10, r100 = acf_stats(M, dt, cap)
        form = ((M[1:] == 1) & (M[:-1] == 0)).sum(axis=0) / (M.shape[0] * dt / 1000.0)
        for j in range(M.shape[1]):
            a_, b_ = sorted((int(h["res_i"][j]), int(h["res_j"][j])))
            rows.append(dict(obs="hb", key=f"{a_}-{b_}", rep=rep, tau_ns=tau[j], acf10=r10[j], acf100=r100[j],
                             level=M[:, j].mean(), sd=np.nan, rate=form[j]))
    return pd.DataFrame(rows)


def switches(A, B, gap):
    """switches from bond A to bond B and back, allowing <= gap frames with neither in between"""
    st = np.where((A == 1) & (B == 0), 1, np.where((B == 1) & (A == 0), 2, 0))
    n, last, since = 0, 0, 0
    for s in st:
        if s == 0:
            since += 1
            continue
        if last and s != last and since <= gap:
            n += 1
        last, since = s, 0
    return n


def core_hbonds(S, reps, method, dt):
    rows = []
    for rep in reps:
        h = S[rep]["hbond"]
        cols = {tuple(sorted((int(i), int(j)))): k for k, (i, j) in enumerate(zip(h["res_i"], h["res_j"]))}
        M = h["hb"]
        for name, (p, q) in HB_PAIRS.items():
            if p not in cols or q not in cols:
                rows.append(dict(method=method, rep=rep, pair=name, note="bond not formed"))
                continue
            A, B = M[:, cols[p]].astype(int), M[:, cols[q]].astype(int)
            oa, ob, both = A.mean(), B.mean(), (A & B).mean()
            rows.append(dict(method=method, rep=rep, pair=name, occ_first=round(oa, 3), occ_second=round(ob, 3),
                             both=round(both, 3), both_vs_independent=round(both / (oa * ob), 2) if oa * ob else np.nan,
                             switches_per_us=round(switches(A, B, int(round(5 / dt))) / (len(A) * dt / 1000), 1),
                             note=""))
    return rows


def spectra(S, reps, dt):
    from scipy.signal import welch
    out = {}
    for rep in reps:
        c = S[rep]["chi"]
        th = np.radians(c["angles"].astype(float))
        dbl = np.array([(r, int(k)) in SYMMETRIC for r, k in zip(c["resname"].astype(str), c["chi"])])
        th[:, dbl] *= 2
        sig = {"chi": np.concatenate([np.cos(th), np.sin(th)], axis=1), "sasa": S[rep]["sasa"]["sasa"].astype(float)}
        M = S[rep]["hbond"]["hb"].astype(float)
        occ = M.mean(axis=0)
        sig["hb"] = M[:, (occ >= 0.05) & (occ <= 0.95)]
        for o, X in sig.items():
            X = X[:, X.std(axis=0) > 1e-9]
            f, P = welch(X, fs=1.0 / dt, nperseg=min(512, X.shape[0]), axis=0, detrend="linear")
            P = P / P[1:].sum(axis=0, keepdims=True)
            out.setdefault(o, []).append(np.exp(np.log(P[1:] + 1e-30).mean(axis=1)))
            out[o + "_f"] = f[1:]
    return out


def cmd_kinetics(a):
    os.makedirs(a.out, exist_ok=True)
    H, V = load_set(a.out_hmr15, HMR_REPS), load_set(a.out_vanilla, VAN_REPS)
    dt = a.dt_ns
    ph = per_signal(H, HMR_REPS, dt, a.cap_ns).assign(method="HMR")
    pv = per_signal(V, VAN_REPS, dt, a.cap_ns).assign(method="vanilla")
    P = pd.concat([ph, pv], ignore_index=True)
    P.to_csv(os.path.join(a.out, "per_signal_replica.csv"), index=False, float_format="%.5g")

    # signals kept for the comparison
    keep = []
    for o, d in P.groupby("obs"):
        g = d.groupby("key")
        if o == "sasa":
            ok = g["sd"].min() >= 1.0
        elif o == "hb":
            ok = g["level"].mean().between(0.05, 0.95) & (g["rep"].nunique() == 6)
        else:
            ok = g["level"].min() < 0.98                 # angles that move at all
        keep += [(o, k) for k in ok[ok].index]
    K = set(keep)
    P = P[[(o, k) in K for o, k in zip(P.obs, P.key)]]

    # method means per signal, and replica spread within each method
    lines, rows = [], []
    rng = np.random.default_rng(1)
    np.seterr(divide="ignore", invalid="ignore")
    for o, d in P.groupby("obs"):
        for metric in (["tau_ns", "acf10", "acf100", "level"] + (["sd"] if o == "sasa" else [])
                       + (["rate"] if o == "hb" else [])):
            w = d.pivot_table(index="key", columns=["method", "rep"], values=metric, aggfunc="first")
            mh = w["HMR"].mean(axis=1)
            mv = w["vanilla"].mean(axis=1)
            both = mh.notna() & mv.notna() & (mh > 0) & (mv > 0) if metric in ("tau_ns", "rate", "sd") else \
                mh.notna() & mv.notna()
            mh, mv = mh[both], mv[both]
            if not len(mh):
                continue
            if metric in ("tau_ns", "rate", "sd"):
                lr = np.log(mv / mh)
                boots = [np.median(rng.choice(lr.values, len(lr))) for _ in range(2000)]
                # replica spread within each method: |log ratio| between replicas of one method
                within = []
                for m_ in ("HMR", "vanilla"):
                    x = w[m_].loc[mh.index]
                    for i in range(3):
                        for j in range(i + 1, 3):
                            r_ = np.log(x.iloc[:, i] / x.iloc[:, j]).replace([np.inf, -np.inf], np.nan).dropna()
                            within.append(np.median(np.abs(r_)))
                xr = []
                for i in range(3):
                    for j in range(3):
                        r_ = np.log(w["vanilla"].loc[mh.index].iloc[:, j] / w["HMR"].loc[mh.index].iloc[:, i])
                        xr.append(np.median(np.abs(r_.replace([np.inf, -np.inf], np.nan).dropna())))
                rows.append(dict(observable=o, metric=metric, signals=len(lr),
                                 median_ratio_vanilla_over_HMR=round(float(np.exp(np.median(lr))), 3),
                                 ci95=f"{np.exp(np.percentile(boots, 2.5)):.3f}-{np.exp(np.percentile(boots, 97.5)):.3f}",
                                 replica_spread_within=round(float(np.exp(np.mean(within))), 3),
                                 replica_spread_between=round(float(np.exp(np.mean(xr))), 3),
                                 censored_HMR=int(w["HMR"].isna().any(axis=1).sum()) if metric == "tau_ns" else "",
                                 censored_vanilla=int(w["vanilla"].isna().any(axis=1).sum()) if metric == "tau_ns" else ""))
            else:
                r = float(np.corrcoef(mh, mv)[0, 1])
                within = []
                for m_ in ("HMR", "vanilla"):
                    x = w[m_].loc[mh.index]
                    for i in range(3):
                        for j in range(i + 1, 3):
                            within.append(float(np.corrcoef(x.iloc[:, i], x.iloc[:, j])[0, 1]))
                xr = [float(np.corrcoef(w["vanilla"].loc[mh.index].iloc[:, j], w["HMR"].loc[mh.index].iloc[:, i])[0, 1])
                      for i in range(3) for j in range(3)]
                rows.append(dict(observable=o, metric=metric, signals=len(mh),
                                 corr_method_means=round(r, 3),
                                 mean_abs_diff=round(float((mv - mh).abs().mean()), 4),
                                 replica_corr_within=round(float(np.mean(within)), 3),
                                 replica_corr_between=round(float(np.mean(xr)), 3)))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(a.out, "kinetics_summary.csv"), index=False)
    T_time = T[T.metric.isin(["tau_ns", "rate", "sd"])].dropna(axis=1, how="all")
    T_level = T[T.metric.isin(["level", "acf10", "acf100"])].dropna(axis=1, how="all")

    C = pd.DataFrame(core_hbonds(H, HMR_REPS, "HMR", dt) + core_hbonds(V, VAN_REPS, "vanilla", dt))
    C.to_csv(os.path.join(a.out, "core_hbonds.csv"), index=False)

    SP = {"HMR": spectra(H, HMR_REPS, dt), "vanilla": spectra(V, VAN_REPS, dt)}
    sp_rows = []
    for o in ("chi", "sasa", "hb"):
        f = SP["HMR"][o + "_f"]
        ph_, pv_ = np.mean(SP["HMR"][o], axis=0), np.mean(SP["vanilla"][o], axis=0)
        for fi, x, y in zip(f, ph_, pv_):
            sp_rows.append(dict(observable=o, period_ns=1 / fi, power_HMR=x, power_vanilla=y, ratio=y / x))
    SPD = pd.DataFrame(sp_rows)
    SPD.to_csv(os.path.join(a.out, "spectra.csv"), index=False, float_format="%.5g")
    plot_kinetics(P, SPD, os.path.join(a.out, "kinetics.png"))

    with pd.option_context("display.width", 220, "display.max_columns", 20):
        rep = ["HMR (first {:g} us) against 2 fs without HMR, {} frames at {:g} ns".format(
            a.frames * dt / 1000, a.frames, dt), "",
            "Ratios are vanilla / HMR, medians over signals (1 = same).",
            "replica_spread_*: typical ratio between two replicas, within one method and across methods.",
            "If 'between' is close to 'within', the methods differ no more than replicas do.", "",
            "Timescales (tau_ns: integrated autocorrelation time; rate: H-bond formations per us; sd: SASA SD):",
            T_time.to_string(index=False), "",
            "Per-signal values compared by correlation over signals (level: chi mean resultant length, SASA mean,",
            "H-bond occupancy; acf10 / acf100: autocorrelation at 10 and 100 ns):",
            T_level.to_string(index=False), "", "Core H-bond pairs:", C.to_string(index=False), "",
            "Spectra: vanilla / HMR power ratio, median over periods in each range:"]
        for o in ("chi", "sasa", "hb"):
            d = SPD[SPD.observable == o]
            for lo, hi in ((2, 10), (10, 50), (50, 250)):
                x = d[(d.period_ns >= lo) & (d.period_ns < hi)].ratio
                if len(x):
                    rep.append(f"  {o:4s} {lo:>3}-{hi:<3} ns: {np.median(x):.2f}")
    open(os.path.join(a.out, "kinetics_report.txt"), "w").write("\n".join(rep) + "\n")
    print("\n".join(rep))


def plot_kinetics(P, SPD, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"chi": "#b83b72", "sasa": "#2a9d8f", "hb": "#d08c1f"}
    name = {"chi": "χ", "sasa": "SASA", "hb": "H-bond"}
    fig, ax = plt.subplots(2, 3, figsize=(10, 6.2))
    for i, o in enumerate(("chi", "sasa", "hb")):
        d = P[P.obs == o]
        w = d.groupby(["key", "method"]).tau_ns.mean().unstack()
        w = w.dropna()
        x = ax[0, i]
        x.loglog(w["HMR"], w["vanilla"], "o", ms=2.5, color=col[o], alpha=0.6)
        lim = [max(0.8, w.values.min() * 0.8), w.values.max() * 1.25]
        x.plot(lim, lim, color="#888", lw=0.8)
        x.set_xlim(lim)
        x.set_ylim(lim)
        x.set_xlabel("autocorrelation time, HMR (ns)")
        x.set_ylabel("autocorrelation time, 2 fs (ns)")
        x.set_title(f"{name[o]} ({len(w)} signals)", fontsize=9)
        s = SPD[SPD.observable == o]
        y = ax[1, i]
        y.loglog(s.period_ns, s.power_HMR, color=col[o], lw=1.2, label="HMR")
        y.loglog(s.period_ns, s.power_vanilla, color="#333", lw=1.0, ls="--", label="2 fs")
        y.axvspan(80, 120, color="#eee", lw=0)
        y.set_xlabel("period (ns)")
        y.set_ylabel("normalised power")
        y.legend(frameon=False, fontsize=8)
        for z in (x, y):
            z.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


# ============================================================================ coherence
def newest(pattern):
    c = glob.glob(pattern)
    return max(c, key=os.path.getmtime) if c else None


def coherence_runs(a, d, reps):
    py = sys.executable
    for bn in a.bands:
        b = BANDS[bn]
        common = ["--indir", d, "--reps", *reps, "--pmin", str(b["pmin"]), "--pmax", str(b["pmax"]),
                  "--thr", str(a.thr), "--min-ep", str(b["ep"]), "--tsmooth", "2", "--tdec", str(b["tdec"]),
                  "--dj", str(1 / 12)]
        tag = f"P{b['pmin']}-{b['pmax']}_thr{a.thr:g}_ep{b['ep']}_ts2_"
        if a.force or not newest(os.path.join(d, tag + "*", "residue_pairs.csv")):
            run([py, os.path.join(HERE, "kras_hmr_wco.py"), *common])
        if a.force or not newest(os.path.join(d, "SASA_" + tag + "*sep3*", "edges.csv")):
            run([py, os.path.join(HERE, "kras_hmr_sasa_wco.py"), *common, "--min-seqsep", "3", "--weak-sd", "1"])
        if a.force or not newest(os.path.join(d, "HB_" + tag + "*pf0.05*", "hyperedges.csv")):
            run([py, os.path.join(HERE, "kras_hmr_hbond_wco.py"), *common, "--occ-min", "0.05", "--occ-max", "0.95",
                 "--min-trans", "10", "--pow-floor", "0.05"])


def read_networks(d, b, thr):
    tag = f"P{b['pmin']}-{b['pmax']}_thr{thr:g}_ep{b['ep']}_ts2_"
    out = {}
    f = newest(os.path.join(d, tag + "*", "residue_pairs.csv"))
    x = pd.read_csv(f)
    out["chi"] = {(int(i), int(j)): int(n) for i, j, n in zip(x.resid_i, x.resid_j, x.n_reps_any) if n > 0}
    f = newest(os.path.join(d, "SASA_" + tag + "*sep3*", "edges.csv"))
    x = pd.read_csv(f)
    out["sasa"] = {(int(i), int(j)): int(n) for i, j, n in zip(x.resid_i, x.resid_j, x.n_reps_any) if n > 0}
    f = newest(os.path.join(d, "HB_" + tag + "*pf0.05*", "hyperedges.csv"))
    x = pd.read_csv(f)
    hb = {}
    for h1, h2, n in zip(x.hb1, x.hb2, x.n_reps):
        p = tuple(sorted(map(int, str(h1).split("-"))))
        q = tuple(sorted(map(int, str(h2).split("-"))))
        hb[frozenset((p, q))] = int(n)
    out["hb"] = hb
    return out


def cmd_coherence(a):
    os.makedirs(a.out, exist_ok=True)
    coherence_runs(a, a.out_hmr15, HMR_REPS)
    coherence_runs(a, a.out_vanilla, VAN_REPS)
    rows, core = [], []
    for bn in a.bands:
        b = BANDS[bn]
        NH, NV = read_networks(a.out_hmr15, b, a.thr), read_networks(a.out_vanilla, b, a.thr)
        for o in ("chi", "sasa", "hb"):
            h, v = NH[o], NV[o]
            row = dict(band=f"{b['pmin']}-{b['pmax']} ns", observable=o)
            for m, n in (("HMR", h), ("vanilla", v)):
                for k in (1, 2, 3):
                    row[f"{m}_ge{k}" if k < 3 else f"{m}_3"] = sum(1 for x in n.values() if x >= k)
            for k in (1, 2):
                hs = {p for p, x in h.items() if x >= k}
                vs = {p for p, x in v.items() if x >= k}
                row[f"shared_ge{k}"] = len(hs & vs)
                row[f"jaccard_ge{k}"] = round(len(hs & vs) / len(hs | vs), 3) if hs | vs else np.nan
            rows.append(row)
            for p in CORE[o]:
                key = frozenset(p) if o == "hb" else p
                lab = " | ".join(f"{x}-{y}" for x, y in p) if o == "hb" else f"{p[0]}-{p[1]}"
                core.append(dict(band=f"{b['pmin']}-{b['pmax']} ns", observable=o, pair=lab,
                                 HMR_replicas=h.get(key, 0), vanilla_replicas=v.get(key, 0)))
    R, CP = pd.DataFrame(rows), pd.DataFrame(core)
    R.to_csv(os.path.join(a.out, "coherence_counts.csv"), index=False)
    CP.to_csv(os.path.join(a.out, "coherence_core_pairs.csv"), index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        txt = ["Coherence on the first {:g} us, HMR against 2 fs without HMR (no controls; descriptive)".format(
            a.frames * a.dt_ns / 1000), "", R.to_string(index=False), "", "Core pairs (replicas):",
            CP.to_string(index=False)]
    open(os.path.join(a.out, "coherence_report.txt"), "w").write("\n".join(txt) + "\n")
    print("\n".join(txt))


# ============================================================================ main
PROTECTED = ("wco_hmr", "wco_hmr_null", "wco_hmr_null_reswise", "wlcc_band", "wlcc_broad", "vanilla_MD_results",
             "KRAS_unbound_mg_gtp_1")


def guard(a):
    """every output folder must be new and separate from the inputs and the existing analyses"""
    outs = {"--out-vanilla": a.out_vanilla, "--out-hmr15": a.out_hmr15, "--out": a.out}
    ins = [a.hmr_dir, a.vanilla_dir] + list(PROTECTED)
    real = lambda x: os.path.realpath(x)
    for flag, d in outs.items():
        for x in ins:
            if real(d) == real(x) or real(d).startswith(real(x) + os.sep):
                sys.exit(f"{flag} {d} is (inside) {x}, an input or existing analysis folder; choose another name")
    if len({real(d) for d in outs.values()}) != 3:
        sys.exit("--out-vanilla, --out-hmr15 and --out must be three different folders")
    print("outputs: vanilla signals -> {}/ ; HMR signals cut -> {}/ ; reports -> {}/".format(
        a.out_vanilla, a.out_hmr15, a.out), flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["signals", "kinetics", "coherence", "all"])
    p.add_argument("--vanilla-dir", default="vanilla_MD_results")
    p.add_argument("--psf", default="KRAS_unbound_mg_gtp.psf")
    p.add_argument("--dcd-glob", default="KRAS_unbound_mg_gtp_*_wrapped_stride10.dcd")
    p.add_argument("--hmr-dir", default="wco_hmr")
    p.add_argument("--out-vanilla", default=None, help="default wco_vanilla_first<N>ns")
    p.add_argument("--out-hmr15", default=None, help="default wco_hmr_first<N>ns")
    p.add_argument("--out", default=None, help="default vanilla_vs_hmr_first<N>ns")
    p.add_argument("--frames", type=int, default=1500)
    p.add_argument("--dt-ns", type=float, default=1.0)
    p.add_argument("--keep", default="protein or resname GTP MG")
    p.add_argument("--nproc", type=int, default=3)
    p.add_argument("--cap-ns", type=float, default=500.0, help="longest decorrelation time searched")
    p.add_argument("--bands", nargs="+", default=["P20", "P40"], choices=list(BANDS))
    p.add_argument("--thr", type=float, default=0.75)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    span = f"first{a.frames * a.dt_ns:g}ns"
    a.out_vanilla = a.out_vanilla or f"wco_vanilla_{span}"
    a.out_hmr15 = a.out_hmr15 or f"wco_hmr_{span}"
    a.out = a.out or f"vanilla_vs_hmr_{span}"
    guard(a)
    if a.step in ("signals", "all"):
        cmd_signals(a)
    if a.step in ("kinetics", "all"):
        cmd_kinetics(a)
    if a.step in ("coherence", "all"):
        cmd_coherence(a)


if __name__ == "__main__":
    main()
