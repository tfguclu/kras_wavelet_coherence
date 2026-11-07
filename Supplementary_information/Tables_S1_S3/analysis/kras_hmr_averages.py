#!/usr/bin/env python
"""
kras_hmr_averages.py -- what does coherence add to whole-trajectory averages? (Section 3.7)

For every tested signal pair at baseline (80-120 ns, coherence 0.75, 500 ns), the
zero-lag squared correlation over the analysed window of each replica is computed:
  SASA      r^2 between the two residue SASA series
  H-bond    phi^2 between the two binary H-bond signals (Pearson on 0/1)
  chi       time-domain RV coefficient of the two (cos, sin) signals (the zero-lag,
            all-frequency analogue of the wavelet RV coherence); residue pair = max
            over its chi-angle pairs (same "any chi combination" rule as the network)
Every association found at baseline is then placed in the distribution of all tested
pairs of its observable (percentile per replica) and set against its coherent coverage
(fraction of the analysed window inside coherent episodes).

For associations in >= 2 replicas, band coherence R(t) (80-120 ns) is recomputed for
the pair alone with the unchanged pipeline functions (and checked against mean_R in
the run's pair tables), together with the raw signals. For the example candidates a
broad-band coherence map (10-500 ns) is computed as well.

Inputs : wco_hmr/{rep}_chi.npz, {rep}_sasa.npz, {rep}_hbond.npz and the baseline runs
         (located and validated through kras_hmr_runcheck; --accept-legacy-real as before)
Output : wco_hmr/averages/
  pairs_r2_<obs>.csv     every tested pair x replica: r2 (and signed r for SASA/H-bond)
  associations.csv       every baseline association: replicas, per replica r2, percentile,
                         coverage fraction, recomputed mean_R and the run's mean_R (check)
  summary.csv            per observable: percentiles of reproducible associations, overlap
                         of reproducible associations with the top pairs by r2
  examples.npz           R(t) per replica, valid mask, episodes and raw signals for every
                         association in >= 2 replicas
  maps.npz               10-500 ns coherence maps for the example candidates
  candidates.txt         suggested examples (high-r2 recurrent / low-r2 recurrent)

  python kras_hmr_averages.py --accept-legacy-real
  python kras_hmr_averages.py --accept-legacy-real --map-extra "hb:69-102|99-102" "sasa:75-104"
  python kras_hmr_averages.py --selftest
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kras_hmr_runcheck as rc  # noqa: E402
from kras_hmr_wco import (FF, make_scales, cwt_batch, Smoother, self_D, pair_R,  # noqa: E402
                          episodes, chi_to_signals)
from kras_hmr_sensitivity import load_edges  # noqa: E402

BASE = dict(pmin=80, pmax=120, thr=0.75, ep=500, tdec=10)
TSMOOTH, DJ, POW_FLOOR, WEAK_SD, MIN_SEP = rc.TSMOOTH, rc.DJ, 0.05, 1.0, 3
OCC, MIN_TRANS = (0.05, 0.95), 10


# ----------------------------------------------------------------------------- helpers
def band_setup(T, dt, tdec, pmin=80, pmax=120):
    scales = make_scales(pmin, pmax, DJ, 0.3)
    periods = scales * FF
    band = (periods >= pmin) & (periods <= pmax)
    Td = len(range(0, T, tdec))
    sm = Smoother(scales, Td, dt * tdec, DJ, TSMOOTH)
    kmax = min(np.where(band)[0].max() + sm.nbox // 2, len(scales) - 1)
    edge = np.sqrt(2) * scales[kmax] + 2 * TSMOOTH * scales[kmax]
    t = np.arange(Td) * dt * tdec
    valid = (t >= edge) & (t <= (T - 1) * dt - edge)
    fvalid = np.zeros(T, bool)
    fvalid[int(np.ceil(edge / dt)):int((T - 1) - np.ceil(edge / dt)) + 1] = True
    return dict(scales=scales, periods=periods, band=band, sm=sm, valid=valid,
                fvalid=fvalid, edge=edge, dtd=dt * tdec, Td=Td)


def rv0_matrix(sig):
    """sig (n, 2, T) -> zero-lag RV coefficient between all pairs (n, n)."""
    n, _, T = sig.shape
    X = (sig - sig.mean(axis=2, keepdims=True)).reshape(2 * n, T)
    S = X @ X.T / T
    S2 = (S ** 2).reshape(n, 2, n, 2).sum(axis=(1, 3))
    d = np.sqrt(np.diag(S2))
    return S2 / np.outer(d, d)


def pct_rank(values, x):
    """percentile of x within values (fraction of values <= x) x 100."""
    v = np.sort(values[np.isfinite(values)])
    return 100.0 * np.searchsorted(v, x, side="right") / len(v) if len(v) and np.isfinite(x) else np.nan


def transitions(x):
    return int(((x[1:] == 1) & (x[:-1] == 0)).sum())


def cov_frac_from_eps(ep, valid_ns):
    return sum(l for *_, l in ep) / valid_ns


# ----------------------------------------------------------------------------- coherence of one pair
def pair_band_chi(x1, x2, B, tdec):
    """x1, x2: (2, T) cos/sin -> band R(t), full R(s,t) on the decimated grid."""
    W = cwt_batch(np.vstack([x1, x2]), 1.0 * B["dt"], B["scales"])[:, :, ::tdec]
    W = W.reshape(2, 2, len(B["scales"]), -1)
    D, _ = self_D(W, B["sm"])
    R = pair_R(W[0], W[1:2], D[0], D[1:2], B["sm"])[0]
    return R[B["band"]].mean(axis=0), R


def pair_band_scalar(x1, x2, B, tdec, floor=None):
    """scalar signals; floor: H-bond local power floor (fraction of median in-band power)."""
    W = cwt_batch(np.vstack([x1, x2]).astype(np.float64), 1.0 * B["dt"], B["scales"])[:, :, ::tdec]
    sm = B["sm"]
    P = np.maximum(sm(np.abs(W) ** 2).real, 0.0)
    C = sm(W[0][None] * np.conj(W[1][None]))[0]
    R = np.abs(C) ** 2 / (P[0] * P[1] + 1e-30)
    if floor is not None:   # same rule as kras_hmr_hbond_wco.py, per signal over band cells
        Pb = P[:, B["band"], :]
        f = [floor * np.median(Pb[k][:, B["valid"]]) for k in range(2)]
        ok = (Pb[0] >= f[0]) & (Pb[1] >= f[1])
        Rb = np.where(ok, R[B["band"]], 0.0).mean(axis=0)
        # map: per-scale floor (median at each scale over valid time)
        fm = [floor * np.median(P[k][:, B["valid"]], axis=1)[:, None] for k in range(2)]
        R = np.where((P[0] >= fm[0]) & (P[1] >= fm[1]), R, 0.0)
        return Rb, R
    return R[B["band"]].mean(axis=0), R


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="wco_hmr")
    p.add_argument("--reps", nargs="+", default=["namd_3", "namd_3_2", "namd_3_3"])
    p.add_argument("--accept-legacy-real", action="store_true")
    p.add_argument("--map-pmin", type=float, default=10.0)
    p.add_argument("--map-pmax", type=float, default=500.0)
    p.add_argument("--n-candidates", type=int, default=3)
    p.add_argument("--map-extra", nargs="*", default=[],
                   help="extra map pairs, e.g. 'sasa:75-104' 'chi:99-102' 'hb:69-102|99-102'")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    a.indir = os.path.abspath(a.indir)
    out = os.path.join(a.indir, "averages"); os.makedirs(out, exist_ok=True)
    tdec = BASE["tdec"]

    # ---------------- baseline runs
    runs = {}
    for o in ["chi", "sasa", "hb"]:
        h = rc.input_hashes(a.indir, o, a.reps)
        d = rc.find_valid(o, BASE, a.indir, a.reps, rc.expectation(o, BASE, a.reps, h),
                          accept_legacy=a.accept_legacy_real)
        if not d:
            sys.exit(f"no valid baseline run for {o} (use --accept-legacy-real for legacy outputs)")
        runs[o] = d
        print(f"[{o}] baseline: {d}")

    A, R2, EX, MAPREQ = [], {}, {}, []
    valid_ns = {}

    # ================================================================ chi
    Z = {r: np.load(os.path.join(a.indir, f"{r}_chi.npz")) for r in a.reps}
    resid, resname, chi = Z[a.reps[0]]["resid"], Z[a.reps[0]]["resname"], Z[a.reps[0]]["chi"]
    rows = []
    sigs, Bs = {}, {}
    for r in a.reps:
        ang = Z[r]["angles"]; T = ang.shape[0]
        dt = float(np.diff(Z[r]["time_ns"])[0])
        B = band_setup(T, dt, tdec); B["dt"] = dt; Bs[r] = B
        valid_ns[r] = B["valid"].sum() * B["dtd"]
        sig = chi_to_signals(ang, resname, chi)          # (n, 2, T)
        sigs[r] = sig
        M = rv0_matrix(sig[:, :, B["fvalid"]])
        iu = np.triu_indices(len(resid), 1)
        keep = resid[iu[0]] != resid[iu[1]]
        df = pd.DataFrame(dict(resid_i=resid[iu[0]][keep], chi_i=chi[iu[0]][keep],
                               resid_j=resid[iu[1]][keep], chi_j=chi[iu[1]][keep], r2=M[iu][keep]))
        df["replica"] = r
        rows.append(df)
    P = pd.concat(rows)
    P.to_csv(os.path.join(out, "pairs_r2_chi_angles.csv"), index=False, float_format="%.5g")
    Pres = P.groupby(["replica", "resid_i", "resid_j"], as_index=False).r2.max()
    Pres.to_csv(os.path.join(out, "pairs_r2_chi.csv"), index=False, float_format="%.5g")
    R2["chi"] = Pres
    rp = pd.read_csv(os.path.join(runs["chi"], "residue_pairs.csv"))
    rp = rp[rp.n_reps_any > 0]
    pairs_tab = {r: pd.read_csv(os.path.join(runs["chi"], f"{r}_pairs.csv"),
                                usecols=["resid_i", "chi_i", "resid_j", "chi_j", "mean_R", "n_episodes",
                                         "total_ep_ns"]) for r in a.reps}
    for e in rp.itertuples():
        i, j = int(e.resid_i), int(e.resid_j)
        rec = dict(observable="chi", association=f"{i}-{j}", label=f"{e.resname_i}{i}-{e.resname_j}{j}",
                   replicas=int(e.n_reps_any))
        # representative chi-angle pair: most replicas with an episode, then longest total
        sc = {}
        for r in a.reps:
            t = pairs_tab[r]; t = t[(t.resid_i == i) & (t.resid_j == j) & (t.n_episodes > 0)]
            for x in t.itertuples():
                k = (int(x.chi_i), int(x.chi_j)); n_, tot = sc.get(k, (0, 0.0))
                sc[k] = (n_ + 1, tot + x.total_ep_ns)
        ci, cj = max(sc, key=lambda k: sc[k])
        rec["rep_chi"] = f"chi{ci}-chi{cj}"
        for r in a.reps:
            pr = Pres[(Pres.replica == r)]
            x = pr[(pr.resid_i == i) & (pr.resid_j == j)].r2
            v = float(x.iloc[0]) if len(x) else np.nan
            rec[f"r2_{r}"] = v
            rec[f"pct_{r}"] = pct_rank(pr.r2.values, v)
            cov = getattr(e, f"cov_ns_{r}", 0.0)
            rec[f"cov_frac_{r}"] = float(cov) / valid_ns[r] if np.isfinite(cov) else 0.0
            t = pairs_tab[r]
            m = t[(t.resid_i == i) & (t.chi_i == ci) & (t.resid_j == j) & (t.chi_j == cj)]
            rec[f"mean_R_run_{r}"] = float(m.mean_R.iloc[0]) if len(m) else np.nan
        if rec["replicas"] >= 2:
            ki = np.where((resid == i) & (chi == ci))[0][0]; kj = np.where((resid == j) & (chi == cj))[0][0]
            ex = {}
            for r in a.reps:
                B = Bs[r]
                Rb, _ = pair_band_chi(sigs[r][ki], sigs[r][kj], B, tdec)
                rec[f"mean_R_recomp_{r}"] = float(Rb[B["valid"]].mean())
                ex[r] = dict(Rb=Rb, raw=np.stack([Z[r]["angles"][:, ki], Z[r]["angles"][:, kj]]),
                             t0=float(Z[r]["time_ns"][0]))
            EX[("chi", f"{i}-{j}")] = ex
            MAPREQ.append(("chi", f"{i}-{j}", (ki, kj)))
        A.append(rec)

    # ================================================================ SASA
    Z = {r: np.load(os.path.join(a.indir, f"{r}_sasa.npz")) for r in a.reps}
    resid, resname = Z[a.reps[0]]["resid"], Z[a.reps[0]]["resname"]
    weak = np.zeros(len(resid), bool)
    for r in a.reps:
        weak |= Z[r]["sasa"].astype(np.float64).std(axis=0) < WEAK_SD
    ok = ~weak
    rows = []
    for r in a.reps:
        x = Z[r]["sasa"].astype(np.float64)
        T = x.shape[0]; dt = float(np.diff(Z[r]["time_ns"])[0])
        B = band_setup(T, dt, tdec); B["dt"] = dt; Bs[("sasa", r)] = B
        C = np.corrcoef(x[B["fvalid"]].T)
        iu = np.triu_indices(len(resid), 1)
        keep = (np.abs(resid[iu[0]] - resid[iu[1]]) >= MIN_SEP) & ok[iu[0]] & ok[iu[1]]
        rows.append(pd.DataFrame(dict(replica=r, resid_i=resid[iu[0]][keep], resid_j=resid[iu[1]][keep],
                                      r=C[iu][keep], r2=C[iu][keep] ** 2)))
    Ps = pd.concat(rows); Ps.to_csv(os.path.join(out, "pairs_r2_sasa.csv"), index=False, float_format="%.5g")
    R2["sasa"] = Ps
    ed = pd.read_csv(os.path.join(runs["sasa"], "edges.csv"))
    ptab = {r: pd.read_csv(os.path.join(runs["sasa"], f"{r}_pairs.csv"), usecols=["resid_i", "resid_j", "mean_R"])
            for r in a.reps}
    idx = {int(v): k for k, v in enumerate(resid)}
    for e in ed.itertuples():
        i, j = int(e.resid_i), int(e.resid_j)
        rec = dict(observable="sasa", association=f"{i}-{j}", label=f"{e.resname_i}{i}-{e.resname_j}{j}",
                   replicas=int(e.n_reps_any))
        for r in a.reps:
            pr = Ps[Ps.replica == r]; m = pr[(pr.resid_i == i) & (pr.resid_j == j)]
            v = float(m.r2.iloc[0]) if len(m) else np.nan
            rec[f"r_{r}"] = float(m.r.iloc[0]) if len(m) else np.nan
            rec[f"r2_{r}"] = v; rec[f"pct_{r}"] = pct_rank(pr.r2.values, v)
            rec[f"cov_frac_{r}"] = float(getattr(e, f"cov_ns_{r}", 0.0)) / valid_ns[r]
            t = ptab[r]; m = t[(t.resid_i == i) & (t.resid_j == j)]
            rec[f"mean_R_run_{r}"] = float(m.mean_R.iloc[0]) if len(m) else np.nan
        if rec["replicas"] >= 2:
            ex = {}
            for r in a.reps:
                B = Bs[("sasa", r)]; x = Z[r]["sasa"]
                Rb, _ = pair_band_scalar(x[:, idx[i]], x[:, idx[j]], B, tdec)
                rec[f"mean_R_recomp_{r}"] = float(Rb[B["valid"]].mean())
                ex[r] = dict(Rb=Rb, raw=np.stack([x[:, idx[i]], x[:, idx[j]]]).astype(np.float32),
                             t0=float(Z[r]["time_ns"][0]))
            EX[("sasa", f"{i}-{j}")] = ex
            MAPREQ.append(("sasa", f"{i}-{j}", (idx[i], idx[j])))
        A.append(rec)

    # ================================================================ H-bond
    Z = {r: np.load(os.path.join(a.indir, f"{r}_hbond.npz")) for r in a.reps}
    rows, used = [], {}
    for r in a.reps:
        z = Z[r]; M = z["hb"]
        occ = M.mean(axis=0); trn = np.array([transitions(M[:, k]) for k in range(M.shape[1])])
        use = (occ >= OCC[0]) & (occ <= OCC[1]) & (trn >= MIN_TRANS)
        lab = [tuple(sorted(t)) for t in zip(z["res_i"].tolist(), z["res_j"].tolist())]
        cols = np.where(use)[0]
        used[r] = {lab[k]: k for k in cols}
        T = M.shape[0]; dt = float(np.diff(z["time_ns"])[0])
        B = band_setup(T, dt, tdec); B["dt"] = dt; Bs[("hb", r)] = B
        X = M[:, cols][B["fvalid"]].astype(np.float64)
        C = np.corrcoef(X.T)
        iu = np.triu_indices(len(cols), 1)
        L = [lab[k] for k in cols]
        a1 = [L[u] for u in iu[0]]; a2 = [L[v] for v in iu[1]]
        hb1 = [f"{min(x, y)[0]}-{min(x, y)[1]}" for x, y in zip(a1, a2)]
        hb2 = [f"{max(x, y)[0]}-{max(x, y)[1]}" for x, y in zip(a1, a2)]
        rows.append(pd.DataFrame(dict(replica=r, hb1=hb1, hb2=hb2, r=C[iu], r2=C[iu] ** 2)))
    Ph = pd.concat(rows); Ph.to_csv(os.path.join(out, "pairs_r2_hb.csv"), index=False, float_format="%.5g")
    R2["hb"] = Ph
    hy = pd.read_csv(os.path.join(runs["hb"], "hyperedges.csv"))
    ptab = {r: pd.read_csv(os.path.join(runs["hb"], f"{r}_pairs.csv"), usecols=["hb1", "hb2", "mean_R"])
            for r in a.reps}
    for e in hy.itertuples():
        A1 = tuple(sorted(map(int, e.hb1.split("-")))); B1 = tuple(sorted(map(int, e.hb2.split("-"))))
        A1, B1 = min(A1, B1), max(A1, B1)
        key = f"{A1[0]}-{A1[1]} | {B1[0]}-{B1[1]}"
        rec = dict(observable="hb", association=key, label=key, replicas=int(e.n_reps))
        for r in a.reps:
            pr = Ph[Ph.replica == r]
            m = pr[(pr.hb1 == f"{A1[0]}-{A1[1]}") & (pr.hb2 == f"{B1[0]}-{B1[1]}")]
            v = float(m.r2.iloc[0]) if len(m) else np.nan
            rec[f"r_{r}"] = float(m.r.iloc[0]) if len(m) else np.nan
            rec[f"r2_{r}"] = v; rec[f"pct_{r}"] = pct_rank(pr.r2.values, v)
            rec[f"cov_frac_{r}"] = float(getattr(e, f"cov_ns_{r}", 0.0)) / valid_ns[r]
            t = ptab[r]; m = t[(t.hb1 == f"{A1[0]}-{A1[1]}") & (t.hb2 == f"{B1[0]}-{B1[1]}")]
            rec[f"mean_R_run_{r}"] = float(m.mean_R.iloc[0]) if len(m) else np.nan
        if rec["replicas"] >= 2:
            ex = {}
            for r in a.reps:
                if A1 not in used[r] or B1 not in used[r]:
                    continue                                 # signal filtered out in this replica
                B = Bs[("hb", r)]; M = Z[r]["hb"]
                k1, k2 = used[r][A1], used[r][B1]
                Rb, _ = pair_band_scalar(M[:, k1], M[:, k2], B, tdec, floor=POW_FLOOR)
                rec[f"mean_R_recomp_{r}"] = float(Rb[B["valid"]].mean())
                ex[r] = dict(Rb=Rb, raw=np.stack([M[:, k1], M[:, k2]]).astype(np.uint8),
                             t0=float(Z[r]["time_ns"][0]))
            EX[("hb", key)] = ex
            MAPREQ.append(("hb", key, (A1, B1)))
        A.append(rec)

    # ================================================================ tables
    AS = pd.DataFrame(A)
    r2c = [f"r2_{r}" for r in a.reps]; pc = [f"pct_{r}" for r in a.reps]; cc = [f"cov_frac_{r}" for r in a.reps]
    AS["r2_median"] = AS[r2c].median(axis=1); AS["pct_median"] = AS[pc].median(axis=1)
    AS["cov_frac_max"] = AS[cc].max(axis=1)
    chk = []
    for r in a.reps:
        if f"mean_R_recomp_{r}" in AS:
            chk.append((AS[f"mean_R_recomp_{r}"] - AS[f"mean_R_run_{r}"]).abs())
    AS["mean_R_check_maxdiff"] = pd.concat(chk, axis=1).max(axis=1) if chk else np.nan
    AS = AS.sort_values(["observable", "replicas", "r2_median"], ascending=[True, False, False])
    AS.to_csv(os.path.join(out, "associations.csv"), index=False, float_format="%.4g")
    bad = AS[AS.mean_R_check_maxdiff > 1e-3]
    if len(bad):
        print(f"WARNING: recomputed mean_R differs from the run by > 1e-3 for {len(bad)} associations:")
        print(bad[["observable", "association", "mean_R_check_maxdiff"]].to_string(index=False))

    S = []
    for o in ["chi", "sasa", "hb"]:
        sub = AS[AS.observable == o]; rep = sub[sub.replicas >= 2]; core = sub[sub.replicas == 3]
        Pr = R2[o]
        row = dict(observable=o, tested_pairs_per_replica=int(Pr.groupby("replica").size().median()),
                   n_reproducible=len(rep), n_three=len(core),
                   pct_median_reproducible=rep.pct_median.median(),
                   pct_median_three=core.pct_median.median(),
                   frac_reproducible_in_top1pct=(rep.pct_median >= 99).mean(),
                   frac_reproducible_below_90pct=(rep.pct_median < 90).mean(),
                   cov_frac_median_reproducible=rep[cc].replace(0, np.nan).median(axis=1).median())
        # overlap: top-K pairs by r2 (K = number reproducible) vs reproducible set, per replica
        ov = []
        keyc = ["hb1", "hb2"] if o == "hb" else ["resid_i", "resid_j"]
        repkeys = set(rep.association)
        for r in a.reps:
            pr = Pr[Pr.replica == r].nlargest(len(rep), "r2")
            ks = (pr.hb1 + " | " + pr.hb2) if o == "hb" else (pr.resid_i.astype(str) + "-" + pr.resid_j.astype(str))
            ov.append(len(set(ks) & repkeys))
        row["topK_overlap_per_replica"] = "/".join(map(str, ov))
        row["r2_p99_all"] = float(np.nanpercentile(Pr.r2, 99))
        row["r2_median_all"] = float(np.nanmedian(Pr.r2))
        S.append(row)
    S = pd.DataFrame(S); S.to_csv(os.path.join(out, "summary.csv"), index=False, float_format="%.4g")

    # candidates: recurrent + high r2 / recurrent + low r2 percentile
    rep = AS[AS.replicas >= 2]
    hi = rep[rep.replicas == 3].sort_values("r2_median", ascending=False).head(a.n_candidates)
    lo = rep.sort_values("pct_median").head(a.n_candidates)
    with open(os.path.join(out, "candidates.txt"), "w") as fh:
        for name, tab in [("HIGH whole-trajectory r2 (three-replica core)", hi),
                          ("LOW whole-trajectory r2 percentile (>= 2 replicas)", lo)]:
            fh.write(name + "\n" + tab[["observable", "association", "label", "replicas", "r2_median",
                                        "pct_median", "cov_frac_max"]].to_string(index=False) + "\n\n")
    print(open(os.path.join(out, "candidates.txt")).read())

    # ================================================================ examples.npz
    blob = {}
    for (o, key), ex in EX.items():
        tag = f"{o}|{key}"
        for r, d in ex.items():
            B = Bs[r] if o == "chi" else Bs[(o, r)]
            blob[f"{tag}|{r}|Rb"] = d["Rb"].astype(np.float32)
            blob[f"{tag}|{r}|raw"] = d["raw"]
            blob[f"{tag}|{r}|t_dec"] = (d["t0"] + np.arange(B["Td"]) * B["dtd"]).astype(np.float32)
            blob[f"{tag}|{r}|valid"] = B["valid"]
            ep = episodes((d["Rb"] >= BASE["thr"]) & B["valid"], B["dtd"], rc.eff_ep(BASE))
            blob[f"{tag}|{r}|episodes"] = np.array([[s + d["t0"], e + d["t0"]] for s, e, _ in ep]).reshape(-1, 2)
    blob["meta"] = np.array(json.dumps(dict(base=BASE, tsmooth=TSMOOTH, dj=DJ, pow_floor=POW_FLOOR,
                                            reps=a.reps, runs=runs)))
    np.savez_compressed(os.path.join(out, "examples.npz"), **blob)

    # ================================================================ maps
    want = set(map(tuple, pd.concat([hi, lo])[["observable", "association"]].values))
    for s in a.map_extra:
        o, k = s.split(":", 1)
        if o == "hb":
            x, y = k.split("|"); k = f"{x.strip()} | {y.strip()}"
        want.add((o, k))
    maps = {}
    for o, key, ids in MAPREQ:
        if (o, key) not in want:
            continue
        for r in a.reps:
            if o == "chi":
                z = np.load(os.path.join(a.indir, f"{r}_chi.npz"))
                sig = chi_to_signals(z["angles"], z["resname"], z["chi"])
                T = sig.shape[2]; dt = float(np.diff(z["time_ns"])[0])
            elif o == "sasa":
                z = np.load(os.path.join(a.indir, f"{r}_sasa.npz")); x = z["sasa"]
                T = x.shape[0]; dt = float(np.diff(z["time_ns"])[0])
            else:
                z = np.load(os.path.join(a.indir, f"{r}_hbond.npz"))
                if ids[0] not in used[r] or ids[1] not in used[r]:
                    continue
                T = z["hb"].shape[0]; dt = float(np.diff(z["time_ns"])[0])
            B = band_setup(T, dt, tdec, a.map_pmin, a.map_pmax); B["dt"] = dt
            B["band"] = (B["periods"] >= a.map_pmin) & (B["periods"] <= a.map_pmax)
            if o == "chi":
                _, R = pair_band_chi(sig[ids[0]], sig[ids[1]], B, tdec)
            elif o == "sasa":
                _, R = pair_band_scalar(x[:, ids[0]], x[:, ids[1]], B, tdec)
            else:
                M = z["hb"]
                _, R = pair_band_scalar(M[:, used[r][ids[0]]], M[:, used[r][ids[1]]], B, tdec, floor=POW_FLOOR)
            t = np.arange(B["Td"]) * B["dtd"]
            edge_s = np.sqrt(2) * B["scales"] + 2 * TSMOOTH * B["scales"]   # per-scale edge zone
            coi = (t[None, :] >= edge_s[:, None]) & (t[None, :] <= (T - 1) * dt - edge_s[:, None])
            tag = f"{o}|{key}|{r}"
            maps[f"{tag}|R"] = R.astype(np.float16)
            maps[f"{tag}|coi"] = coi
            maps[f"{tag}|periods"] = B["periods"].astype(np.float32)
            maps[f"{tag}|t"] = (float(z["time_ns"][0]) + t).astype(np.float32)
    np.savez_compressed(os.path.join(out, "maps.npz"), **maps)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(S.to_string(index=False))
    print(f"wrote {out}/")


# ----------------------------------------------------------------------------- selftest
def selftest():
    rng = np.random.default_rng(0)
    T = 4000
    th = np.cumsum(rng.normal(0, 0.2, T))
    s1 = np.stack([np.cos(th), np.sin(th)])
    s2 = np.stack([np.cos(th + 1.3), np.sin(th + 1.3)])          # constant offset
    s3 = np.stack([np.cos(np.cumsum(rng.normal(0, 0.2, T))), np.sin(np.cumsum(rng.normal(0, 0.2, T)))])
    M = rv0_matrix(np.stack([s1, s2, s3]))
    ok = abs(M[0, 1] - 1) < 1e-6 and M[0, 2] < 0.5
    print(f"RV0 identical-with-offset {M[0, 1]:.4f} (expect 1), independent {M[0, 2]:.3f}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    main()
