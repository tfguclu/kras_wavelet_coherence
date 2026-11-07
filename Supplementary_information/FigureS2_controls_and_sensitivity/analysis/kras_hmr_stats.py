#!/usr/bin/env python
"""
kras_hmr_stats.py -- statistics for the manuscript, built on the existing outputs.

Nothing in the analysis pipeline is changed. This script either (a) runs the
existing scan scripts with more surrogate sets, or (b) reads their outputs and
adds the statistics the text now reports.

Subcommands
-----------
  run        more surrogate sets: baseline null (default 200 sets), period-band
             scan and threshold scan (default 20 sets each). Calls the existing
             kras_hmr_coherence_null.py / kras_hmr_multiband.py /
             kras_hmr_threshold.py with --n-surr; sets already on disk are
             reused, so the run can be stopped and restarted.
  chance     network-level chance share ("what fraction of the recurrent network
             would shuffled data produce"): mean surrogate count / real count, with
             the 2.5-97.5 % range over surrogate sets. -> chance_share.csv
  perassoc   per-association empirical p with the current number of sets, plus
             Benjamini-Hochberg q within each observable and over all.
             -> per_association_q.csv
  offset     re-counts every surrogate set keeping only detections whose two
             signals were shifted at least --min-rel-ns apart (default 1000 ns), and
             a Horvitz-Thompson rescaled count that corrects for the pairs removed
             by the filter. -> offset_filtered.csv, offset_per_association.csv
  spearman   Pearson and Spearman r over the analyzed window for the SASA
             associations in >= 2 replicas. -> sasa_spearman.csv
  all        chance + perassoc + offset + spearman (no new runs)

Why a filter and not a new shift scheme
---------------------------------------
Every signal is shifted by its own random offset, so some signal pairs end up
with a small relative offset and keep part of their real timing. Requiring
>= 1 us between ALL pairs is impossible (hundreds of signals on a 10 us circle),
so the check is done afterwards on the pairs that were actually detected:
offsets are stored for every set (offsets_<rep>_<kind>.csv). For chi, a residue
pair counts in a replica if at least one of its chi-angle pairs with an episode
is >= --min-rel-ns apart; for H-bonds both bond signals of the hyperedge must be.

Rescaling: in replica r a fraction p_r of signal pairs passes the filter (estimated
from the stored offsets). An association kept in replicas R counts
1 / prod_{r in R} p_r (Horvitz-Thompson). This assumes the filter removes pairs
at random apart from the offset itself; associations losing a replica to the
filter move to a lower recurrence class. Both raw-filtered and rescaled counts
are reported; the rescaled one is the comparable chance level.

Usage
-----
  python kras_hmr_stats.py run --n-null 200 --n-scan 20      # long; restartable
  python kras_hmr_stats.py run --dry-run                      # print commands only
  python kras_hmr_stats.py all                                # after the runs
  python kras_hmr_stats.py offset --min-rel-ns 1000 --only sasa hb
"""
import argparse, glob, json, os, subprocess, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REPS = ["namd_3", "namd_3_2", "namd_3_3"]
KIND = {"chi": "chi", "sasa": "sasa", "hb": "hbond"}
BASE = dict(pmin=80, pmax=120, thr=0.75, ep=500, tdec=10)


# ------------------------------------------------------------------ helpers
def bh(p):
    """Benjamini-Hochberg q-values (NaN-safe)."""
    p = np.asarray(p, float); q = np.full_like(p, np.nan)
    ok = np.isfinite(p); n = ok.sum()
    if n == 0:
        return q
    pv = p[ok]; o = np.argsort(pv)
    r = pv[o] * n / np.arange(1, n + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(r, 1.0)
    q[ok] = out
    return q


def circ_rel(o1, o2, T):
    d = np.abs(np.asarray(o1) - np.asarray(o2)) % T
    return np.minimum(d, T - d)


def null_dir(a):
    return os.path.join(a.indir, "coherence_null")


def need(path):
    if not os.path.exists(path):
        sys.exit(f"missing {path} -- run kras_hmr_coherence_null.py (or 'kras_hmr_stats.py run') first")
    return path


# ------------------------------------------------------------------ run
def cmd_run(a):
    py = sys.executable
    common = ["--indir", a.indir, "--reps", *a.reps, "--seed", str(a.seed),
              "--min-shift-ns", str(a.min_shift_ns), "--only", *a.only]
    if a.accept_legacy_real:
        common.append("--accept-legacy-real")
    cmds = [[py, os.path.join(HERE, "kras_hmr_coherence_null.py"), "--n-surr", str(a.n_null), *common]]
    if a.n_scan:
        cmds.append([py, os.path.join(HERE, "kras_hmr_multiband.py"), "--n-surr", str(a.n_scan), *common])
        cmds.append([py, os.path.join(HERE, "kras_hmr_threshold.py"), "--n-surr", str(a.n_scan), *common])
    for c in cmds:
        print(" ".join(c), flush=True)
        if not a.dry_run:
            r = subprocess.run(c)
            if r.returncode:
                sys.exit(f"failed ({r.returncode}): {' '.join(c)}")
    print("\nNOTE: surrogate sets 1..N are shared between the three scans (same seed per set),\n"
          "so the baseline null and the 80-120 ns entries of the scans use the same shifts.")


# ------------------------------------------------------------------ chance share
def cmd_chance(a):
    S = pd.read_csv(need(os.path.join(null_dir(a), "null_summary.csv")))
    C = pd.read_csv(need(os.path.join(null_dir(a), "null_vs_real.csv")))
    rows = []
    for r in C.itertuples():
        s = S.loc[S.observable == r.observable, r.count].astype(float)
        if r.real == 0 or s.empty:
            continue
        lo, hi = np.percentile(s, [2.5, 97.5])
        rows.append(dict(observable=r.observable, count=r.count, real=int(r.real), n_sets=len(s),
                         surrogate_mean=s.mean(), surrogate_sd=s.std(ddof=1) if len(s) > 1 else np.nan,
                         chance_share=s.mean() / r.real,
                         chance_share_lo=lo / r.real, chance_share_hi=hi / r.real,
                         sets_at_or_above_real=int((s >= r.real).sum())))
    D = pd.DataFrame(rows)
    out = os.path.join(null_dir(a), "chance_share.csv")
    D.to_csv(out, index=False, float_format="%.4g")
    print("network-level chance share = mean surrogate count / real count "
          "(a whole-network estimate; it does not say WHICH associations are chance)")
    print(D.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"-> {out}")
    return D


# ------------------------------------------------------------------ per-association q
def cmd_perassoc(a):
    P = pd.read_csv(need(os.path.join(null_dir(a), "per_association.csv")))
    N = int(P.n_surrogate_sets.max())
    P["p_floor"] = 1.0 / (1 + N)
    P["q_bh_within_observable"] = np.nan
    for o, g in P.groupby("observable"):
        P.loc[g.index, "q_bh_within_observable"] = bh(g.p_empirical.values)
    P["q_bh_all"] = bh(P.p_empirical.values)
    out = os.path.join(null_dir(a), "per_association_q.csv")
    P.to_csv(out, index=False, float_format="%.4g")
    print(f"{len(P)} associations in >= 2 replicas; {N} surrogate sets -> smallest possible p = {1/(1+N):.4f}")
    print(P.groupby("observable").agg(n=("association", "size"),
                                      at_floor=("p_empirical", lambda v: int((v <= 1 / (1 + N) + 1e-12).sum())),
                                      q_below_0_05=("q_bh_all", lambda v: int((v < 0.05).sum())),
                                      max_q=("q_bh_all", "max")).to_string())
    print(f"-> {out}")
    return P


# ------------------------------------------------------------------ offset filter
def column_map(a, obs, rep):
    """column index -> residue id (chi, sasa) or (res_i, res_j) (hb), from the REAL inputs."""
    z = np.load(need(os.path.join(a.indir, f"{rep}_{KIND[obs]}.npz")), allow_pickle=True)
    T = len(z["time_ns"]); dt = float(np.diff(z["time_ns"])[0])
    if obs == "chi":
        return dict(resid=z["resid"].astype(int), chi=z["chi"].astype(int)), T, dt
    if obs == "sasa":
        return dict(resid=z["resid"].astype(int)), T, dt
    return dict(pairs=list(zip(z["res_i"].astype(int).tolist(), z["res_j"].astype(int).tolist()))), T, dt


def load_offsets(sdir, rep, obs):
    f = os.path.join(sdir, f"offsets_{rep}_{KIND[obs]}.csv")
    if not os.path.exists(f):
        return None
    return pd.read_csv(f).sort_values("column").offset_frames.values


def surrogate_dirs(a, obs):
    """surrogate run folders as recorded by kras_hmr_coherence_null.py (null_summary.csv)."""
    S = pd.read_csv(need(os.path.join(null_dir(a), "null_summary.csv")))
    s = S[S.observable == obs].sort_values("surrogate")
    return [(int(r.surrogate), r.folder) for r in s.itertuples()]


def detections(obs, d, reps):
    """per association: {rep: [signal-pair keys with an episode in that rep]}.
    chi : key (resid_i, chi_i, resid_j, chi_j); association (resid_i, resid_j)
    sasa: key (resid_i, resid_j);               association (resid_i, resid_j)
    hb  : key ((a,b),(c,d)) the two bond signals; association 'a-b | c-d'"""
    out = {}
    if obs == "chi":
        E = pd.read_csv(os.path.join(d, "episodes_all_reps.csv"))
        for r in E.itertuples(index=False):
            k = (int(r.resid_i), int(r.resid_j))
            for rep in reps:
                if getattr(r, f"n_episodes_{rep}", 0) and getattr(r, f"n_episodes_{rep}") > 0:
                    out.setdefault(k, {}).setdefault(rep, []).append(
                        (int(r.resid_i), int(r.chi_i), int(r.resid_j), int(r.chi_j)))
    elif obs == "sasa":
        E = pd.read_csv(os.path.join(d, "edges.csv"))
        for r in E.itertuples(index=False):
            k = (int(r.resid_i), int(r.resid_j))
            for rep in reps:
                if getattr(r, f"cov_ns_{rep}", 0) > 0:
                    out.setdefault(k, {}).setdefault(rep, []).append(k)
    else:
        H = pd.read_csv(os.path.join(d, "hyperedges.csv"))
        for r in H.itertuples(index=False):
            A = tuple(sorted(map(int, r.hb1.split("-")))); B = tuple(sorted(map(int, r.hb2.split("-"))))
            k = f"{A[0]}-{A[1]} | {B[0]}-{B[1]}"
            for rep in reps:
                if getattr(r, f"cov_ns_{rep}", 0) > 0:
                    out.setdefault(k, {}).setdefault(rep, []).append((A, B))
    return out


def pass_fraction(offs, T, ms, rng, n=200000):
    """fraction of random signal pairs whose relative circular offset is >= ms frames."""
    i = rng.integers(0, len(offs), n); j = rng.integers(0, len(offs), n)
    keep = i != j
    return float((circ_rel(offs[i[keep]], offs[j[keep]], T) >= ms).mean())


def cmd_offset(a):
    rng = np.random.default_rng(0)
    rows, prow = [], []
    real_rec = {}
    if os.path.exists(os.path.join(null_dir(a), "per_association.csv")):
        P = pd.read_csv(os.path.join(null_dir(a), "per_association.csv"))
        real_rec = {(r.observable, str(r.association)): int(r.real_replicas) for r in P.itertuples()}
    for obs in a.only:
        maps = {rep: column_map(a, obs, rep) for rep in a.reps}
        T = maps[a.reps[0]][1]; dt = maps[a.reps[0]][2]; ms = int(round(a.min_rel_ns / dt))
        idx = {}
        for rep, (m, _, _) in maps.items():
            if obs == "chi":
                idx[rep] = {(int(r), int(c)): k for k, (r, c) in enumerate(zip(m["resid"], m["chi"]))}
            elif obs == "sasa":
                idx[rep] = {int(r): k for k, r in enumerate(m["resid"])}
            else:
                idx[rep] = {tuple(sorted(p)): k for k, p in enumerate(m["pairs"])}
        assoc_hits = {}
        for k_set, d in surrogate_dirs(a, obs):
            sdir = os.path.dirname(os.path.abspath(d))
            offs = {rep: load_offsets(sdir, rep, obs) for rep in a.reps}
            if any(v is None for v in offs.values()):
                print(f"  [{obs} set {k_set}] offsets not found in {sdir}; skipped"); continue
            p_r = {rep: pass_fraction(offs[rep], T, ms, rng) for rep in a.reps}
            det = detections(obs, d, a.reps)
            raw = dict(ge2=0, all3=0); kept = dict(ge2=0, all3=0); ht = dict(ge2=0.0, all3=0.0)
            for key, per in det.items():
                n_raw = len(per)
                keep_reps = []
                for rep, sigs in per.items():
                    o = offs[rep]; ok = False
                    for s in sigs:
                        if obs == "chi":
                            c1 = idx[rep].get((s[0], s[1])); c2 = idx[rep].get((s[2], s[3]))
                        elif obs == "sasa":
                            c1 = idx[rep].get(s[0]); c2 = idx[rep].get(s[1])
                        else:
                            c1 = idx[rep].get(s[0]); c2 = idx[rep].get(s[1])
                        if c1 is None or c2 is None:
                            continue
                        if circ_rel(o[c1], o[c2], T) >= ms:
                            ok = True; break
                    if ok:
                        keep_reps.append(rep)
                n_keep = len(keep_reps)
                w = 1.0 / np.prod([p_r[r] for r in keep_reps]) if n_keep else 0.0
                raw["ge2"] += n_raw >= 2; raw["all3"] += n_raw == 3
                kept["ge2"] += n_keep >= 2; kept["all3"] += n_keep == 3
                ht["ge2"] += w * (n_keep >= 2); ht["all3"] += w * (n_keep == 3)
                lab = key if obs == "hb" else f"{key[0]}-{key[1]}"
                assoc_hits.setdefault(lab, []).append(n_keep)
            rows.append(dict(observable=obs, surrogate=k_set, min_rel_ns=a.min_rel_ns,
                             pass_fraction_mean=np.mean(list(p_r.values())),
                             raw_ge2=raw["ge2"], raw_3=raw["all3"],
                             filtered_ge2=kept["ge2"], filtered_3=kept["all3"],
                             rescaled_ge2=ht["ge2"], rescaled_3=ht["all3"]))
        N = sum(1 for r in rows if r["observable"] == obs)
        for (o_, lab), n_real in real_rec.items():
            if o_ != obs:
                continue
            ks = assoc_hits.get(lab, [])
            k_ge = sum(v >= n_real for v in ks)
            prow.append(dict(observable=obs, association=lab, real_replicas=n_real, n_sets=N,
                             sets_at_or_above_real_filtered=k_ge,
                             p_empirical_filtered=(1 + k_ge) / (1 + N) if N else np.nan))
    R = pd.DataFrame(rows)
    out = os.path.join(null_dir(a), "offset_filtered.csv"); R.to_csv(out, index=False, float_format="%.4g")
    if len(R):
        print(f"relative-offset filter >= {a.min_rel_ns:g} ns (mean over surrogate sets)")
        print(R.groupby("observable")[["pass_fraction_mean", "raw_ge2", "filtered_ge2", "rescaled_ge2",
                                        "raw_3", "filtered_3", "rescaled_3"]].mean().to_string(
            float_format=lambda v: f"{v:.3g}"))
    if prow:
        PR = pd.DataFrame(prow)
        PR["q_bh_all"] = bh(PR.p_empirical_filtered.values)
        PR.to_csv(os.path.join(null_dir(a), "offset_per_association.csv"), index=False, float_format="%.4g")
    print(f"-> {out}")
    return R


# ------------------------------------------------------------------ Spearman
def cmd_spearman(a):
    from scipy.stats import pearsonr, spearmanr
    P = pd.read_csv(need(os.path.join(null_dir(a), "per_association.csv")))
    P = P[P.observable == "sasa"]
    rows = []
    for rep in a.reps:
        z = np.load(need(os.path.join(a.indir, f"{rep}_sasa.npz")), allow_pickle=True)
        X = z["sasa"].astype(float); res = {int(r): k for k, r in enumerate(z["resid"])}
        t = z["time_ns"]; win = (t >= t[0] + a.edge_ns) & (t <= t[-1] - a.edge_ns)
        for lab in P.association:
            i, j = map(int, str(lab).split("-"))
            if i not in res or j not in res:
                continue
            x, y = X[win, res[i]], X[win, res[j]]
            rows.append(dict(association=lab, replica=rep, pearson=pearsonr(x, y)[0],
                             spearman=spearmanr(x, y)[0]))
    D = pd.DataFrame(rows)
    if len(D):
        D["difference"] = D.spearman - D.pearson
    out = os.path.join(null_dir(a), "sasa_spearman.csv"); D.to_csv(out, index=False, float_format="%.3f")
    if len(D):
        W = D.pivot(index="association", columns="replica", values=["pearson", "spearman"])
        print(W.to_string(float_format=lambda v: f"{v:+.2f}"))
        print(f"largest |Spearman - Pearson|: {D.difference.abs().max():.2f}")
    print(f"descriptive only (autocorrelated signals; no p-values). window: edges of {a.edge_ns:g} ns excluded")
    print(f"-> {out}")
    return D


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["run", "chance", "perassoc", "offset", "spearman", "all"])
    p.add_argument("--indir", default="wco_hmr")
    p.add_argument("--reps", nargs="+", default=REPS)
    p.add_argument("--only", nargs="+", choices=["chi", "sasa", "hb"], default=["chi", "sasa", "hb"])
    p.add_argument("--n-null", type=int, default=200, help="surrogate sets for the baseline null")
    p.add_argument("--n-scan", type=int, default=20, help="surrogate sets for band and threshold scans (0: skip)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-shift-ns", type=float, default=1000.0)
    p.add_argument("--min-rel-ns", type=float, default=1000.0, help="offset filter: minimum relative shift")
    p.add_argument("--edge-ns", type=float, default=722.0, help="edge exclusion at 80-120 ns (Spearman window)")
    p.add_argument("--accept-legacy-real", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    a.indir = os.path.abspath(a.indir)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd in ("chance", "all"):
        cmd_chance(a); print()
    if a.cmd in ("perassoc", "all"):
        cmd_perassoc(a); print()
    if a.cmd in ("offset", "all"):
        cmd_offset(a); print()
    if a.cmd in ("spearman", "all"):
        cmd_spearman(a)
    json.dump(vars(a), open(os.path.join(null_dir(a), f"stats_{a.cmd}_params.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
