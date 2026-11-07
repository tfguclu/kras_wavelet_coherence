#!/usr/bin/env python
"""
kras_hmr_threshold.py -- why coherence >= 0.75? Chance rate per signal pair and a
threshold scan against time-shifted surrogates (80-120 ns band, 500 ns episode).

(1) Surrogate episode fraction per tested signal pair. For every run folder (real and
    surrogate), per replica, the fraction of tested signal pairs with at least one
    coherent episode. Tested units differ by observable: inter-residue chi-angle pairs;
    eligible SASA residue pairs (residues weak in ANY replica excluded from all, as in
    the network analysis); pairs of H-bond occupancy signals. Every replica CSV is
    required. In surrogates this is a descriptive threshold benchmark, not a calibrated
    false-positive rate for the MD data or a false-discovery fraction among edges.
    Means give equal weight to each replica and set.
(2) Threshold scan. Thresholds 0.60-0.90 (step 0.05) on real signals and on
    surrogate sets 1..N (same deterministic shifts as kras_hmr_coherence_null.py).
    For each threshold: associations (all / >= 2 / 3 replicas), real vs surrogate.

All runs go through kras_hmr_runcheck (explicit settings, validated reuse). Existing
outputs are reused: real 0.70/0.75/0.80 (sensitivity scan; legacy accepted with
--accept-legacy-real) and surrogate 0.75 (20-set null).

Output (<indir>/threshold/):
  pair_rates.csv      threshold x observable x set (real = 0) x replica: pairs tested,
                      pairs with an episode, rate
  scan.csv            threshold x observable: real all/>=2/3, surrogate mean/min/max,
                      per-pair rate real and surrogate (mean over replicas and sets)
  threshold_scan.png  (a) associations in >= 2 replicas, real vs surrogate range;
                      (b) per-pair episode rate in surrogates (log scale)

  python kras_hmr_threshold.py --accept-legacy-real            # 7 thresholds, 4 surrogate sets
  python kras_hmr_threshold.py --n-surr 4 --thresholds 0.7 0.75 0.8
  python kras_hmr_threshold.py --no-run                        # collect only
  # parallel: disjoint set ranges, no real runs, no collection; then collect once
  python kras_hmr_threshold.py --n-surr 200 --first 21 --last 110 --no-real --no-collect
  python kras_hmr_threshold.py --n-surr 200 --first 111 --last 200 --no-real --no-collect
  python kras_hmr_threshold.py --n-surr 200 --no-run --accept-legacy-real
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kras_hmr_runcheck as rc  # noqa: E402
from kras_hmr_coherence_null import make_surrogate, surrogate_tag, counts  # noqa: E402

THRS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def setting(t):
    return dict(pmin=80, pmax=120, thr=round(float(t), 2), ep=500, tdec=10)


def pair_rates(obs, d, reps):
    """per-replica fraction of tested signal pairs with >= 1 episode.
    Every replica CSV is required. For SASA, residues flagged weak (SD < cutoff) in ANY
    replica are excluded from every replica, as in the network analysis (edges.csv)."""
    X = {}
    for r in reps:
        f = os.path.join(d, f"{r}_pairs.csv")
        if not os.path.exists(f):
            raise FileNotFoundError(f)
        X[r] = pd.read_csv(f)
    if obs == "sasa":
        weak = set()
        for r, x in X.items():
            if not {"weak_i", "weak_j", "resid_i", "resid_j"} <= set(x.columns):
                raise ValueError(f"{d}/{r}_pairs.csv lacks weak_i/weak_j flags; cannot apply the common exclusion")
            weak |= set(x.loc[x.weak_i.astype(bool), "resid_i"]) | set(x.loc[x.weak_j.astype(bool), "resid_j"])
        X = {r: x[~(x.resid_i.isin(weak) | x.resid_j.isin(weak))] for r, x in X.items()}
    out = []
    for r, x in X.items():
        n = len(x); k = int((x.n_episodes > 0).sum())
        out.append(dict(replica=r, pairs_tested=n, pairs_with_episode=k, rate=k / n if n else np.nan))
    return out


def safe_rates(obs, d, reps, missing, label):
    try:
        return pair_rates(obs, d, reps)
    except (FileNotFoundError, ValueError) as e:
        missing.append(f"{label}: {e}")
        return []


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="wco_hmr")
    p.add_argument("--reps", nargs="+", default=["namd_3", "namd_3_2", "namd_3_3"])
    p.add_argument("--thresholds", nargs="+", type=float, default=THRS)
    p.add_argument("--only", nargs="+", choices=["chi", "sasa", "hb"], default=["chi", "sasa", "hb"])
    p.add_argument("--n-surr", type=int, default=4)
    p.add_argument("--min-shift-ns", type=float, default=1000.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--accept-legacy-real", action="store_true")
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument("--no-run", action="store_true")
    p.add_argument("--keep-npz", action="store_true")
    p.add_argument("--first", type=int, default=1, help="first control set to run (parallel instances: disjoint ranges)")
    p.add_argument("--last", type=int, default=None, help="last control set to run (default: --n-surr)")
    p.add_argument("--no-real", action="store_true", help="skip the real-data runs (use in parallel instances)")
    p.add_argument("--no-collect", action="store_true", help="run only; collect later with --no-run")
    a = p.parse_args()
    a.indir = os.path.abspath(a.indir)
    thrs = [round(t, 2) for t in a.thresholds]
    outdir = os.path.join(a.indir, "threshold"); os.makedirs(outdir, exist_ok=True)
    log = []
    real_h = {o: rc.input_hashes(a.indir, o, a.reps) for o in a.only}
    e_real = lambda o, t: rc.expectation(o, setting(t), a.reps, real_h[o])
    e_surr = lambda o, t, k: rc.expectation(o, setting(t), a.reps, real_h[o], dict(surrogate_tag(a), set=k))

    # 1. real
    for t in ([] if a.no_real else thrs):
        for o in a.only:
            if rc.find_valid(o, setting(t), a.indir, a.reps, e_real(o, t), a.accept_legacy_real, verbose=False):
                continue
            print(f"[real thr {t} {o}]")
            if not a.no_run:
                rc.run_and_tag(o, setting(t), a.indir, a.reps, e_real(o, t), log)
    # 2. surrogates
    last = a.n_surr if a.last is None else min(a.last, a.n_surr)
    for k in range(max(1, a.first), last + 1):
        sdir = os.path.join(a.indir + "_null", f"s{k}")
        todo = [(t, o) for t in thrs for o in a.only
                if not rc.find_valid(o, setting(t), sdir, a.reps, e_surr(o, t, k), verbose=False)]
        print(f"[surrogate {k}] {len(todo)} runs to do")
        if not todo or a.no_run:
            continue
        make_surrogate(a, k, kinds=sorted({rc.INPUT[o] for _, o in todo}))
        for t, o in todo:
            print(f"[surr {k} thr {t} {o}]")
            rc.run_and_tag(o, setting(t), sdir, a.reps, e_surr(o, t, k), log)
        if not a.keep_npz:
            for f in glob.glob(os.path.join(sdir, "*.npz")):
                os.remove(f)
    with open(os.path.join(outdir, "commands.txt"), "a") as fh:
        fh.write("\n".join(log) + ("\n" if log else ""))

    if a.no_collect:
        print(f"sets {max(1, a.first)}-{last} done (collection skipped)")
        return

    # 3. collect
    PR, SC, missing = [], [], []
    for t in thrs:
        for o in a.only:
            d = rc.find_valid(o, setting(t), a.indir, a.reps, e_real(o, t), a.accept_legacy_real, verbose=False)
            if not d:
                missing.append(f"real {t} {o}"); continue
            real = counts(o, d)
            rr = safe_rates(o, d, a.reps, missing, f"real {t} {o}")
            for r in rr:
                PR.append(dict(threshold=t, observable=o, set=0, **r))
            S, rates = [], []
            for k in range(1, a.n_surr + 1):
                sd = rc.find_valid(o, setting(t), os.path.join(a.indir + "_null", f"s{k}"), a.reps,
                                   e_surr(o, t, k), verbose=False)
                if not sd:
                    missing.append(f"surrogate {k} {t} {o}"); continue
                S.append(counts(o, sd))
                for r in safe_rates(o, sd, a.reps, missing, f"surrogate {k} {t} {o}"):
                    PR.append(dict(threshold=t, observable=o, set=k, **r)); rates.append(r["rate"])
            row = dict(threshold=t, observable=o, n_surr=len(S),
                       real_all=real["n_associations"], real_ge2=real["n_ge2_reps"], real_3=real["n_3_reps"],
                       real_pair_rate=np.mean([r["rate"] for r in rr]) if rr else np.nan,
                       surr_pair_rate=np.mean(rates) if rates else np.nan)
            for c in ("n_associations", "n_ge2_reps", "n_3_reps"):
                v = [s[c] for s in S]
                key = {"n_associations": "all", "n_ge2_reps": "ge2", "n_3_reps": "r3"}[c]
                row[f"surr_{key}_mean"] = np.mean(v) if v else np.nan
                row[f"surr_{key}_min"] = min(v) if v else np.nan
                row[f"surr_{key}_max"] = max(v) if v else np.nan
            SC.append(row)
    if missing and not a.allow_missing:
        sys.exit("missing or invalid runs: " + ", ".join(missing) + " (rerun, or --allow-missing)")
    PR = pd.DataFrame(PR); SC = pd.DataFrame(SC)
    PR.to_csv(os.path.join(outdir, "pair_rates.csv"), index=False, float_format="%.5g")
    SC.to_csv(os.path.join(outdir, "scan.csv"), index=False, float_format="%.4g")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(SC.to_string(index=False))
    if len(SC):
        plot(SC, a, os.path.join(outdir, "threshold_scan.png"))
    print(f"wrote {outdir}/")


def plot(S, a, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"chi": "#b83b72", "sasa": "#2a9d8f", "hb": "#d08c1f"}
    name = {"chi": "χ", "sasa": "SASA", "hb": "H-bond"}
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for o in a.only:
        s = S[S.observable == o].sort_values("threshold")
        ax[0].plot(s.threshold, s.real_ge2, "-o", color=col[o], label=f"{name[o]} real")
        ax[0].fill_between(s.threshold, s.surr_ge2_min, s.surr_ge2_max, color=col[o], alpha=.2, lw=0)
        ax[0].plot(s.threshold, s.surr_ge2_mean, ":", color=col[o], label=f"{name[o]} surrogate")
        ax[1].plot(s.threshold, s.surr_pair_rate, "-s", color=col[o], label=f"{name[o]} surrogate")
        ax[1].plot(s.threshold, s.real_pair_rate, "--", color=col[o], alpha=.6, label=f"{name[o]} real")
    for x in ax:
        x.axvline(0.75, color="#888", lw=.7, ls="--"); x.set_xlabel("coherence threshold")
        x.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("associations in ≥ 2 replicas"); ax[0].set_yscale("symlog", linthresh=5)
    ax[1].set_ylabel("episode fraction per tested signal pair"); ax[1].set_yscale("symlog", linthresh=1e-5)
    ax[0].legend(fontsize=7, frameon=False); ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=200)


if __name__ == "__main__":
    main()
