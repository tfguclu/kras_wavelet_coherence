#!/usr/bin/env python
"""
kras_hmr_smoothing.py -- does the core survive a change of the time smoothing?

Reviewer request: smoothing (Gaussian in time, SD = tsmooth x scale) was fixed at 2 in
every sensitivity test. This wrapper reruns the unchanged coherence scripts at the
baseline band, threshold and episode (80-120 ns, 0.75, 500 ns) with other smoothing
widths, on the real signals and on N time-shifted controls (same seeds and offsets as
the original controls 1..N), and reports for each smoothing:

  * network size, pairs in >= 2 and in 3 replicas (real vs control range)
  * replicas (0-3) of every headline item of kras_hmr_sensitivity.ITEMS
    (the 16 core pairs + the non-robust chi pairs)

Notes
  * The excluded ends grow with the smoothing (edge = sqrt(2) s + 2 tsmooth s), so the
    analysed trajectory is shorter for larger tsmooth; it is reported per run.
  * The minimum episode is kept at 500 ns. With tsmooth = 1 the smoothing is narrower
    than the episode; with tsmooth = 3 it is wider (~680 ns at half height at 100 ns).
  * Nothing in the pipeline is changed: kras_hmr_runcheck.TSMOOTH is set for the
    duration of each block, so run folders are named ..._ts<value>_... and validated
    against that value; baseline (ts 2) outputs are reused after validation.

Place next to the kras_hmr_*.py scripts and run from the analysis folder:
  python kras_hmr_smoothing.py --accept-legacy-real                 # ts 1, 2, 3; 200 controls
  python kras_hmr_smoothing.py --tsmooth 1.5 3 --n-surr 20 --only sasa hb
  python kras_hmr_smoothing.py --jobs 8 --threads 2 --accept-legacy-real   # 8 control sets in parallel
  python kras_hmr_smoothing.py --no-run                              # collect only
Parallel: one process per control set; sets never share folders, so workers cannot
interfere. Interrupting is safe: finished runs carry a provenance sidecar and are reused,
unfinished ones lack it and are redone.
Output: <indir>/smoothing/summary.csv, per_association.csv, items.csv, items.png, commands.txt
"""
import argparse, glob, os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kras_hmr_runcheck as rc                                   # noqa: E402
from kras_hmr_sensitivity import ITEMS, DMS_DIRECT, load_edges, pair_reps, hyper_reps   # noqa: E402
from kras_hmr_coherence_null import make_surrogate, SCHEME, assoc  # noqa: E402

BASE = dict(pmin=80, pmax=120, thr=0.75, ep=500, tdec=10)


def counts(obs, E):
    n = np.array([e["n"] for e in E]) if obs == "hb" else E.n.values
    return dict(n_associations=int(len(n)), n_ge2_reps=int((n >= 2).sum()), n_3_reps=int((n == 3).sum()))


def _set_threads(n):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(n)


def _do_set(job):
    """one worker = one control set k, all smoothings and observables (sets never share folders)."""
    k, ad, real_h = job
    a = SimpleNamespace(**ad)
    sa = SimpleNamespace(indir=a.indir, reps=a.reps, seed=a.seed, min_shift_ns=a.min_shift_ns)
    tag = dict(set=k, seed=a.seed, min_shift_ns=a.min_shift_ns, scheme=SCHEME)
    sdir = os.path.join(a.indir + "_null", f"s{k}")
    log = []
    for ts in a.tsmooth:
        rc.TSMOOTH = ts
        exp = {o: rc.expectation(o, BASE, a.reps, real_h[o], tag) for o in a.only}
        todo = [o for o in a.only if not rc.find_valid(o, BASE, sdir, a.reps, exp[o], verbose=False)]
        if todo:
            make_surrogate(sa, k, kinds=[rc.INPUT[o] for o in todo])
            for o in todo:
                rc.run_and_tag(o, BASE, sdir, a.reps, exp[o], log)
    for f in glob.glob(os.path.join(sdir, "*.npz")):
        os.remove(f)
    return k, log


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="wco_hmr")
    p.add_argument("--reps", nargs="+", default=["namd_3", "namd_3_2", "namd_3_3"])
    p.add_argument("--only", nargs="+", choices=["chi", "sasa", "hb"], default=["chi", "sasa", "hb"])
    p.add_argument("--tsmooth", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    p.add_argument("--n-surr", type=int, default=200, help="controls per smoothing (sets 1..N of the original seed)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-shift-ns", type=float, default=1000.0)
    p.add_argument("--jobs", type=int, default=6, help="control sets run in parallel (one process per set)")
    p.add_argument("--threads", type=int, default=2, help="BLAS/OpenMP threads per process")
    p.add_argument("--accept-legacy-real", action="store_true")
    p.add_argument("--no-run", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    a.indir = os.path.abspath(a.indir)
    _set_threads(a.threads)
    outdir = os.path.join(a.indir, "smoothing"); os.makedirs(outdir, exist_ok=True)
    real_h = {o: rc.input_hashes(a.indir, o, a.reps) for o in a.only}
    stag = lambda k: dict(set=k, seed=a.seed, min_shift_ns=a.min_shift_ns, scheme=SCHEME)
    log, summ, items, pa = [], [], [], []
    ts0 = rc.TSMOOTH
    try:
        # 1. real signals (serial, few runs)
        real = {}
        for ts in a.tsmooth:
            rc.TSMOOTH = ts
            for o in a.only:
                e = rc.expectation(o, BASE, a.reps, real_h[o])
                d = rc.find_valid(o, BASE, a.indir, a.reps, e, accept_legacy=a.accept_legacy_real, verbose=False)
                if not d and not a.no_run:
                    d = rc.run_and_tag(o, BASE, a.indir, a.reps, e, log, dry=a.dry_run)
                real[(ts, o)] = d
        # 2. controls: which sets still miss a run?
        todo_sets = []
        for k in range(1, a.n_surr + 1):
            sdir = os.path.join(a.indir + "_null", f"s{k}")
            miss = False
            for ts in a.tsmooth:
                rc.TSMOOTH = ts
                for o in a.only:
                    if not rc.find_valid(o, BASE, sdir, a.reps, rc.expectation(o, BASE, a.reps, real_h[o], stag(k)),
                                         verbose=False):
                        miss = True
            if miss:
                todo_sets.append(k)
        print(f"control sets to run: {len(todo_sets)} of {a.n_surr} ({a.jobs} in parallel)", flush=True)
        if todo_sets and not a.no_run and not a.dry_run:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            ad = {k_: v for k_, v in vars(a).items()}
            done = 0
            with ProcessPoolExecutor(max_workers=a.jobs) as ex:
                futs = [ex.submit(_do_set, (k, ad, real_h)) for k in todo_sets]
                for f in as_completed(futs):
                    k, lg = f.result(); log += lg; done += 1
                    print(f"[{done}/{len(todo_sets)}] control set {k} done", flush=True)
                    with open(os.path.join(outdir, "commands.txt"), "a") as fh:
                        fh.write("\n".join(lg) + ("\n" if lg else ""))
            log = []
        if a.dry_run:
            return
        # 3. collect
        for ts in a.tsmooth:
            rc.TSMOOTH = ts
            ctrl = {o: [] for o in a.only}; ctrl_assoc = {o: [] for o in a.only}
            for k in range(1, a.n_surr + 1):
                sdir = os.path.join(a.indir + "_null", f"s{k}")
                for o in a.only:
                    d = rc.find_valid(o, BASE, sdir, a.reps, rc.expectation(o, BASE, a.reps, real_h[o], stag(k)),
                                      verbose=False)
                    if d:
                        ctrl[o].append(counts(o, load_edges(o, d))); ctrl_assoc[o].append(assoc(o, d))
            for o in a.only:
                if not real[(ts, o)]:
                    print(f"    missing real {o} at ts {ts:g}"); continue
                E = load_edges(o, real[(ts, o)]); c = counts(o, E)
                C = pd.DataFrame(ctrl[o])
                row = dict(tsmooth=ts, observable=o, valid_ns=np.mean(rc.per_rep_meta(real[(ts, o)], a.reps, "valid_ns")),
                           **{f"real_{k}": v for k, v in c.items()}, n_controls=len(C))
                for col in ["n_associations", "n_ge2_reps", "n_3_reps"]:
                    if len(C):
                        row[f"ctrl_{col}_min"] = int(C[col].min()); row[f"ctrl_{col}_max"] = int(C[col].max())
                        row[f"ctrl_{col}_mean"] = round(float(C[col].mean()), 2)
                summ.append(row)
                N = len(ctrl_assoc[o])
                for key, n in assoc(o, real[(ts, o)]).items():
                    if n >= 2:
                        k_ge = sum(s_.get(key, 0) >= n for s_ in ctrl_assoc[o])
                        pa.append(dict(tsmooth=ts, observable=o, association=key, real_replicas=n, n_controls=N,
                                       controls_at_or_above_real=k_ge, p_empirical=(1 + k_ge) / (1 + N) if N else np.nan))
                for lab, oo, q in ITEMS:
                    if oo == o:
                        items.append(dict(item=lab, tsmooth=ts,
                                          replicas=hyper_reps(E, q) if o == "hb" else pair_reps(o, E, q)))
                for oo, q in DMS_DIRECT:                       # direct DMS overlaps (as in the sensitivity scan)
                    if oo == o:
                        items.append(dict(item=f"DMS {o.upper() if o != 'chi' else 'χ'}  {q[0]}–{q[1]}", tsmooth=ts,
                                          replicas=pair_reps(o, E, q)))
    finally:
        rc.TSMOOTH = ts0
    with open(os.path.join(outdir, "commands.txt"), "a") as fh:
        fh.write("\n".join(log) + ("\n" if log else ""))
    S = pd.DataFrame(summ); S.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    pd.DataFrame(pa).to_csv(os.path.join(outdir, "per_association.csv"), index=False, float_format="%.3g")
    if items:
        I = pd.DataFrame(items)
        M = I.pivot_table(index="item", columns="tsmooth", values="replicas", aggfunc="max")
        M = M.reindex(index=list(dict.fromkeys(I.item)))
        M.to_csv(os.path.join(outdir, "items.csv"))
        rc.plot_items(M, os.path.join(outdir, "items.png"), xlabels=[f"ts {t:g}" for t in M.columns])
        print(M.to_string(float_format="%.0f"))
    with pd.option_context("display.width", 220):
        print(S.to_string(index=False))
    print(f"wrote {outdir}/")


if __name__ == "__main__":
    main()
