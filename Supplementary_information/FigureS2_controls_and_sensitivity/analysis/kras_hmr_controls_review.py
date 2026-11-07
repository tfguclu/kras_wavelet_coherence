#!/usr/bin/env python
"""
kras_hmr_controls_review.py -- two checks of the time-shifted controls requested in review.

1. seam  (no new coherence runs)
   A circular shift joins the end of each signal to its start, so every shifted signal
   has one artificial jump (the seam). Coherence near a seam is unreliable in the same
   way as near the ends of a record. For control set k, a zone of +- edge (the excluded
   end width, 722 ns at 80-120 ns) around the seam of each signal is removed from the
   episodes of every pair that involves the signal, and pieces shorter than the minimum
   episode are dropped. The SAME zones (same signals, same replica, same times) are
   removed from the episodes of the real signals, so real and control are compared on
   identical analysed time. Reported per control set and observable: pairs in >= 2 and
   in 3 replicas, real-masked vs control-masked; and per real reproducible pair, the
   number of control sets in which the same pair reached the real (masked) recurrence.
   Needs: the existing control runs (<indir>_null/s<k>/..., offsets_<rep>_<kind>.csv)
   and the real baseline runs; real <rep>_<kind>.npz for signal identities.

2. reswise  (new chi runs)
   The chi network counts a residue pair if ANY of its chi-angle pairs is coherent, but
   the original controls shift every chi angle independently, which breaks the
   dependence between angles of the same residue. Here all chi angles of one residue are
   shifted by the same offset (one offset per residue, replica and set), and the
   unchanged kras_hmr_wco.py is run at baseline. Same summary as the original controls.

3. offset  (no new coherence runs)
   A circular shift with independent offsets leaves some signal pairs with a small relative
   offset (for 10 us records, 23% less than 1 us apart), and such pairs can keep part of the
   real coordination. For control set k, a signal pair is eligible in a replica only if its two
   signals were shifted at least --min-rel-ns apart in that replica (chi: the two chi angles;
   H-bonds: the two bond signals of the hyperedge). The SAME pair-by-replica eligibility mask is
   applied to the real episodes, so real and control are counted on identical pairs and no
   rescaling is needed. For chi a residue pair counts a replica if any eligible chi-angle pair
   has an episode there. Reported as for the seam check.

Place next to the kras_hmr_*.py scripts and run from the analysis folder:
  python kras_hmr_controls_review.py seam    --n-surr 200 --accept-legacy-real
  python kras_hmr_controls_review.py reswise --n-surr 200 --jobs 8 --accept-legacy-real
  python kras_hmr_controls_review.py offset  --n-surr 200 --accept-legacy-real   [--min-rel-ns 1000]
Output: <indir>/controls_review/{seam,reswise,offset}_*.csv
"""
import argparse, glob, os, sys, zlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kras_hmr_runcheck as rc                                   # noqa: E402
from kras_hmr_sensitivity import load_edges                      # noqa: E402
import kras_hmr_coherence_null as cn                             # noqa: E402

BASE = dict(pmin=80, pmax=120, thr=0.75, ep=500, tdec=10)
KIND = {"chi": "chi", "sasa": "sasa", "hb": "hbond"}


# ----------------------------------------------------------------------------- helpers
def parse_eps(s):
    if not isinstance(s, str) or not s.strip():
        return []
    return [tuple(map(float, x.split("-"))) for x in s.split(";") if x]


def subtract(ep, zones, min_ep):
    """episode (s, e) minus union of zones -> True if a piece >= min_ep remains."""
    pieces = [ep]
    for zs, ze in zones:
        new = []
        for s, e in pieces:
            if ze <= s or zs >= e:
                new.append((s, e)); continue
            if zs > s: new.append((s, zs))
            if ze < e: new.append((ze, e))
        pieces = new
    return any(e - s >= min_ep - 1e-6 for s, e in pieces)


def signal_ids(indir, rep, obs):
    z = np.load(os.path.join(indir, f"{rep}_{KIND[obs]}.npz"), allow_pickle=True)
    if obs == "chi":
        ids = [(int(r), int(c)) for r, c in zip(z["resid"], z["chi"])]
    elif obs == "sasa":
        ids = [int(r) for r in z["resid"]]
    else:
        ids = [f"{min(int(i), int(j))}-{max(int(i), int(j))}" for i, j in zip(z["res_i"], z["res_j"])]
    return ids, np.asarray(z["time_ns"], float)


def seam_zones(sdir, indir, rep, obs, edge):
    """{signal id: [(start, end)]} of masked time around each signal's seam."""
    f = os.path.join(sdir, f"offsets_{rep}_{KIND[obs]}.csv")
    if not os.path.exists(f):
        return None
    off = pd.read_csv(f)
    ids, t = signal_ids(indir, rep, obs)
    Z = {}
    for col, o in zip(off.column, off.offset_frames):
        ts = t[int(o)]                      # where original frame 0 now sits: the seam
        Z.setdefault(ids[int(col)], []).append((ts - edge, ts + edge))
    return Z


def offset_map(sdir, indir, rep, obs):
    """{signal id: offset in frames} for one control set and replica, and the record length T."""
    f = os.path.join(sdir, f"offsets_{rep}_{KIND[obs]}.csv")
    if not os.path.exists(f):
        return None, None
    off = pd.read_csv(f)
    ids, t = signal_ids(indir, rep, obs)
    return {ids[int(c)]: int(o) for c, o in zip(off.column, off.offset_frames)}, len(t)


def eligible_reps(table, offs_by_rep, T, ms, reps):
    """{pair key: replicas with an episode on a pair whose two signals were shifted >= ms frames
    apart}; for chi a residue pair counts a replica if any eligible chi-angle pair has an episode.
    Also returns (eligible, total) over all signal pairs in the table that have an episode."""
    agg, n_ok, n_all = {}, 0, 0
    for key, a_, b_, eps in table:
        got = agg.setdefault(key, set())
        for rep in reps:
            if not eps.get(rep):
                continue
            O = offs_by_rep[rep]
            if a_ not in O or b_ not in O:
                continue
            d = abs(O[a_] - O[b_]) % T
            n_all += 1
            if min(d, T - d) >= ms:
                n_ok += 1
                got.add(rep)
    return {k: len(v) for k, v in agg.items()}, (n_ok, n_all)


def episodes_table(obs, d, reps):
    """-> list of (pair key, signal a, signal b, {rep: [episodes]}) restricted to network pairs."""
    out = []
    if obs == "hb":
        h = pd.read_csv(os.path.join(d, "hyperedges.csv"))
        for r in h.itertuples():
            a_, b_ = r.hb1, r.hb2
            out.append((f"{a_} | {b_}", a_, b_, {rep: parse_eps(getattr(r, f"episodes_ns_{rep}", ""))
                                                   for rep in reps}))
        return out
    E = load_edges(obs, d)
    keep = set(zip(E.i.astype(int), E.j.astype(int)))
    ep = pd.read_csv(os.path.join(d, "episodes_all_reps.csv"))
    for r in ep.itertuples():
        i, j = int(r.resid_i), int(r.resid_j)
        if (i, j) not in keep:
            continue
        if obs == "chi":
            a_, b_ = (i, int(r.chi_i)), (j, int(r.chi_j))
        else:
            a_, b_ = i, j
        out.append((f"{i}-{j}", a_, b_, {rep: parse_eps(getattr(r, f"episodes_ns_{rep}", "")) for rep in reps}))
    return out


def masked_reps(table, zones_by_rep, reps, min_ep):
    """{pair key: replicas with a surviving episode}; for chi a residue pair counts a replica
    if any of its chi-angle pairs keeps an episode."""
    agg = {}
    for key, a_, b_, eps in table:
        got = agg.setdefault(key, set())
        for rep in reps:
            Z = zones_by_rep.get(rep) or {}
            zz = Z.get(a_, []) + Z.get(b_, [])
            if any(subtract(e, zz, min_ep) for e in eps.get(rep, [])):
                got.add(rep)
    return {k: len(v) for k, v in agg.items()}


def summarize(rep_counts):
    v = np.array(list(rep_counts.values())) if rep_counts else np.zeros(0)
    return dict(n_associations=int((v >= 1).sum()), n_ge2_reps=int((v >= 2).sum()), n_3_reps=int((v == 3).sum()))


# ----------------------------------------------------------------------------- 1. seam
def run_seam(a):
    import kras_hmr_averages as av
    outdir = os.path.join(a.indir, "controls_review"); os.makedirs(outdir, exist_ok=True)
    real_h = {o: rc.input_hashes(a.indir, o, a.reps) for o in a.only}
    stag = lambda k: dict(set=k, seed=a.seed, min_shift_ns=a.min_shift_ns, scheme=cn.SCHEME)
    min_ep = rc.eff_ep(BASE)
    rows, pa = [], []
    for o in a.only:
        dreal = rc.find_valid(o, BASE, a.indir, a.reps, rc.expectation(o, BASE, a.reps, real_h[o]),
                              accept_legacy=a.accept_legacy_real, verbose=False)
        if not dreal:
            sys.exit(f"real baseline run for {o} not found")
        T = len(signal_ids(a.indir, a.reps[0], o)[1])
        edge = av.band_setup(T, 1.0, BASE["tdec"])["edge"] if a.edge_ns is None else a.edge_ns
        Treal = episodes_table(o, dreal, a.reps)
        real_unmasked = masked_reps(Treal, {}, a.reps, min_ep)
        ref = {k: n for k, n in real_unmasked.items() if n >= 2}
        hits = {k: 0 for k in ref}; nsets = 0
        for k in range(1, a.n_surr + 1):
            sdir = os.path.join(a.indir + "_null", f"s{k}")
            d = rc.find_valid(o, BASE, sdir, a.reps, rc.expectation(o, BASE, a.reps, real_h[o], stag(k)),
                              verbose=False)
            zones = {rep: seam_zones(sdir, a.indir, rep, o, edge) for rep in a.reps}
            if not d or any(z is None for z in zones.values()):
                print(f"    control {k} {o}: run or offsets missing, skipped"); continue
            nsets += 1
            ctrl = masked_reps(episodes_table(o, d, a.reps), zones, a.reps, min_ep)
            real = masked_reps(Treal, zones, a.reps, min_ep)
            rows.append(dict(observable=o, control=k, edge_ns=edge,
                             **{f"real_masked_{c}": v for c, v in summarize(real).items()},
                             **{f"ctrl_masked_{c}": v for c, v in summarize(ctrl).items()}))
            for key in ref:
                if ctrl.get(key, 0) >= max(real.get(key, 0), 2):
                    hits[key] += 1
        for key, n in ref.items():
            pa.append(dict(observable=o, association=key, real_replicas=n, n_controls=nsets,
                           controls_at_or_above_real_masked=hits[key],
                           p_empirical=(1 + hits[key]) / (1 + nsets) if nsets else np.nan))
    R = pd.DataFrame(rows); P = pd.DataFrame(pa)
    R.to_csv(os.path.join(outdir, "seam_per_control.csv"), index=False)
    P.to_csv(os.path.join(outdir, "seam_per_association.csv"), index=False, float_format="%.3g")
    if len(R):
        g = R.groupby("observable").agg(
            real_ge2_masked_min=("real_masked_n_ge2_reps", "min"), real_ge2_masked_max=("real_masked_n_ge2_reps", "max"),
            ctrl_ge2_masked_mean=("ctrl_masked_n_ge2_reps", "mean"), ctrl_ge2_masked_max=("ctrl_masked_n_ge2_reps", "max"),
            real_3_masked_min=("real_masked_n_3_reps", "min"), ctrl_3_masked_max=("ctrl_masked_n_3_reps", "max"),
            sets_ctrl_ge2_reaching_real=("ctrl_masked_n_ge2_reps",
                                         lambda v: int((v.values >= R.loc[v.index, "real_masked_n_ge2_reps"].values).sum())))
        print(g.to_string()); g.to_csv(os.path.join(outdir, "seam_summary.csv"))
    print(f"wrote {outdir}/seam_*.csv")


# ----------------------------------------------------------------------------- 3. offset
def run_offset(a):
    outdir = os.path.join(a.indir, "controls_review"); os.makedirs(outdir, exist_ok=True)
    real_h = {o: rc.input_hashes(a.indir, o, a.reps) for o in a.only}
    stag = lambda k: dict(set=k, seed=a.seed, min_shift_ns=a.min_shift_ns, scheme=cn.SCHEME)
    min_ep = rc.eff_ep(BASE)
    rows, pa = [], []
    for o in a.only:
        dreal = rc.find_valid(o, BASE, a.indir, a.reps, rc.expectation(o, BASE, a.reps, real_h[o]),
                              accept_legacy=a.accept_legacy_real, verbose=False)
        if not dreal:
            sys.exit(f"real baseline run for {o} not found")
        Treal = episodes_table(o, dreal, a.reps)
        ref = {k: n for k, n in masked_reps(Treal, {}, a.reps, min_ep).items() if n >= 2}
        hits = {k: 0 for k in ref}; nsets = 0
        for k in range(1, a.n_surr + 1):
            sdir = os.path.join(a.indir + "_null", f"s{k}")
            d = rc.find_valid(o, BASE, sdir, a.reps, rc.expectation(o, BASE, a.reps, real_h[o], stag(k)),
                              verbose=False)
            om = {rep: offset_map(sdir, a.indir, rep, o) for rep in a.reps}
            if not d or any(v[0] is None for v in om.values()):
                print(f"    control {k} {o}: run or offsets missing, skipped"); continue
            T = om[a.reps[0]][1]
            dt = float(np.diff(signal_ids(a.indir, a.reps[0], o)[1])[0])
            ms = int(round(a.min_rel_ns / dt))
            offs = {rep: v[0] for rep, v in om.items()}
            nsets += 1
            ctrl, (c_ok, c_all) = eligible_reps(episodes_table(o, d, a.reps), offs, T, ms, a.reps)
            real, (r_ok, r_all) = eligible_reps(Treal, offs, T, ms, a.reps)
            rows.append(dict(observable=o, control=k, min_rel_ns=a.min_rel_ns,
                             real_eligible_fraction=r_ok / r_all if r_all else np.nan,
                             ctrl_eligible_fraction=c_ok / c_all if c_all else np.nan,
                             **{f"real_masked_{c}": v for c, v in summarize(real).items()},
                             **{f"ctrl_masked_{c}": v for c, v in summarize(ctrl).items()}))
            for key in ref:
                if ctrl.get(key, 0) >= max(real.get(key, 0), 2):
                    hits[key] += 1
        for key, n in ref.items():
            pa.append(dict(observable=o, association=key, real_replicas=n, n_controls=nsets,
                           controls_at_or_above_real_masked=hits[key],
                           p_empirical=(1 + hits[key]) / (1 + nsets) if nsets else np.nan))
    R = pd.DataFrame(rows); P = pd.DataFrame(pa)
    R.to_csv(os.path.join(outdir, "offset_per_control.csv"), index=False, float_format="%.4g")
    P.to_csv(os.path.join(outdir, "offset_per_association.csv"), index=False, float_format="%.3g")
    if len(R):
        g = R.groupby("observable").agg(
            real_ge2_masked_min=("real_masked_n_ge2_reps", "min"), real_ge2_masked_max=("real_masked_n_ge2_reps", "max"),
            ctrl_ge2_masked_mean=("ctrl_masked_n_ge2_reps", "mean"), ctrl_ge2_masked_max=("ctrl_masked_n_ge2_reps", "max"),
            real_3_masked_min=("real_masked_n_3_reps", "min"), ctrl_3_masked_max=("ctrl_masked_n_3_reps", "max"),
            sets_ctrl_ge2_reaching_real=("ctrl_masked_n_ge2_reps",
                                         lambda v: int((v.values >= R.loc[v.index, "real_masked_n_ge2_reps"].values).sum())),
            real_eligible_fraction=("real_eligible_fraction", "mean"),
            ctrl_eligible_fraction=("ctrl_eligible_fraction", "mean"))
        print(g.to_string()); g.to_csv(os.path.join(outdir, "offset_summary.csv"))
        if len(P):
            print("\nreal reproducible pairs reached by a masked control (controls_at_or_above_real_masked > 0):")
            hit = P[P.controls_at_or_above_real_masked > 0]
            print(hit.to_string(index=False) if len(hit) else "  none")
    print(f"wrote {outdir}/offset_*.csv")


# ----------------------------------------------------------------------------- 2. reswise
RES_SCHEME = "circshift-reswise-v1"


def make_reswise(a, k):
    out = os.path.join(a.indir + "_null_reswise", f"s{k}"); os.makedirs(out, exist_ok=True)
    for rep in a.reps:
        src = os.path.join(a.indir, f"{rep}_chi.npz"); dst = os.path.join(out, f"{rep}_chi.npz")
        z = np.load(src, allow_pickle=True); d = {f: z[f] for f in z.files}
        dt = float(np.diff(d["time_ns"])[0]); ms = int(round(a.min_shift_ns / dt))
        X = d["angles"]; T, n = X.shape
        rng = np.random.default_rng(np.random.SeedSequence([a.seed, k, zlib.crc32(rep.encode()), 99]))
        res = np.asarray(d["resid"]).astype(int); ures = np.unique(res)
        off_res = dict(zip(ures, rng.integers(ms, T - ms, size=len(ures))))
        offs = np.array([off_res[r] for r in res])
        d["angles"] = np.stack([np.roll(X[:, j], offs[j]) for j in range(n)], axis=1)
        d.update(surr_offsets=offs, surr_seed=np.int64(a.seed), surr_set=np.int64(k),
                 surr_scheme=np.array(RES_SCHEME), surr_src_sha256=np.array(rc.sha256(src)),
                 surr_min_shift_ns=np.float64(a.min_shift_ns))
        np.savez_compressed(dst, **d)
        pd.DataFrame(dict(column=np.arange(n), resid=res, offset_frames=offs, offset_ns=offs * dt)).to_csv(
            os.path.join(out, f"offsets_{rep}_chi.csv"), index=False)
    return out


def _do_reswise(job):
    k, ad, real_h = job
    from types import SimpleNamespace
    a = SimpleNamespace(**ad); log = []
    stag = dict(set=k, seed=a.seed, min_shift_ns=a.min_shift_ns, scheme=RES_SCHEME)
    sdir = os.path.join(a.indir + "_null_reswise", f"s{k}")
    exp = rc.expectation("chi", BASE, a.reps, real_h, stag)
    if not rc.find_valid("chi", BASE, sdir, a.reps, exp, verbose=False):
        make_reswise(a, k); rc.run_and_tag("chi", BASE, sdir, a.reps, exp, log)
    for f in glob.glob(os.path.join(sdir, "*.npz")):
        os.remove(f)
    return k, log


def run_reswise(a):
    outdir = os.path.join(a.indir, "controls_review"); os.makedirs(outdir, exist_ok=True)
    o = "chi"; real_h = rc.input_hashes(a.indir, o, a.reps)
    stag = lambda k: dict(set=k, seed=a.seed, min_shift_ns=a.min_shift_ns, scheme=RES_SCHEME)
    dreal = rc.find_valid(o, BASE, a.indir, a.reps, rc.expectation(o, BASE, a.reps, real_h),
                          accept_legacy=a.accept_legacy_real, verbose=False)
    if not dreal:
        sys.exit("real chi baseline run not found")
    sd = lambda k: os.path.join(a.indir + "_null_reswise", f"s{k}")
    todo = [k for k in range(1, a.n_surr + 1)
            if not rc.find_valid(o, BASE, sd(k), a.reps, rc.expectation(o, BASE, a.reps, real_h, stag(k)), verbose=False)]
    print(f"reswise control sets to run: {len(todo)} of {a.n_surr} ({a.jobs} in parallel)", flush=True)
    if todo and not a.no_run:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ad = dict(vars(a)); n = 0
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for f in as_completed([ex.submit(_do_reswise, (k, ad, real_h)) for k in todo]):
                k, lg = f.result(); n += 1
                print(f"[{n}/{len(todo)}] reswise set {k} done", flush=True)
                with open(os.path.join(outdir, "reswise_commands.txt"), "a") as fh:
                    fh.write("\n".join(lg) + ("\n" if lg else ""))
    real_assoc = cn.assoc(o, dreal); rows, sets = [], []
    for k in range(1, a.n_surr + 1):
        d = rc.find_valid(o, BASE, sd(k), a.reps, rc.expectation(o, BASE, a.reps, real_h, stag(k)), verbose=False)
        if d:
            rows.append(dict(control=k, **cn.counts(o, d))); sets.append(cn.assoc(o, d))
    S = pd.DataFrame(rows); S.to_csv(os.path.join(outdir, "reswise_per_control.csv"), index=False)
    print("real:", cn.counts(o, dreal))
    if len(S):
        print(S[["n_associations", "n_ge2_reps", "n_3_reps"]].agg(["mean", "min", "max"]).to_string())
    pa = []
    for key, n in real_assoc.items():
        if n < 2:
            continue
        kge = sum(s_.get(key, 0) >= n for s_ in sets)
        pa.append(dict(association=key, real_replicas=n, n_controls=len(sets), controls_at_or_above_real=kge,
                       p_empirical=(1 + kge) / (1 + len(sets)) if sets else np.nan))
    pd.DataFrame(pa).to_csv(os.path.join(outdir, "reswise_per_association.csv"), index=False, float_format="%.3g")
    print(f"wrote {outdir}/reswise_*.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["seam", "reswise", "offset"])
    p.add_argument("--indir", default="wco_hmr")
    p.add_argument("--reps", nargs="+", default=["namd_3", "namd_3_2", "namd_3_3"])
    p.add_argument("--only", nargs="+", choices=["chi", "sasa", "hb"], default=["chi", "sasa", "hb"])
    p.add_argument("--n-surr", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-shift-ns", type=float, default=1000.0)
    p.add_argument("--edge-ns", type=float, default=None, help="half-width of the seam mask (default: excluded-end width)")
    p.add_argument("--jobs", type=int, default=6, help="reswise: control sets run in parallel")
    p.add_argument("--threads", type=int, default=2, help="BLAS/OpenMP threads per process")
    p.add_argument("--min-rel-ns", type=float, default=1000.0, help="offset: minimum relative shift of a pair")
    p.add_argument("--accept-legacy-real", action="store_true")
    p.add_argument("--no-run", action="store_true")
    a = p.parse_args()
    a.indir = os.path.abspath(a.indir)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(a.threads)
    {"seam": run_seam, "reswise": run_reswise, "offset": run_offset}[a.mode](a)


if __name__ == "__main__":
    main()
