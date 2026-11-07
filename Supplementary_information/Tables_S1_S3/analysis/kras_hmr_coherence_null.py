#!/usr/bin/env python
"""
kras_hmr_coherence_null.py -- how many associations pass the criterion with time-shifted surrogates?

Surrogate data: every signal (each chi angle, each residue's SASA, each H-bond
residue-pair series) is circularly shifted in time by its own random offset in
[min_shift, T - min_shift]. A shift keeps each signal's distribution and circular
autocorrelation but destroys most of its timing relative to other signals.
Each replica gets independent shifts, so cross-replica recurrence is tested too.

Reproducibility (scheme "circshift-v2")
  Each (surrogate set, replica, observable) has its own random stream,
  numpy SeedSequence([seed, set, crc32(replica), observable index]), so a file
  regenerated after a partial restart is identical to the original. The offsets
  are stored in the shifted .npz (surr_offsets, surr_seed, surr_set, surr_scheme,
  surr_src_sha256) and in <null>/s<k>/offsets_<rep>_<obs>.csv, which is kept.
  An existing shifted .npz is reused only if its stored seed, set, scheme and
  source hash match; otherwise it is rewritten.

The analysis code is NOT modified: shifted copies of the input .npz files are
written to <indir>_null/s<k>/ and the unchanged kras_hmr_wco.py,
kras_hmr_sasa_wco.py and kras_hmr_hbond_wco.py are run on them at the baseline
settings, with every setting passed explicitly (see kras_hmr_runcheck.py).
Outputs are reused only after full validation (settings, inputs, seed).

Output (<indir>/coherence_null/):
  null_summary.csv   per surrogate set x observable: associations, in >= 2 and in 3 replicas
  per_association.csv for every real association in >= 2 replicas: in how many surrogate
                     sets the SAME association was detected / reached >= 2 / 3 replicas, and
                     p_empirical = (1 + sets with recurrence >= the real one) / (1 + sets)
  null_vs_real.csv   real count, surrogate mean / min / max, and surrogate_to_real_ratio
                     (= surrogate count / real count; a ratio of network sizes, NOT the
                     fraction of false associations in the observed network)

  python kras_hmr_coherence_null.py --accept-legacy-real   # 3 surrogate sets
  python kras_hmr_coherence_null.py --n-surr 5 --only hb
  python kras_hmr_coherence_null.py --no-run               # collect only
"""
import argparse, glob, json, os, sys, zlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kras_hmr_runcheck as rc  # noqa: E402

SCHEME = "circshift-v2"
KINDS = [("chi", "angles", "chi"), ("sasa", "sasa", "sasa"), ("hbond", "hb", "hb")]  # file kind, key, observable
BASE = dict(pmin=80, pmax=120, thr=0.75, ep=500, tdec=10)


def _stream(seed, k, rep, kind_idx):
    return np.random.default_rng(np.random.SeedSequence([seed, k, zlib.crc32(rep.encode()), kind_idx]))


def surrogate_tag(a):
    return dict(set=None, seed=a.seed, min_shift_ns=a.min_shift_ns, scheme=SCHEME)


def make_surrogate(a, k, kinds=("chi", "sasa", "hbond")):
    """write shifted copies of the inputs for surrogate set k; deterministic per file."""
    out = os.path.join(a.indir + "_null", f"s{k}")
    os.makedirs(out, exist_ok=True)
    for rep in a.reps:
        for ki, (kind, key, _) in enumerate(KINDS):
            if kind not in kinds:
                continue
            src = os.path.join(a.indir, f"{rep}_{kind}.npz")
            dst = os.path.join(out, f"{rep}_{kind}.npz")
            if not os.path.exists(src):
                continue
            src_hash = rc.sha256(src)
            if os.path.exists(dst):
                try:
                    z = np.load(dst, allow_pickle=True)
                    if ("surr_scheme" in z.files and str(z["surr_scheme"]) == SCHEME and int(z["surr_seed"]) == a.seed
                            and int(z["surr_set"]) == k and str(z["surr_src_sha256"]) == src_hash
                            and float(z["surr_min_shift_ns"]) == a.min_shift_ns):
                        continue
                except Exception:
                    pass
                os.remove(dst)
            z = np.load(src, allow_pickle=True)
            d = {f: z[f] for f in z.files}
            dt = float(np.diff(d["time_ns"])[0])
            ms = int(round(a.min_shift_ns / dt))
            X = d[key]; T, n = X.shape
            rng = _stream(a.seed, k, rep, ki)
            offs = rng.integers(ms, T - ms, size=n)
            d[key] = np.stack([np.roll(X[:, j], offs[j]) for j in range(n)], axis=1)
            if kind == "sasa" and "rel_sasa" in d:      # same offset for SASA and relative SASA
                d["rel_sasa"] = np.stack([np.roll(d["rel_sasa"][:, j], offs[j]) for j in range(n)], axis=1)
            d.update(surr_offsets=offs, surr_seed=np.int64(a.seed), surr_set=np.int64(k),
                     surr_scheme=np.array(SCHEME), surr_src_sha256=np.array(src_hash),
                     surr_min_shift_ns=np.float64(a.min_shift_ns))
            np.savez_compressed(dst, **d)
            pd.DataFrame(dict(column=np.arange(n), offset_frames=offs, offset_ns=offs * dt)).to_csv(
                os.path.join(out, f"offsets_{rep}_{kind}.csv"), index=False)
    return out


def assoc(obs, d):
    """{association label: replicas (1-3)} for every association in a run folder."""
    from kras_hmr_sensitivity import load_edges
    E = load_edges(obs, d)
    if obs == "hb":
        return {f"{e['A'][0]}-{e['A'][1]} | {e['B'][0]}-{e['B'][1]}": int(e["n"]) for e in E}
    return {f"{int(r.i)}-{int(r.j)}": int(r.n) for r in E.itertuples()}


def counts(obs, d):
    from kras_hmr_sensitivity import load_edges
    E = load_edges(obs, d)
    n = np.array([e["n"] for e in E]) if obs == "hb" else E.n.values
    return dict(n_associations=int(len(n)), n_ge2_reps=int((n >= 2).sum()), n_3_reps=int((n == 3).sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="wco_hmr")
    p.add_argument("--reps", nargs="+", default=["namd_3", "namd_3_2", "namd_3_3"])
    p.add_argument("--n-surr", type=int, default=3)
    p.add_argument("--min-shift-ns", type=float, default=1000.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--only", nargs="+", choices=["chi", "sasa", "hb"], default=["chi", "sasa", "hb"])
    p.add_argument("--accept-legacy-real", action="store_true",
                   help="use real baseline outputs without a sidecar if params.json matches every setting")
    p.add_argument("--allow-missing", action="store_true", help="collect even if some runs are missing")
    p.add_argument("--no-run", action="store_true")
    p.add_argument("--keep-npz", action="store_true")
    a = p.parse_args()
    a.indir = os.path.abspath(a.indir)
    log = []
    real_h = {o: rc.input_hashes(a.indir, o, a.reps) for o in a.only}
    stag = lambda k: dict(surrogate_tag(a), set=k)

    rows, missing, surr_assoc, real_dirs = [], [], {}, {}
    for k in range(1, a.n_surr + 1):
        sdir = os.path.join(a.indir + "_null", f"s{k}")
        exp = {o: rc.expectation(o, BASE, a.reps, real_h[o], stag(k)) for o in a.only}
        todo = [o for o in a.only if not rc.find_valid(o, BASE, sdir, a.reps, exp[o], verbose=False)]
        if todo and not a.no_run:
            print(f"[surrogate {k}] writing shifted inputs", flush=True)
            make_surrogate(a, k, kinds=[rc.INPUT[o] for o in todo])
            for o in todo:
                rc.run_and_tag(o, BASE, sdir, a.reps, exp[o], log)
            if not a.keep_npz:
                for f in glob.glob(os.path.join(sdir, "*.npz")):
                    os.remove(f)
        for o in a.only:
            d = rc.find_valid(o, BASE, sdir, a.reps, exp[o], verbose=False)
            if d:
                rows.append(dict(surrogate=k, observable=o, folder=d, **counts(o, d)))
                surr_assoc.setdefault(o, []).append(assoc(o, d))
            else:
                missing.append(f"surrogate {k} {o}")
    real = {}
    for o in a.only:
        d = rc.find_valid(o, BASE, a.indir, a.reps, rc.expectation(o, BASE, a.reps, real_h[o]),
                          accept_legacy=a.accept_legacy_real)
        if d:
            real[o] = counts(o, d); real_dirs[o] = d
        else:
            missing.append(f"real {o}")
    if missing and not a.allow_missing:
        sys.exit("missing or invalid runs: " + ", ".join(missing) + " (rerun, or --allow-missing)")
    S = pd.DataFrame(rows)
    outdir = os.path.join(a.indir, "coherence_null"); os.makedirs(outdir, exist_ok=True)
    S.to_csv(os.path.join(outdir, "null_summary.csv"), index=False)
    cmp_ = []
    for o in a.only:
        if o not in real or S.empty:
            continue
        s = S[S.observable == o]
        for col in ["n_associations", "n_ge2_reps", "n_3_reps"]:
            m = s[col].mean()
            cmp_.append(dict(observable=o, count=col, real=real[o][col], surrogate_mean=m,
                             surrogate_min=s[col].min(), surrogate_max=s[col].max(),
                             surrogate_to_real_ratio=(m / real[o][col] if real[o][col] else np.nan),
                             n_surrogates=len(s)))
    C = pd.DataFrame(cmp_)
    C.to_csv(os.path.join(outdir, "null_vs_real.csv"), index=False, float_format="%.3g")

    # per-association chance level: how often does the SAME association reach the same
    # recurrence in surrogate sets? (identity-level; empirical p = (1 + k) / (1 + N))
    pa = []
    for o, d in real_dirs.items():
        sets = surr_assoc.get(o, []); N = len(sets)
        for key, n in assoc(o, d).items():
            if n < 2:
                continue
            ks = [s_.get(key, 0) for s_ in sets]
            k_ge = sum(v >= n for v in ks)
            pa.append(dict(observable=o, association=key, real_replicas=n, n_surrogate_sets=N,
                           sets_detected=sum(v >= 1 for v in ks), sets_ge2=sum(v >= 2 for v in ks),
                           sets_3=sum(v == 3 for v in ks), sets_at_or_above_real=k_ge,
                           p_empirical=(1 + k_ge) / (1 + N) if N else np.nan))
    PA = pd.DataFrame(pa)
    if len(PA):
        PA = PA.sort_values(["observable", "real_replicas", "p_empirical"], ascending=[True, False, True])
        PA.to_csv(os.path.join(outdir, "per_association.csv"), index=False, float_format="%.3g")
        print(f"\nper-association chance level ({len(PA)} associations in >= 2 replicas):")
        print(PA.groupby("observable").agg(n=("association", "size"),
                                           never_in_surrogates=("sets_detected", lambda v: int((v == 0).sum())),
                                           max_sets_ge2=("sets_ge2", "max")).to_string())
    print(C.to_string(index=False))
    json.dump(dict(vars(a), scheme=SCHEME), open(os.path.join(outdir, "params.json"), "w"), indent=2)
    with open(os.path.join(outdir, "commands.txt"), "a") as fh:
        fh.write("\n".join(log) + ("\n" if log else ""))
    print(f"wrote {outdir}/")


if __name__ == "__main__":
    main()
