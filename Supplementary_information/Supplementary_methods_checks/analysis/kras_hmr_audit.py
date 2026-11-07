#!/usr/bin/env python
"""
kras_hmr_audit.py -- check whether existing results need rerunning.

1. Settings actually used: for every coherence output folder under <indir> and
   <indir>_null/s*/, print the recorded tsmooth, tdec, dj, dt_ns, replicas, band,
   threshold and episode from params.json, and flag anything that differs from
   the reported baseline (tsmooth 2, dj 1/12, tdec 10, dt 1 ns, 3 replicas).
2. H-bond label ordering: whether the same H-bond pair appears as (A, B) in one
   replica and (B, A) in another (per-replica *_pairs.csv), whether hyperedges.csv
   contains an association twice after canonical ordering, and whether the H-bond
   input columns are ordered identically and canonically (res_i < res_j, sorted).

  python kras_hmr_audit.py                 # writes <indir>/audit/settings.csv and hbond_order.csv
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

BASE = dict(tsmooth=2.0, dj=1 / 12, tdec=10, dt_ns=1.0)
REPS = ["namd_3", "namd_3_2", "namd_3_3"]


def kind_of(d):
    b = os.path.basename(d)
    return "hb" if b.startswith("HB_") else "sasa" if b.startswith("SASA_") else "chi"


def canon(a, b):
    A = tuple(sorted(map(int, a.split("-")))); B = tuple(sorted(map(int, b.split("-"))))
    return (A, B) if A <= B else (B, A)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="wco_hmr")
    a = p.parse_args()
    out = os.path.join(a.indir, "audit"); os.makedirs(out, exist_ok=True)
    dirs = sorted(set(glob.glob(os.path.join(a.indir, "*", "params.json")) +
                      glob.glob(os.path.join(a.indir + "_null", "s*", "*", "params.json"))))
    rows = []
    for f in dirs:
        d = os.path.dirname(f)
        try:
            A = json.load(open(f)).get("args", {})
        except Exception as e:
            rows.append(dict(folder=d, problem=f"unreadable params.json ({e})")); continue
        if "pmin" not in A:
            continue                                   # not a coherence run (e.g. null summary)
        prob = []
        for k, v in BASE.items():
            if k not in A or A[k] is None or not np.isclose(float(A[k]), v, rtol=1e-6):
                prob.append(f"{k}={A.get(k)}")
        if list(A.get("reps", [])) != REPS:
            prob.append(f"reps={A.get('reps')}")
        rows.append(dict(folder=os.path.relpath(d), observable=kind_of(d), pmin=A.get("pmin"), pmax=A.get("pmax"),
                         thr=A.get("thr"), min_ep=A.get("min_ep"), tsmooth=A.get("tsmooth"), tdec=A.get("tdec"),
                         dj=A.get("dj"), dt_ns=A.get("dt_ns"), reps=" ".join(A.get("reps", [])),
                         sidecar=os.path.exists(os.path.join(d, "multiband_provenance.json")),
                         problem="; ".join(prob)))
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(out, "settings.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 500):
        print(S[["folder", "tsmooth", "tdec", "dj", "dt_ns", "problem"]].to_string(index=False))
    bad = S[S.problem.astype(bool)] if len(S) else S
    print(f"\n{len(S)} coherence folders; {len(bad)} differ from the baseline settings "
          f"(tsmooth 2, dj 1/12, tdec 10, dt 1 ns, replicas {' '.join(REPS)})\n")

    # 2. H-bond ordering
    hrows = []
    for d in sorted(glob.glob(os.path.join(a.indir, "HB_*")) + glob.glob(os.path.join(a.indir + "_null", "s*", "HB_*"))):
        orient = {}
        for f in sorted(glob.glob(os.path.join(d, "*_pairs.csv"))):
            rep = os.path.basename(f)[:-len("_pairs.csv")]
            x = pd.read_csv(f, usecols=["hb1", "hb2", "n_episodes"])
            x = x[x.n_episodes > 0]
            for h1, h2 in zip(x.hb1, x.hb2):
                orient.setdefault(canon(h1, h2), set()).add((h1, h2))
        mixed = sum(len(v) > 1 for v in orient.values())
        noncanon = sum(any(canon(h1, h2) != (tuple(map(int, h1.split("-"))), tuple(map(int, h2.split("-"))))
                           for h1, h2 in v) for v in orient.values())
        dup = np.nan
        hf = os.path.join(d, "hyperedges.csv")
        if os.path.exists(hf):
            h = pd.read_csv(hf, usecols=["hb1", "hb2"])
            keys = [canon(x, y) for x, y in zip(h.hb1, h.hb2)]
            dup = len(keys) - len(set(keys))
        hrows.append(dict(folder=os.path.relpath(d), associations=len(orient),
                          orientation_differs_between_replicas=mixed, non_canonical_labels=noncanon,
                          duplicate_rows_in_hyperedges=dup))
    H = pd.DataFrame(hrows)
    H.to_csv(os.path.join(out, "hbond_order.csv"), index=False)
    if len(H):
        with pd.option_context("display.width", 250, "display.max_colwidth", 70):
            print(H.to_string(index=False))
    # input column order
    cols = {}
    for r in REPS:
        f = os.path.join(a.indir, f"{r}_hbond.npz")
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True)
            cols[r] = list(zip(z["res_i"].tolist(), z["res_j"].tolist()))
    if cols:
        for r, c in cols.items():
            print(f"{r}: {len(c)} H-bond columns; res_i<res_j for all: {all(i < j for i, j in c)}; "
                  f"columns sorted: {c == sorted(c)}")
        ref = next(iter(cols.values()))
        print("column order identical across replicas:", all(c == ref for c in cols.values()))
    print(f"\nwrote {out}/settings.csv and hbond_order.csv")


if __name__ == "__main__":
    main()
