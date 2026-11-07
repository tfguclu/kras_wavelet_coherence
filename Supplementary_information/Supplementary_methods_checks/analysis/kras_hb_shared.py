#!/usr/bin/env python
"""
kras_hb_shared.py -- how often are pairs of H-bond signals that share a residue coherent?

Question (reviewer): the four core H-bond pairs (R102, R73, K88, T144) each join two
H-bonds of one residue, so co-formation or alternation could follow from geometry alone.
If sharing a residue made coherence likely, most shared-residue pairs would be coherent.

Reads the baseline H-bond coherence output (kras_hmr_hbond_wco.py):
  <hbdir>/{rep}_pairs.csv   every tested pair of H-bond signals in that replica
                            (hb1, hb2, residues, size, n_episodes, ...)
size 3 = the two H-bonds share a residue; size 4 = four different residues.
Optionally the same files of the time-shifted controls (--ctrl-glob).

For size-3 and size-4 pairs separately:
  tested     pairs tested in all three replicas (both signals pass the signal filter)
  coherent   pairs with an episode in >= 1, >= 2 and all 3 replicas
and, for the controls, the same counts per control set (mean and maximum).
Also: per shared residue with >= 2 tested pairs, how many of its pairs are coherent.

Run from the analysis folder:
  python kras_hb_shared.py                                  # newest wco_hmr/HB_P80-120*/
  python kras_hb_shared.py --hbdir wco_hmr/HB_P80-120_... --ctrl-glob "wco_hmr_null/s*/HB_P80-120_*"
Writes <hbdir>/shared_residue/ : summary.csv, per_residue.csv, shared_recurrent.csv, report.txt
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

REPS = ["namd_3", "namd_3_2", "namd_3_3"]


def shared_residue(hb1, hb2):
    a, b = set(map(int, hb1.split("-"))), set(map(int, hb2.split("-")))
    s = a & b
    return int(next(iter(s))) if len(s) == 1 else None


def table(d, reps):
    """per pair: tested in how many replicas, coherent in how many"""
    rows = []
    for rep in reps:
        f = os.path.join(d, f"{rep}_pairs.csv")
        if not os.path.exists(f):
            return None
        p = pd.read_csv(f, usecols=["hb1", "hb2", "size", "n_episodes"])
        p["rep"] = rep
        rows.append(p)
    P = pd.concat(rows)
    g = P.groupby(["hb1", "hb2", "size"])
    T = pd.DataFrame(dict(tested=g.size(), coherent=g.n_episodes.apply(lambda x: int((x > 0).sum())))).reset_index()
    return T


def counts(T, nrep):
    out = {}
    for size, lab in [(3, "shared residue (size 3)"), (4, "no shared residue (size 4)")]:
        t = T[(T["size"] == size) & (T.tested == nrep)]
        out[lab] = dict(tested=len(t), ge1=int((t.coherent >= 1).sum()), ge2=int((t.coherent >= 2).sum()),
                        all3=int((t.coherent == nrep).sum()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hbdir", default=None, help="baseline HB_* folder (default: newest wco_hmr/HB_P80-120*)")
    p.add_argument("--reps", nargs="+", default=REPS)
    p.add_argument("--ctrl-glob", default=None,
                   help="control folders, e.g. 'wco_hmr_null/s*/HB_P80-120_*' (default: same tag under wco_hmr_null/s*)")
    a = p.parse_args()
    if a.hbdir is None:
        c = sorted(glob.glob("wco_hmr/HB_P80-120*"), key=os.path.getmtime)
        if not c:
            sys.exit("no wco_hmr/HB_P80-120* folder; give --hbdir")
        a.hbdir = c[-1]
    tag = os.path.basename(os.path.normpath(a.hbdir))
    if a.ctrl_glob is None:
        a.ctrl_glob = os.path.join("wco_hmr_null", "s*", tag)
    out = os.path.join(a.hbdir, "shared_residue"); os.makedirs(out, exist_ok=True)
    nrep = len(a.reps)

    T = table(a.hbdir, a.reps)
    if T is None:
        sys.exit(f"missing {{rep}}_pairs.csv in {a.hbdir}")
    real = counts(T, nrep)

    # per shared residue
    t3 = T[(T["size"] == 3) & (T.tested == nrep)].copy()
    t3["shared"] = [shared_residue(x, y) for x, y in zip(t3.hb1, t3.hb2)]
    per = (t3.groupby("shared").agg(pairs=("coherent", "size"), ge1=("coherent", lambda x: int((x >= 1).sum())),
                                    ge2=("coherent", lambda x: int((x >= 2).sum())),
                                    all3=("coherent", lambda x: int((x == nrep).sum())))
           .reset_index().sort_values(["ge2", "pairs"], ascending=False))
    per.to_csv(os.path.join(out, "per_residue.csv"), index=False)
    t3[t3.coherent >= 2].sort_values("coherent", ascending=False).to_csv(
        os.path.join(out, "shared_recurrent.csv"), index=False)

    # controls
    ctrl = []
    for d in sorted(glob.glob(a.ctrl_glob)):
        Tc = table(d, a.reps)
        if Tc is None:
            continue
        c = counts(Tc, nrep)
        ctrl.append({(k, m): v[m] for k, v in c.items() for m in v})
    rows = []
    for lab, v in real.items():
        r = dict(group=lab, **{f"real_{m}": v[m] for m in v})
        for m in ["ge1", "ge2", "all3"]:
            r[f"real_frac_{m}"] = v[m] / v["tested"] if v["tested"] else np.nan
            if ctrl:
                x = np.array([c[(lab, m)] for c in ctrl])
                r[f"ctrl_mean_{m}"], r[f"ctrl_max_{m}"] = x.mean(), int(x.max())
                r[f"ctrl_reach_{m}"] = int((x >= v[m]).sum())
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(out, "summary.csv"), index=False, float_format="%.4g")

    L = [f"Baseline H-bond coherence: {a.hbdir}", f"Controls: {len(ctrl)} sets ({a.ctrl_glob})",
         "Pairs of H-bond signals tested in all three replicas:", ""]
    for r in rows:
        L.append(f"{r['group']}: {r['real_tested']} tested; coherent in >=1 replica {r['real_ge1']} "
                 f"({100 * r['real_frac_ge1']:.1f}%), >=2 {r['real_ge2']} ({100 * r['real_frac_ge2']:.1f}%), "
                 f"all 3 {r['real_all3']} ({100 * r['real_frac_all3']:.2f}%)")
        if ctrl:
            L.append(f"   controls: >=2 replicas mean {r['ctrl_mean_ge2']:.2f} (max {r['ctrl_max_ge2']}), "
                     f"all 3 mean {r['ctrl_mean_all3']:.2f} (max {r['ctrl_max_all3']}); "
                     f"controls reaching the real >=2 count: {r['ctrl_reach_ge2']} of {len(ctrl)}")
    L += ["", "Shared residues with the most recurrent pairs (pairs tested / >=2 / all 3):"]
    for x in per.head(15).itertuples():
        L.append(f"  residue {x.shared}: {x.pairs} / {x.ge2} / {x.all3}")
    L += ["", "Reading: if sharing a residue made coherence likely, the shared-residue fractions would be high.",
          "The size-4 row is the comparison; the controls give the rate expected by coincidence."]
    open(os.path.join(out, "report.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
