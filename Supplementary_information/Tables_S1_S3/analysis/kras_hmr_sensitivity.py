#!/usr/bin/env python
"""
kras_hmr_sensitivity.py -- do the headline couplings survive modest parameter changes?

Reruns the three coherence analyses (kras_hmr_wco.py, kras_hmr_sasa_wco.py,
kras_hmr_hbond_wco.py) one parameter at a time around the baseline
(band 80-120 ns, coherence 0.75, episode 500 ns) and tracks:

  * network size (pairs / hyperedges) and how many recur in all 3 replicas
  * the headline associations (Arg102 group, Arg73, Lys104, Lys88, Thr144, ...)
  * the 10 direct DMS overlaps and the number of DMS pairs reached within 2 links

Default grid (7 settings):
  base        80-120 ns, 0.75, 500 ns
  band_lo     60-100 ns          band_hi   100-140 ns
  thr_lo      0.70               thr_hi    0.80
  ep_lo       400 ns             ep_hi     600 ns

Every setting is passed explicitly (--tsmooth 2, --dj 1/12, --tdec 10, filters;
see kras_hmr_runcheck.py). Existing runs are reused only after validation of
params.json, required outputs, input hashes and provenance sidecar; legacy runs
without a sidecar only with --accept-legacy-real. Missing runs stop the collection
(or are shown as NA with --allow-missing), never as zero recurrence.

Output (wco_hmr/sensitivity/):
  summary.csv       one row per setting x observable (sizes, overlaps)
  items.csv         headline association x setting -> replicas (0-3)
  sensitivity.png   heat map of items.csv
  commands.txt      the exact commands that were run

  python kras_hmr_sensitivity.py                 # all observables, default grid
  python kras_hmr_sensitivity.py --only hb       # one observable
  python kras_hmr_sensitivity.py --dry-run       # print commands only
"""
import argparse, itertools as it, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kras_hmr_runcheck as rc  # noqa: E402

GRID = {
    "base":    dict(pmin=80, pmax=120, thr=0.75, ep=500, tdec=10),
    "band_lo": dict(pmin=60, pmax=100, thr=0.75, ep=500, tdec=10),
    "band_hi": dict(pmin=100, pmax=140, thr=0.75, ep=500, tdec=10),
    "thr_lo":  dict(pmin=80, pmax=120, thr=0.70, ep=500, tdec=10),
    "thr_hi":  dict(pmin=80, pmax=120, thr=0.80, ep=500, tdec=10),
    "ep_lo":   dict(pmin=80, pmax=120, thr=0.75, ep=400, tdec=10),
    "ep_hi":   dict(pmin=80, pmax=120, thr=0.75, ep=600, tdec=10),
}
# headline associations: (label, observable, residues)  -- residue pair for chi/sasa, (bondA, bondB) for hb
ITEMS = [
    ("χ  Asp69–Arg102", "chi", (69, 102)), ("χ  Gln99–Arg102", "chi", (99, 102)), ("χ  Arg73–Lys104", "chi", (73, 104)),
    ("χ  Lys128–Asp132", "chi", (128, 132)), ("χ  Glu3–Asp54", "chi", (3, 54)),
    ("SASA  Asp69–Arg102", "sasa", (69, 102)), ("SASA  Gly75–Lys104", "sasa", (75, 104)), ("SASA  Lys88–Asp92", "sasa", (88, 92)),
    ("SASA  Arg73–Lys104", "sasa", (73, 104)), ("SASA  His94–Leu133", "sasa", (94, 133)),
    # the six three-replica associations not tracked in the first scan
    ("χ  Met1–Leu52", "chi", (1, 52)), ("χ  Leu52–Asp54", "chi", (52, 54)), ("χ  Asp126–Lys128", "chi", (126, 128)),
    ("SASA  Val8–Lys16", "sasa", (8, 16)), ("SASA  Gly10–Lys16", "sasa", (10, 16)), ("SASA  Cys80–Ile93", "sasa", (80, 93)),
    ("HB  Arg102: Asp69 / Gln99", "hb", ((69, 102), (99, 102))), ("HB  Arg73: Val103 / Asp105", "hb", ((73, 103), (73, 105))),
    ("HB  Thr144: Asn116 / Gln150", "hb", ((116, 144), (144, 150))), ("HB  Lys88: Glu91 / Asp92", "hb", ((88, 91), (88, 92))),
]
DMS_DIRECT = [("chi", (8, 40)), ("chi", (21, 35)), ("chi", (74, 112)), ("chi", (133, 137)),
              ("sasa", (12, 60)), ("sasa", (17, 32)), ("sasa", (21, 40)),
              ("hb", (31, 40)), ("hb", (32, 40)), ("hb", (37, 58))]
RT = 0.001987 * 303


# ----------------------------------------------------------------------------- reading
def load_edges(obs, d):
    """-> DataFrame(i, j, n) for chi/sasa; hyperedges list for hb"""
    if obs == "chi":
        r = pd.read_csv(os.path.join(d, "residue_pairs.csv"))
        r = r[r.n_reps_any > 0]
        return pd.DataFrame(dict(i=r.resid_i, j=r.resid_j, n=r.n_reps_any))
    if obs == "sasa":
        r = pd.read_csv(os.path.join(d, "edges.csv"))
        return pd.DataFrame(dict(i=r.resid_i, j=r.resid_j, n=r.n_reps_any))
    h = pd.read_csv(os.path.join(d, "hyperedges.csv"))
    out = []
    for r in h.itertuples():
        A = tuple(sorted(map(int, r.hb1.split("-")))); B = tuple(sorted(map(int, r.hb2.split("-"))))
        out.append(dict(A=A, B=B, res=set(A) | set(B), n=int(r.n_reps)))
    return out


def as_graph(obs, E):
    import networkx as nx
    G = nx.Graph()
    if obs == "hb":
        for e in E:
            G.add_edges_from(it.combinations(sorted(e["res"]), 2))
    else:
        G.add_edges_from(zip(E.i, E.j))
    return G


def pair_reps(obs, E, p):
    p = tuple(sorted(p))
    if obs == "hb":
        return max([e["n"] for e in E if set(p) <= e["res"]], default=0)
    m = E[(E.i == p[0]) & (E.j == p[1])]
    return int(m.n.max()) if len(m) else 0


def hyper_reps(E, AB):
    key = {tuple(sorted(AB))}
    return max([e["n"] for e in E if tuple(sorted((e["A"], e["B"]))) in key], default=0)


def load_dms(path):
    d = pd.read_csv(path)
    d = d[(d["mean_kcal/mol"].abs() > RT) & (d["ci95_kcal/mol"] < 1)].copy()
    d["i"] = d[["pos_i_can", "pos_j_can"]].min(axis=1); d["j"] = d[["pos_i_can", "pos_j_can"]].max(axis=1)
    return sorted(set(zip(d.i.astype(int), d.j.astype(int))))


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="wco_hmr")
    p.add_argument("--reps", nargs="+", default=["namd_3", "namd_3_2", "namd_3_3"])
    p.add_argument("--only", nargs="+", choices=["chi", "sasa", "hb"], default=["chi", "sasa", "hb"])
    p.add_argument("--settings", nargs="+", default=list(GRID), choices=list(GRID))
    p.add_argument("--dms", default="kras_coupling_edges_full_canonical.csv")
    p.add_argument("--outdir", default=None, help="default: <indir>/sensitivity")
    p.add_argument("--accept-legacy-real", action="store_true",
                   help="reuse outputs without a provenance sidecar if params.json matches every setting")
    p.add_argument("--allow-missing", action="store_true", help="collect even if some runs are missing (shown as NA)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-run", action="store_true", help="only collect existing results")
    a = p.parse_args()
    a.indir = os.path.abspath(a.indir)
    outdir = a.outdir or os.path.join(a.indir, "sensitivity")
    os.makedirs(outdir, exist_ok=True)
    real_h = {o: rc.input_hashes(a.indir, o, a.reps) for o in a.only}
    exp = lambda o, s: rc.expectation(o, s, a.reps, real_h[o])

    # 1. run what is missing (validated reuse only)
    log = []
    for name in a.settings:
        for obs in a.only:
            s = GRID[name]
            print(f"[{name:7s} {obs:4s}]")
            if rc.find_valid(obs, s, a.indir, a.reps, exp(obs, s), a.accept_legacy_real):
                print("    reuse"); continue
            if a.no_run:
                continue
            rc.run_and_tag(obs, s, a.indir, a.reps, exp(obs, s), log, dry=a.dry_run)
    with open(os.path.join(outdir, "commands.txt"), "a") as fh:
        fh.write("\n".join(log) + ("\n" if log else ""))
    if a.dry_run:
        return

    # 2. collect
    dms = load_dms(a.dms) if os.path.exists(a.dms) else None
    if dms is None:
        print(f"WARNING: {a.dms} not found; DMS overlap columns skipped")
    import networkx as nx
    summ, items, missing = [], [], []
    for name in a.settings:
        s = GRID[name]
        for obs in a.only:
            d = rc.find_valid(obs, s, a.indir, a.reps, exp(obs, s), a.accept_legacy_real, verbose=False)
            if not d:
                missing.append(f"{name} {obs}")
                for lab, o, q in ITEMS:
                    if o == obs:
                        items.append(dict(item=lab, setting=name, replicas=np.nan))
                for o, q in DMS_DIRECT:
                    if o == obs:
                        items.append(dict(item=f"DMS {o.upper() if o != 'chi' else 'χ'}  {q[0]}–{q[1]}",
                                          setting=name, replicas=np.nan))
                continue
            E = load_edges(obs, d)
            n_all = len(E); n3 = sum(e["n"] == 3 for e in E) if obs == "hb" else int((E.n == 3).sum())
            row = dict(setting=name, observable=obs, **s, min_ep_effective=rc.eff_ep(s),
                       n_associations=n_all, n_in_3_reps=n3, folder=os.path.basename(d))
            if dms is not None:
                G = as_graph(obs, E)
                both = [q for q in dms if q[0] in G and q[1] in G]
                L = [nx.shortest_path_length(G, *q) if nx.has_path(G, *q) else None for q in both]
                row.update(dms_both_in_net=len(both), dms_direct=sum(l == 1 for l in L),
                           dms_one_step=sum(l == 2 for l in L), dms_any_path=sum(l is not None for l in L))
                ref = [q for o, q in DMS_DIRECT if o == obs]
                row["baseline_direct_kept"] = f"{sum(pair_reps(obs, E, q) > 0 for q in ref)}/{len(ref)}"
            summ.append(row)
            for lab, o, q in ITEMS:
                if o == obs:
                    items.append(dict(item=lab, setting=name,
                                      replicas=hyper_reps(E, q) if obs == "hb" else pair_reps(obs, E, q)))
            for o, q in DMS_DIRECT:
                if o == obs:
                    items.append(dict(item=f"DMS {o.upper() if o != 'chi' else 'χ'}  {q[0]}–{q[1]}", setting=name,
                                      replicas=pair_reps(obs, E, q)))
    if missing and not a.allow_missing:
        sys.exit("missing or invalid runs: " + ", ".join(missing) + " (rerun, or --allow-missing to show them as NA)")
    S = pd.DataFrame(summ); I = pd.DataFrame(items)
    S.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    if len(I):
        M = I.pivot_table(index="item", columns="setting", values="replicas", aggfunc="max", dropna=False)
        M = M.reindex(index=list(dict.fromkeys(I.item)), columns=a.settings)   # NaN = not analysed
        M.to_csv(os.path.join(outdir, "items.csv"))
        plot(M, os.path.join(outdir, "sensitivity.png"))
        with pd.option_context("display.width", 200):
            print(M.to_string(na_rep="NA", float_format="%.0f"))
    cols = [c for c in ["setting", "observable", "n_associations", "n_in_3_reps", "dms_direct", "dms_one_step", "baseline_direct_kept"] if c in S]
    if len(S):
        print(S[cols].to_string(index=False))
    print(f"wrote {outdir}/")


def plot(M, path):
    rc.plot_items(M, path)


if __name__ == "__main__":
    main()
