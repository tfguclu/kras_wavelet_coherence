#!/usr/bin/env python
"""
kras_hmr_multiband.py -- coherence networks across timescales, each with its own chance level.

For each period band the three unchanged coherence scripts (kras_hmr_wco.py,
kras_hmr_sasa_wco.py, kras_hmr_hbond_wco.py) are run on the real signals and on
time-shifted surrogate signals (make_surrogate of kras_hmr_coherence_null.py).

Bands (pmax/pmin = 1.5 as in the 80-120 ns baseline; band centres double):
  name   band (ns)  min episode (ns)  grid (ns)  effective min episode (ns)
  P10    10-15        60              2           60
  P20    20-30       125              4          128
  P40    40-60       250              8          256
  P80    80-120      500             10          500   (baseline)
  P160  160-240     1000             10         1000
  P320  320-480     2000             10         2000
Minimum episode ~5 periods at the band centre; the effective minimum is the
requested one rounded up to the output grid (episodes are counted in grid steps).
All analysis settings are passed explicitly (--tsmooth 2, --dj 1/12, --tdec,
--thr 0.75; SASA --min-seqsep 3); nothing depends on child-script defaults.
The edge exclusion is computed by the child scripts from the largest scale, so
the usable time shrinks with the band (read back per replica from valid_ns).

Network sizes are not directly comparable across bands: the minimum episode and
the usable time both change with the band (at P320 about 4 us remain per replica
for a 2 us minimum episode). Compare each band with its own surrogates.

Cache reuse (provenance)
  Every run started by this script gets a sidecar <outdir>/multiband_provenance.json
  with the full command settings, SHA-256 of the real input .npz files and, for
  surrogates, seed, minimum shift and SHA-256 of the shifted .npz files. An
  existing output is reused only if (i) params.json matches every setting
  (replicas, band, threshold, episode, tsmooth, tdec, dj, time step, observable-
  specific filters), (ii) the required output files exist, and (iii) the sidecar
  matches the current inputs, seed and minimum shift. Outputs without a sidecar
  (e.g. the earlier baseline runs) are rerun unless --accept-legacy-real is given,
  which accepts legacy REAL outputs that pass checks (i)-(ii) only. Legacy
  surrogate outputs are never accepted (their seed cannot be verified).
  Before writing surrogate inputs, existing .npz files in the surrogate folder are
  deleted so stale shifted files are never reused.

Output (<indir>/multiband/):
  summary.csv        band x observable: real all / >=2 / 3 replicas, surrogate min-max,
                     effective min episode, edge and usable time per replica (range)
  items.csv          core association x band -> replicas (0-3); empty = not analysed
  reproducible.csv   every association reaching >=2 replicas in at least one band,
                     with its true 0-3 count in every analysed band (empty = not analysed)
  multiband.png      (a) >=2-replica associations, real vs surrogate range, per band;
                     (b) core-association heat map (hatched = not analysed)
  commands.txt

  python kras_hmr_multiband.py --accept-legacy-real   # reuse the validated baseline real runs
  python kras_hmr_multiband.py --bands P10 P20 --only hb
  python kras_hmr_multiband.py --n-surr 0              # real signals only
  python kras_hmr_multiband.py --no-run                # collect only
  # parallel: disjoint set ranges, no real runs, no collection; then collect once
  python kras_hmr_multiband.py --n-surr 200 --first 21 --last 65 --no-real --no-collect
  python kras_hmr_multiband.py --n-surr 200 --first 66 --last 110 --no-real --no-collect
  python kras_hmr_multiband.py --n-surr 200 --no-run --accept-legacy-real
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kras_hmr_runcheck as rc  # noqa: E402
from kras_hmr_sensitivity import ITEMS, load_edges, pair_reps, hyper_reps  # noqa: E402
from kras_hmr_coherence_null import make_surrogate, surrogate_tag  # noqa: E402

THR = 0.75
BANDS = {
    "P10":  dict(pmin=10,  pmax=15,  thr=THR, ep=60,   tdec=2),
    "P20":  dict(pmin=20,  pmax=30,  thr=THR, ep=125,  tdec=4),
    "P40":  dict(pmin=40,  pmax=60,  thr=THR, ep=250,  tdec=8),
    "P80":  dict(pmin=80,  pmax=120, thr=THR, ep=500,  tdec=10),
    "P160": dict(pmin=160, pmax=240, thr=THR, ep=1000, tdec=10),
    "P320": dict(pmin=320, pmax=480, thr=THR, ep=2000, tdec=10),
}


# ----------------------------------------------------------------------------- reading
def counts(obs, d):
    E = load_edges(obs, d)
    n = np.array([e["n"] for e in E]) if obs == "hb" else E.n.values
    return dict(all=int(len(n)), ge2=int((n >= 2).sum()), r3=int((n == 3).sum()))


def assoc_counts(obs, E):
    """{association label: replicas} for every detected association (1-3)."""
    out = {}
    if obs == "hb":
        for e in E:
            out[f"{e['A'][0]}-{e['A'][1]} | {e['B'][0]}-{e['B'][1]}"] = int(e["n"])
    else:
        for r in E.itertuples():
            out[f"{int(r.i)}-{int(r.j)}"] = int(r.n)
    return out


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="wco_hmr")
    p.add_argument("--reps", nargs="+", default=["namd_3", "namd_3_2", "namd_3_3"])
    p.add_argument("--bands", nargs="+", default=list(BANDS), choices=list(BANDS))
    p.add_argument("--only", nargs="+", choices=["chi", "sasa", "hb"], default=["chi", "sasa", "hb"])
    p.add_argument("--n-surr", type=int, default=3)
    p.add_argument("--min-shift-ns", type=float, default=1000.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--accept-legacy-real", action="store_true",
                   help="reuse real outputs without a sidecar if params.json matches every setting")
    p.add_argument("--allow-missing", action="store_true", help="collect even if some runs are missing (NA)")
    p.add_argument("--no-run", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-npz", action="store_true")
    p.add_argument("--first", type=int, default=1, help="first control set to run (parallel instances: disjoint ranges)")
    p.add_argument("--last", type=int, default=None, help="last control set to run (default: --n-surr)")
    p.add_argument("--no-real", action="store_true", help="skip the real-data runs (use in parallel instances)")
    p.add_argument("--no-collect", action="store_true", help="run only; collect later with --no-run")
    a = p.parse_args()
    a.indir = os.path.abspath(a.indir)
    for o in a.only:
        rc.script_path(o)
    outdir = os.path.join(a.indir, "multiband"); os.makedirs(outdir, exist_ok=True)
    log = []
    real_h = {o: rc.input_hashes(a.indir, o, a.reps) for o in a.only}
    exp_real = lambda o, bn: rc.expectation(o, BANDS[bn], a.reps, real_h[o])
    exp_surr = lambda o, bn, k: rc.expectation(o, BANDS[bn], a.reps, real_h[o], dict(surrogate_tag(a), set=k))

    # 1. real signals
    for bn in ([] if a.no_real else a.bands):
        for o in a.only:
            print(f"[real {bn} {o}]")
            if rc.find_valid(o, BANDS[bn], a.indir, a.reps, exp_real(o, bn), a.accept_legacy_real):
                print("    reuse"); continue
            if not a.no_run:
                rc.run_and_tag(o, BANDS[bn], a.indir, a.reps, exp_real(o, bn), log, dry=a.dry_run)

    # 2. surrogate sets (deterministic per file; see kras_hmr_coherence_null.py)
    last = a.n_surr if a.last is None else min(a.last, a.n_surr)
    for k in range(max(1, a.first), last + 1):
        sdir = os.path.join(a.indir + "_null", f"s{k}")
        todo = [(bn, o) for bn in a.bands for o in a.only
                if not rc.find_valid(o, BANDS[bn], sdir, a.reps, exp_surr(o, bn, k), verbose=False)]
        print(f"[surrogate {k}] {len(todo)} runs to do")
        if not todo or a.no_run:
            continue
        if not a.dry_run:
            make_surrogate(a, k, kinds=sorted({rc.INPUT[o] for _, o in todo}))
        for bn, o in todo:
            print(f"[surr {k} {bn} {o}]")
            rc.run_and_tag(o, BANDS[bn], sdir, a.reps, exp_surr(o, bn, k), log, dry=a.dry_run)
        if not a.keep_npz and not a.dry_run:
            for f in glob.glob(os.path.join(sdir, "*.npz")):
                os.remove(f)
    with open(os.path.join(outdir, "commands.txt"), "a") as fh:
        fh.write("\n".join(log) + ("\n" if log else ""))
    if a.dry_run:
        return
    if a.no_collect:
        print(f"sets {max(1, a.first)}-{last} done (collection skipped)")
        return

    # 3. collect (validated outputs only)
    summ, items, allc, analysed, missing = [], [], {}, {}, []
    for bn in a.bands:
        b = BANDS[bn]
        for o in a.only:
            d = rc.find_valid(o, b, a.indir, a.reps, exp_real(o, bn), a.accept_legacy_real, verbose=False)
            if not d:
                missing.append(f"real {bn} {o}")
                for lab, oo, q in ITEMS:
                    if oo == o:
                        items.append(dict(item=lab, band=bn, replicas=np.nan))
                continue
            analysed.setdefault(o, []).append(bn)
            c = counts(o, d)
            edge = rc.per_rep_meta(d, a.reps, "coi_edge_ns"); valid = rc.per_rep_meta(d, a.reps, "valid_ns")
            row = dict(band=bn, pmin=b["pmin"], pmax=b["pmax"], min_ep_requested=b["ep"], grid_ns=b["tdec"],
                       min_ep_effective=rc.eff_ep(b), observable=o, real_all=c["all"], real_ge2=c["ge2"], real_3=c["r3"],
                       edge_ns_min=min(edge) if edge else np.nan, edge_ns_max=max(edge) if edge else np.nan,
                       usable_us_min=min(valid) / 1000 if valid else np.nan,
                       usable_us_max=max(valid) / 1000 if valid else np.nan)
            S = []
            for k in range(1, a.n_surr + 1):
                sd = rc.find_valid(o, b, os.path.join(a.indir + "_null", f"s{k}"), a.reps, exp_surr(o, bn, k), verbose=False)
                if sd:
                    S.append(counts(o, sd))
                else:
                    missing.append(f"surrogate {k} {bn} {o}")
            for key in ("all", "ge2", "r3"):
                v = [x[key] for x in S]
                row[f"surr_{key}_min"] = min(v) if v else np.nan
                row[f"surr_{key}_max"] = max(v) if v else np.nan
            row["n_surr"] = len(S)
            summ.append(row)
            E = load_edges(o, d)
            for lab, oo, q in ITEMS:
                if oo == o:
                    items.append(dict(item=lab, band=bn, replicas=hyper_reps(E, q) if o == "hb" else pair_reps(o, E, q)))
            for key, n in assoc_counts(o, E).items():
                allc.setdefault((o, key), {})[bn] = n
    if missing and not a.allow_missing:
        sys.exit("missing or invalid runs: " + ", ".join(missing) + " (rerun, or --allow-missing to show them as NA)")
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(outdir, "summary.csv"), index=False, float_format="%.4g")
    if len(S):
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(S.drop(columns=["pmin", "pmax"]).to_string(index=False))

    M = None
    I = pd.DataFrame(items)
    if len(I):
        M = I.pivot_table(index="item", columns="band", values="replicas", aggfunc="max", dropna=False)
        M = M.reindex(index=[x for x, _, _ in ITEMS if x in set(I.item)], columns=a.bands)   # NaN = not analysed
        M.to_csv(os.path.join(outdir, "items.csv"))

    rows = []
    for (o, key), v in allc.items():
        if max(v.values()) < 2:
            continue
        r = dict(observable=o, association=key)
        for bn in a.bands:
            r[bn] = v.get(bn, 0) if bn in analysed.get(o, []) else np.nan   # 0 = analysed, not detected
        rows.append(r)
    R = pd.DataFrame(rows)
    if len(R):
        R["n_bands_ge2"] = (R[a.bands] >= 2).sum(axis=1)
        R["n_bands_analysed"] = R[a.bands].notna().sum(axis=1)
        R = R.sort_values(["observable", "n_bands_ge2", "association"], ascending=[True, False, True])
        R.to_csv(os.path.join(outdir, "reproducible.csv"), index=False, float_format="%.0f")
    if len(S) and M is not None:
        plot(S, M, a, os.path.join(outdir, "multiband.png"))
    print(f"wrote {outdir}/")


def plot(S, M, a, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    col = {"chi": "#b83b72", "sasa": "#2a9d8f", "hb": "#d08c1f"}
    name = {"chi": "χ", "sasa": "SASA", "hb": "H-bond"}
    lab = lambda b: f"{BANDS[b]['pmin']}–{BANDS[b]['pmax']}"
    fig = plt.figure(figsize=(11, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(a.bands))
    for o in a.only:
        s = S[S.observable == o].set_index("band").reindex(a.bands)
        ax.plot(x, s.real_ge2, "-o", color=col[o], label=f"{name[o]} real")
        if s.n_surr.max() > 0:
            ax.fill_between(x, s.surr_ge2_min, s.surr_ge2_max, color=col[o], alpha=0.18, lw=0)
            ax.plot(x, s.surr_ge2_max, ":", color=col[o], lw=1, label=f"{name[o]} surrogate (range)")
    ax.set_xticks(x); ax.set_xticklabels([lab(b) for b in a.bands], rotation=30)
    ax.set_xlabel("period band (ns)"); ax.set_ylabel("associations in ≥ 2 replicas")
    ax.set_yscale("symlog", linthresh=5); ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(gs[1])
    cmap = ListedColormap(["#ffffff", "#d4d4d4", "#8f8f8f", "#2b2b2b"]); norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4)
    V = M.values.astype(float)
    ax.imshow(np.ma.masked_invalid(V), cmap=cmap, norm=norm, aspect="auto")
    for (r, c), v in np.ndenumerate(V):
        if np.isnan(v):
            ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, hatch="///", ec="#bbbbbb", lw=0))
        else:
            ax.text(c, r, int(v), ha="center", va="center", fontsize=6.5, color="white" if v >= 2.5 else "#333")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([lab(b) for b in M.columns], rotation=30, fontsize=7)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index, fontsize=7)
    ax.set_title("replicas in which the association is detected (hatched: not analysed)", fontsize=8, loc="left")
    fig.savefig(path, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
