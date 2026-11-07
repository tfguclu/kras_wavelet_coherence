#!/usr/bin/env python
"""
kras_hmr_sasa_wco.py -- wavelet coherence between per-residue SASA time series.

Input : wco_hmr/{rep}_sasa.npz  (kras_hmr_sasa.py)
        kras_hmr_wco.py must sit in the same folder (CWT/smoothing reused).
Output: wco_hmr/SASA_P{band}_thr{}_ep{}_ts{}_dt{}_dec{}_dj{}/
          {rep}_pairs.csv         all residue pairs
          {rep}_res_quality.csv   mean SASA, rel. SASA, CV, band power per residue
          episodes_all_reps.csv   pairs with >=1 episode in any replica
          edges.csv               residue-pair edge list (max coverage, n_reps_any),
                                  same columns as kras_wco_network_edges.csv
          params.json

Method (same settings as the chi analysis):
* signal = SASA(t) per residue (A^2), mean removed; scalar signal, so the
  coefficient is standard wavelet coherence  R = |S(Wi Wj*)|^2 / (S|Wi|^2 S|Wj|^2)
  (the RV coefficient of kras_hmr_wco.py reduces to this for 1 component)
* Morlet CWT, Torrence & Webster smoothing (time sigma = tsmooth * scale,
  0.6-octave scale boxcar); band-mean over periods in [pmin, pmax]
* edge exclusion: sqrt(2)*s + 2*tsmooth*s (s = largest scale entering band)
* episode = contiguous R_band(t) >= thr lasting >= min_ep ns; replicas separate
* dt and the time origin are read from time_ns per replica (stride-safe);
  ep_phase_deg = mean phase per episode (~0 exposed together, ~180 anti-correlated),
  ep_phase_R = phase consistency (1 constant phase; low = unstable, phase not meaningful)
* residues with SASA sd < --weak-sd are flagged and, unless --keep-weak, left out
  of edges.csv (still present in the pair tables)
Interpretation: "residue SASA associations" / coherent exposure fluctuations.
Sequence AND spatial neighbours share occlusion, and global conformational
changes move many SASA signals together; --min-seqsep removes only sequence
neighbours. chi-based synthetic pass rates do not transfer to SASA.

  python kras_hmr_sasa_wco.py                        # 80-120 ns, 0.75, 500 ns, ts2
  python kras_hmr_sasa_wco.py --min-seqsep 3
  python kras_hmr_sasa_wco.py --selftest
"""
import argparse, ast, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kras_hmr_wco import (FF, make_scales, cwt_batch, Smoother, episodes,  # noqa: E402
                          telegraph)


def analyse(sasa, resid, resname, a, log=print):
    dt, dtd = a.dt_ns, a.dt_ns * a.tdec
    scales = make_scales(a.pmin, a.pmax, a.dj, 0.3)
    periods = scales * FF
    band = (periods >= a.pmin) & (periods <= a.pmax)
    if not band.any():
        raise ValueError("no wavelet scales inside the requested band")
    x = sasa.astype(np.float64).T                        # (n, T)
    n, T = x.shape
    W = cwt_batch(x, dt, scales)[:, :, ::a.tdec]          # (n, S, Td)
    Td = W.shape[-1]
    sm = Smoother(scales, Td, dtd, a.dj, a.tsmooth)
    P = np.concatenate([sm(np.abs(W[k:k + 32]) ** 2).real for k in range(0, n, 32)]
                       ).astype(np.float32)               # (n, S, Td)
    kmax = min(np.where(band)[0].max() + sm.nbox // 2, len(scales) - 1)
    s_edge = scales[kmax]
    edge = np.sqrt(2) * s_edge + 2 * a.tsmooth * s_edge
    t = np.arange(Td) * dtd
    valid = (t >= edge) & (t <= (T - 1) * dt - edge)
    if not valid.any():
        raise ValueError("no valid time points after edge exclusion; trajectory too short")
    tvalid = valid.sum() * dtd
    Pb = P[:, band, :].mean(axis=1)                       # (n, Td)
    fv = np.zeros(T, bool); fv[int(np.ceil(edge / dt)):int((T - 1) - np.ceil(edge / dt)) + 1] = True
    var_v = x[:, fv].var(axis=1)                          # variance over the same interval
    sd = x.std(axis=1)
    weak = sd < a.weak_sd
    qual = dict(resid=resid, resname=resname, mean_sasa=x.mean(axis=1), sd_sasa=sd,
                cv=sd / np.maximum(x.mean(axis=1), 1e-6),
                band_power=np.array([Pb[k][valid].mean() for k in range(n)]),
                # index under fixed settings, NOT a variance fraction
                band_power_over_variance=np.array([Pb[k][valid].mean() / max(var_v[k], 1e-12)
                                                   for k in range(n)]),
                weak_signal=weak)
    if weak.any():
        log(f"    weak/near-constant SASA (sd < {a.weak_sd} A^2): "
            + ", ".join(f"{resname[k]}{resid[k]}" for k in np.where(weak)[0]))
    rows, t0 = [], time.time()
    for i in range(n - 1):
        js = np.array([j for j in range(i + 1, n) if abs(resid[j] - resid[i]) >= a.min_seqsep])
        for b0 in range(0, js.size, a.chunk):
            jb = js[b0:b0 + a.chunk]
            C = sm(W[i][None] * np.conj(W[jb]))           # (m, S, Td)
            R = np.abs(C) ** 2 / (P[i][None] * P[jb] + 1e-30)
            Rb = R[:, band, :].mean(axis=1)
            Cb = (C[:, band, :] / np.sqrt(P[i][None][:, band, :] * P[jb][:, band, :] + 1e-30)
                  ).mean(axis=1)                          # normalised complex coherency
            for m, j in enumerate(jb):
                r = Rb[m]
                ep = episodes((r >= a.thr) & valid, dtd, a.min_ep)
                t0n = getattr(a, "t0_ns", 0.0)
                rows.append(dict(
                    resid_i=int(resid[i]), resname_i=str(resname[i]),
                    resid_j=int(resid[j]), resname_j=str(resname[j]),
                    seqsep=int(abs(resid[j] - resid[i])),
                    mean_R=float(r[valid].mean()),
                    frac_above=float(((r >= a.thr) & valid).sum() * dtd / tvalid),
                    n_episodes=len(ep),
                    total_ep_ns=float(sum(l for *_, l in ep)),
                    max_ep_ns=float(max((l for *_, l in ep), default=0.0)),
                    episodes_ns=";".join(f"{s0 + t0n:.0f}-{e0 + t0n:.0f}" for s0, e0, _ in ep),
                    # phase of i relative to j: ~0 deg = exposed together,
                    # ~180 deg = one exposed while the other buried
                    ep_phase_deg=";".join(
                        f"{np.degrees(np.angle(Cb[m][int(s0/dtd):int(e0/dtd)].mean())):.0f}"
                        for s0, e0, _ in ep),
                    # phase consistency within the episode: resultant length of the
                    # unit phasors (1 = constant phase, ~0 = phase wanders/cancels)
                    ep_phase_R=";".join(
                        f"{np.abs(np.exp(1j * np.angle(Cb[m][int(s0/dtd):int(e0/dtd)])).mean()):.2f}"
                        for s0, e0, _ in ep),
                    ep_pow=";".join(f"{Pb[i][int(s0/dtd):int(e0/dtd)].mean():.4g}/"
                                    f"{Pb[j][int(s0/dtd):int(e0/dtd)].mean():.4g}"
                                    for s0, e0, _ in ep),
                    rel_sasa_i=float(a._rel[i]), rel_sasa_j=float(a._rel[j]),
                    weak_i=bool(weak[i]), weak_j=bool(weak[j]),
                ))
        if i % 25 == 0:
            log(f"    res {i}/{n}  {time.time() - t0:.0f} s")
    meta = dict(periods_ns=periods.tolist(), band_periods_ns=periods[band].tolist(),
                edge_scale=float(s_edge), coi_edge_ns=float(edge), valid_ns=float(tvalid),
                n_res=n, dt_dec_ns=dtd)
    return rows, meta, qual


def read_meta(z):
    """SASA provenance as a dict: JSON (current) or python-dict string (older);
    {} if absent (files from the first prep version)."""
    if "meta" not in z.files:
        return {}
    raw = str(z["meta"])
    for parse in (json.loads, ast.literal_eval):
        try:
            m = parse(raw)
            if isinstance(m, dict):
                return m
        except (ValueError, SyntaxError):
            pass
    return {"unparsed": raw}


def union_ns(iv):
    tot, cs, ce = 0.0, None, None
    for s0, e0 in sorted(iv):
        if ce is None or s0 > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s0, e0
        else:
            ce = max(ce, e0)
    return tot + (ce - cs if ce is not None else 0.0)


def selftest(a):
    rng = np.random.default_rng(3)
    T = 10000
    A = telegraph(T, 50, rng, states=(80., 120., 160.))
    B = 0.8 * A + 30 + rng.normal(0, 8, T)
    B[:3000] = telegraph(3000, 50, rng, states=(80., 120., 160.))
    C = telegraph(T, 50, rng, states=(80., 120., 160.))
    x = np.stack([A, B, C], 1) + rng.normal(0, 5, (T, 3))
    a._rel = np.ones(3); a.dt_ns = 1.0
    rows, *_ = analyse(x.astype(np.float32), np.array([1, 5, 9]), np.array(["ALA"] * 3),
                       a, log=lambda *z: None)
    g = {(r["resid_i"], r["resid_j"]): r for r in rows}
    ab, ac = g[(1, 5)], g[(1, 9)]
    print(f"A-B coupled 3-10us : mean_R {ab['mean_R']:.2f}  [{ab['episodes_ns']}]  "
          f"phase {ab['ep_phase_deg']} deg (R_phase {ab['ep_phase_R']}; in phase expected ~0)")
    print(f"A-C independent    : mean_R {ac['mean_R']:.2f}  episodes {ac['n_episodes']}")
    ok = ab["total_ep_ns"] > 3000 and ac["n_episodes"] == 0
    print("SELFTEST", "PASS" if ok else "FAIL")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="wco_hmr")
    p.add_argument("--reps", nargs="+", default=["namd_3", "namd_3_2", "namd_3_3"])
    p.add_argument("--pmin", type=float, default=80.0)
    p.add_argument("--pmax", type=float, default=120.0)
    p.add_argument("--thr", type=float, default=0.75)
    p.add_argument("--min-ep", type=float, default=500.0)
    p.add_argument("--tsmooth", type=float, default=2.0)
    p.add_argument("--weak-sd", type=float, default=1.0,
                   help="flag residues with SASA sd below this (A^2)")
    p.add_argument("--tdec", type=int, default=10)
    p.add_argument("--dj", type=float, default=1 / 12)
    p.add_argument("--min-seqsep", type=int, default=1)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--keep-weak", action="store_true",
                   help="keep pairs with a weak/near-constant SASA signal in edges.csv "
                        "(default: excluded from edges.csv, still listed elsewhere)")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest(a); return

    import pandas as pd
    # time axis and SASA provenance from the input files
    z0 = np.load(os.path.join(a.indir, f"{a.reps[0]}_sasa.npz"))
    tns = z0["time_ns"]; dts = np.diff(tns)
    if len(dts) == 0 or (dts <= 0).any() or not np.allclose(dts, dts[0], rtol=1e-6):
        sys.exit("time_ns is not uniformly increasing; aborting")
    a.dt_ns = float(dts[0]); a.t0_ns = float(tns[0])
    smeta = read_meta(z0)
    if not smeta:
        print("WARNING: SASA files carry no provenance (older prep); regenerate with "
              "kras_hmr_sasa.py for a fully identified run", flush=True)
    res_tag = (f"LR{smeta['slices']}" if smeta.get("algorithm") == "LeeRichards"
               else f"SR{smeta.get('sphere_points')}" if smeta.get("sphere_points") else "NA")
    tag = (f"SASA_P{a.pmin:g}-{a.pmax:g}_thr{a.thr:g}_ep{a.min_ep:g}_ts{a.tsmooth:g}"
           f"_dt{a.dt_ns:g}_dec{a.tdec}_dj{a.dj:.4g}_sep{a.min_seqsep}_{res_tag}")
    outdir = os.path.join(a.indir, tag); os.makedirs(outdir, exist_ok=True)
    per, metas, ref = {}, {}, None
    for rep in a.reps:
        z = np.load(os.path.join(a.indir, f"{rep}_sasa.npz"))
        ids = list(zip(z["resid"].tolist(), z["resname"].tolist()))
        if ref is None: ref = ids
        elif ids != ref: sys.exit(f"residue identities in {rep} differ; aborting")
        if not np.allclose(np.diff(z["time_ns"]), a.dt_ns) or read_meta(z) != smeta:
            sys.exit(f"{rep}: time axis or SASA settings differ from {a.reps[0]}; aborting")
        a.t0_ns = float(z["time_ns"][0])          # per-replica time origin
        if not np.isfinite(z["sasa"]).all():
            sys.exit(f"{rep}: non-finite SASA values; aborting")
        a._rel = z["rel_sasa"].mean(axis=0)
        print(f"[{rep}] {z['sasa'].shape[0]} frames, {z['sasa'].shape[1]} residues")
        t0 = time.time()
        rows, meta, q = analyse(z["sasa"], z["resid"], z["resname"], a)
        q["mean_rel_sasa"] = a._rel
        pd.DataFrame(q).to_csv(os.path.join(outdir, f"{rep}_res_quality.csv"), index=False)
        df = pd.DataFrame(rows); df.to_csv(os.path.join(outdir, f"{rep}_pairs.csv"), index=False)
        per[rep], metas[rep] = df, meta
        print(f"[{rep}] {len(df)} pairs, {(df.n_episodes > 0).sum()} with episodes, "
              f"{time.time() - t0:.0f} s")

    weak_res = set()
    for rep, df in per.items():
        weak_res |= set(df.loc[df.weak_i, "resid_i"]) | set(df.loc[df.weak_j, "resid_j"])
    key = ["resid_i", "resname_i", "resid_j", "resname_j", "seqsep"]
    cols = ["mean_R", "n_episodes", "total_ep_ns", "max_ep_ns", "episodes_ns"]
    comb = None
    for rep, df in per.items():
        d = df[key + cols].rename(columns={c: f"{c}_{rep}" for c in cols})
        comb = d if comb is None else comb.merge(d, on=key, how="outer")
    comb["n_reps_any"] = sum((comb[f"n_episodes_{r}"] > 0).astype(int) for r in per)
    hit = comb[comb.n_reps_any > 0].copy()
    for r in per:
        hit[f"cov_ns_{r}"] = [union_ns([tuple(map(float, x.split("-"))) for x in str(e).split(";")])
                              if isinstance(e, str) and e else 0.0 for e in hit[f"episodes_ns_{r}"]]
    hit["max_cov_ns"] = hit[[f"cov_ns_{r}" for r in per]].max(axis=1)
    hit["best_rep"] = hit[[f"cov_ns_{r}" for r in per]].idxmax(axis=1).str[len("cov_ns_"):]
    hit["weak_any"] = hit.resid_i.isin(weak_res) | hit.resid_j.isin(weak_res)
    hit = hit.sort_values(["n_reps_any", "max_cov_ns"], ascending=False)
    hit.to_csv(os.path.join(outdir, "episodes_all_reps.csv"), index=False)
    edges = hit if a.keep_weak else hit[~hit.weak_any]
    edges[key + ["max_cov_ns", "best_rep", "n_reps_any", "weak_any"] + [f"cov_ns_{r}" for r in per]
          ].to_csv(os.path.join(outdir, "edges.csv"), index=False)
    print(f"edges.csv: {len(edges)} pairs ({int(hit.weak_any.sum())} weak-signal pairs "
          f"{'kept' if a.keep_weak else 'excluded'}; weak residues: {sorted(weak_res)})")
    print(f"\n{len(hit)} residue pairs with >=1 episode; by #replicas: "
          f"{hit.n_reps_any.value_counts().sort_index().to_dict()} -> {outdir}")
    with open(os.path.join(outdir, "params.json"), "w") as fh:
        json.dump(dict(args={k: v for k, v in vars(a).items() if not k.startswith("_")},
                       sasa_provenance=smeta, meta=metas), fh, indent=1, default=str)


if __name__ == "__main__":
    main()
