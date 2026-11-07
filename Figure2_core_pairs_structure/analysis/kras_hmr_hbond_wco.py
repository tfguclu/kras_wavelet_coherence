#!/usr/bin/env python
"""
kras_hmr_hbond_wco.py -- wavelet coherence between residue-pair H-bond signals.

Each signal is b_ij(t) in {0,1}: residues i and j share >= 1 H-bond in frame t
(kras_hmr_hbond.py). Coherence between two signals b_ij and b_kl couples two
H-bonds, i.e. the residue set {i,j,k,l}: a hyperedge of size 4 (size 3 when the
two H-bonds share a residue).

Input : wco_hmr/{rep}_hbond.npz ; kras_hmr_wco.py in the same folder
        optional --dms kras_coupling_edges_full_canonical.csv (DMS overlap)
Output: wco_hmr/HB_P{band}_thr{}_ep{}_ts{}_dt{}_dec{}_dj{}_occ{}-{}_tr{}/
          {rep}_signals.csv      H-bond signals used (occupancy, transitions)
          {rep}_pairs.csv        all signal pairs: coherence, episodes, phase
          hyperedges.csv         signal pairs with >=1 episode in any replica,
                                 per-replica coverage, n_reps, size, DMS pairs inside
          params.json

Detection rule (same as chi/SASA): band-mean coherence over 80-120 ns >= 0.75,
sustained >= 500 ns, Torrence-Webster smoothing 2 x scale, edge exclusion
sqrt(2)s + 2*tsmooth*s; replicas analysed separately; dt read from time_ns.
Signal filter (per replica): occupancy in [occ_min, occ_max] and >= min_trans
0->1 transitions -- an H-bond that is always or never present has no
fluctuation to be coherent with (normalised coherence of near-constant signals
is unreliable).
Local power floor: a time-scale cell counts only where both signals have band power
>= pow_floor x their own median (default 0.05); elsewhere coherence is set to 0,
so long stretches where an H-bond is constantly absent/present cannot pass.
Phase per episode: ~0 deg = the two H-bonds form/break together, ~180 deg = one
forms while the other breaks (e.g. partner switching); ep_phase_R = consistency
(1 constant, low = phase not meaningful).

  python kras_hmr_hbond_wco.py --dms kras_coupling_edges_full_canonical.csv
  python kras_hmr_hbond_wco.py --selftest
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kras_hmr_wco import FF, make_scales, cwt_batch, Smoother, episodes, telegraph  # noqa: E402


def transitions(x):
    return int(((x[1:] == 1) & (x[:-1] == 0)).sum())


def analyse(M, labels, a, log=print):
    """M (T, n) 0/1 ; labels list of (i, j). Returns rows, meta."""
    dt, dtd = a.dt_ns, a.dt_ns * a.tdec
    scales = make_scales(a.pmin, a.pmax, a.dj, 0.3)
    periods = scales * FF
    band = (periods >= a.pmin) & (periods <= a.pmax)
    if not band.any():
        raise ValueError("no wavelet scales inside the band")
    x = M.astype(np.float64).T
    n, T = x.shape
    W = cwt_batch(x, dt, scales)[:, :, ::a.tdec]
    Td = W.shape[-1]
    sm = Smoother(scales, Td, dtd, a.dj, a.tsmooth)
    P = np.concatenate([sm(np.abs(W[k:k + 32]) ** 2).real for k in range(0, n, 32)]).astype(np.float32)
    P = np.maximum(P, 0.0)                      # FFT rounding can give tiny negatives
    kmax = min(np.where(band)[0].max() + sm.nbox // 2, len(scales) - 1)
    s_edge = scales[kmax]
    edge = np.sqrt(2) * s_edge + 2 * a.tsmooth * s_edge
    t = np.arange(Td) * dtd
    valid = (t >= edge) & (t <= (T - 1) * dt - edge)
    if not valid.any():
        raise ValueError("no valid time after edge exclusion")
    tvalid = valid.sum() * dtd
    # local power floor: a cell counts only where BOTH signals fluctuate in-band.
    # Binary H-bond signals sit constant (absent/present) for long stretches; there
    # coherence is 0/0 and can come out spuriously high.
    Pband = P[:, band, :]
    floor = a.pow_floor * np.array([np.median(Pband[k][:, valid]) for k in range(n)])
    active = Pband >= floor[:, None, None]           # (n, Sb, Td)
    t0n = getattr(a, "t0_ns", 0.0)
    rows, tic = [], time.time()
    for i in range(n - 1):
        jb_all = np.arange(i + 1, n)
        for b0 in range(0, jb_all.size, a.chunk):
            jb = jb_all[b0:b0 + a.chunk]
            C = sm(W[i][None] * np.conj(W[jb]))
            Pi, Pj = P[i][None][:, band, :], P[jb][:, band, :]
            ok = active[i][None] & active[jb]
            R = np.where(ok, np.abs(C[:, band, :]) ** 2 / (Pi * Pj + 1e-30), 0.0)
            Rb = R.mean(axis=1)
            Cb = np.where(ok, C[:, band, :] / np.sqrt(Pi * Pj + 1e-30), 0.0).mean(axis=1)
            for m, j in enumerate(jb):
                r = Rb[m]
                ep = episodes((r >= a.thr) & valid, dtd, a.min_ep)
                # canonical labels: each bond as (low, high), the pair ordered by bond;
                # if the pair is swapped, the coherency is conjugated -> phase sign flips
                L1, L2 = tuple(sorted(labels[i])), tuple(sorted(labels[j]))
                sgn = 1.0
                if L1 > L2:
                    L1, L2, sgn = L2, L1, -1.0
                res = sorted(set(L1) | set(L2))
                rows.append(dict(
                    hb1=f"{L1[0]}-{L1[1]}", hb2=f"{L2[0]}-{L2[1]}", residues=";".join(map(str, res)),
                    size=len(res), mean_R=float(r[valid].mean()),
                    frac_above=float(((r >= a.thr) & valid).sum() * dtd / tvalid),
                    n_episodes=len(ep), total_ep_ns=float(sum(l for *_, l in ep)),
                    max_ep_ns=float(max((l for *_, l in ep), default=0.0)),
                    episodes_ns=";".join(f"{s0 + t0n:g}-{e0 + t0n:g}" for s0, e0, _ in ep),
                    ep_phase_deg=";".join(
                        f"{sgn * np.degrees(np.angle(Cb[m][int(s0/dtd):int(e0/dtd)].mean())):.0f}"
                        for s0, e0, _ in ep),
                    ep_phase_R=";".join(
                        f"{np.abs(np.exp(1j*np.angle(Cb[m][int(s0/dtd):int(e0/dtd)])).mean()):.2f}"
                        for s0, e0, _ in ep)))
        if i % 50 == 0:
            log(f"    signal {i}/{n}  {time.time() - tic:.0f} s")
    return rows, dict(band_periods_ns=periods[band].tolist(), edge_scale=float(s_edge),
                      coi_edge_ns=float(edge), valid_ns=float(tvalid), dt_dec_ns=dtd)


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


def load_dms(path):
    import pandas as pd
    d = pd.read_csv(path)
    RT = 0.001987 * 303
    s = d[(d["mean_kcal/mol"].abs() > RT) & (d["ci95_kcal/mol"] < 1)]
    out = {}
    for (i, j), g in s.groupby([s[["pos_i_can", "pos_j_can"]].min(axis=1),
                                s[["pos_i_can", "pos_j_can"]].max(axis=1)]):
        if i != j:
            out[(int(i), int(j))] = ",".join(sorted(set(g.trait.str.lower())))
    return out


def selftest(a):
    rng = np.random.default_rng(7)
    T = 10000
    def binar(x): return (x > 120).astype(np.uint8)
    A = binar(telegraph(T, 50, rng))
    B = A.copy(); flip = rng.random(T) < 0.03; B[flip] = 1 - B[flip]; B[:3000] = binar(telegraph(3000, 50, rng))
    Cn = 1 - A; Cn[:3000] = binar(telegraph(3000, 50, rng))          # anti-phase partner
    D = binar(telegraph(T, 50, rng))
    a.dt_ns = 1.0; a.pow_floor = getattr(a, "pow_floor", 0.05)
    rows, _ = analyse(np.stack([A, B, Cn, D], 1), [(1, 2), (3, 4), (5, 6), (7, 8)], a,
                      log=lambda *z: None)
    g = {(r["hb1"], r["hb2"]): r for r in rows}
    ab, ac, ad = g[("1-2", "3-4")], g[("1-2", "5-6")], g[("1-2", "7-8")]
    print(f"in-phase pair  : {ab['episodes_ns']}  phase {ab['ep_phase_deg']} (R {ab['ep_phase_R']})")
    print(f"anti-phase pair: {ac['episodes_ns']}  phase {ac['ep_phase_deg']} (R {ac['ep_phase_R']})")
    print(f"independent    : episodes {ad['n_episodes']}")
    ok = (ab["total_ep_ns"] > 3000 and ac["total_ep_ns"] > 3000 and ad["n_episodes"] == 0
          and abs(float(ab["ep_phase_deg"].split(";")[0])) < 30
          and abs(float(ac["ep_phase_deg"].split(";")[0])) > 150)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


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
    p.add_argument("--tdec", type=int, default=10)
    p.add_argument("--dj", type=float, default=1 / 12)
    p.add_argument("--occ-min", type=float, default=0.05)
    p.add_argument("--occ-max", type=float, default=0.95)
    p.add_argument("--min-trans", type=int, default=10, help="min 0->1 transitions")
    p.add_argument("--pow-floor", type=float, default=0.05,
                   help="local band power must be >= this x the signal's median, for both signals")
    p.add_argument("--dms", default=None, help="DMS coupling csv (canonical numbering)")
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a) else 1)

    import pandas as pd
    z0 = np.load(os.path.join(a.indir, f"{a.reps[0]}_hbond.npz"))
    a.dt_ns = float(np.diff(z0["time_ns"])[0])
    meta0 = json.loads(str(z0["meta"]))
    tag = (f"HB_P{a.pmin:g}-{a.pmax:g}_thr{a.thr:g}_ep{a.min_ep:g}_ts{a.tsmooth:g}_dt{a.dt_ns:g}"
           f"_dec{a.tdec}_dj{a.dj:.4g}_occ{a.occ_min:g}-{a.occ_max:g}_tr{a.min_trans}_pf{a.pow_floor:g}"
           f"_da{meta0['d_a_cutoff_A']:g}_ang{meta0['dha_angle_deg']:g}")
    outdir = os.path.join(a.indir, tag); os.makedirs(outdir, exist_ok=True)
    per, metas = {}, {}
    for rep in a.reps:
        z = np.load(os.path.join(a.indir, f"{rep}_hbond.npz"))
        d = np.diff(z["time_ns"])
        if (d <= 0).any() or not np.allclose(d, a.dt_ns):
            sys.exit(f"{rep}: time axis differs or is not uniform; aborting")
        if json.loads(str(z["meta"])) != meta0:
            sys.exit(f"{rep}: H-bond settings differ from {a.reps[0]}; aborting")
        a.t0_ns = float(z["time_ns"][0])
        M = z["hb"]
        occ = M.mean(axis=0)
        trn = np.array([transitions(M[:, k]) for k in range(M.shape[1])])
        use = (occ >= a.occ_min) & (occ <= a.occ_max) & (trn >= a.min_trans)
        labels = list(zip(z["res_i"][use].tolist(), z["res_j"][use].tolist()))
        pd.DataFrame(dict(res_i=z["res_i"], name_i=z["name_i"], res_j=z["res_j"], name_j=z["name_j"],
                          occupancy=occ, transitions=trn, n_atom_pairs=z["n_atom_pairs"], used=use)
                     ).to_csv(os.path.join(outdir, f"{rep}_signals.csv"), index=False)
        print(f"[{rep}] {M.shape[1]} H-bonded residue pairs, {use.sum()} pass the signal filter")
        t0 = time.time()
        rows, meta = analyse(M[:, use], labels, a)
        df = pd.DataFrame(rows); df.to_csv(os.path.join(outdir, f"{rep}_pairs.csv"), index=False)
        per[rep], metas[rep] = df, meta
        print(f"[{rep}] {len(df)} H-bond pairs, {(df.n_episodes > 0).sum()} with episodes, "
              f"{time.time() - t0:.0f} s")

    key = ["hb1", "hb2", "residues", "size"]
    comb = None
    for rep, df in per.items():
        h = df[df.n_episodes > 0][key + ["episodes_ns", "ep_phase_deg", "ep_phase_R"]].copy()
        h[f"cov_ns_{rep}"] = [union_ns([tuple(map(float, x.split("-"))) for x in e.split(";")])
                              for e in h.episodes_ns]
        h = h.rename(columns={c: f"{c}_{rep}" for c in ["episodes_ns", "ep_phase_deg", "ep_phase_R"]})
        comb = h if comb is None else comb.merge(h, on=key, how="outer")
    for r in per:
        comb[f"cov_ns_{r}"] = comb[f"cov_ns_{r}"].fillna(0.0)
    comb["n_reps"] = sum((comb[f"cov_ns_{r}"] > 0).astype(int) for r in per)
    comb["max_cov_ns"] = comb[[f"cov_ns_{r}" for r in per]].max(axis=1)
    if a.dms:
        dms = load_dms(a.dms)
        hit, kind = [], []
        for hb1, hb2, res in zip(comb.hb1, comb.hb2, comb.residues):
            rs = sorted(map(int, res.split(";")))
            within = {tuple(map(int, hb1.split("-"))), tuple(map(int, hb2.split("-")))}
            h_, k_ = [], []
            for x in range(len(rs)):
                for y in range(x + 1, len(rs)):
                    pq = (rs[x], rs[y])
                    if pq in dms:
                        h_.append(f"{pq[0]}-{pq[1]}({dms[pq]})")
                        k_.append("within-Hbond" if pq in within else "across-Hbonds")
            hit.append(";".join(h_)); kind.append(";".join(k_))
        comb["dms_pairs"], comb["dms_kind"] = hit, kind
    comb = comb.sort_values(["n_reps", "max_cov_ns"], ascending=False)
    comb.to_csv(os.path.join(outdir, "hyperedges.csv"), index=False)
    msg = (f"\n{len(comb)} hyperedges (H-bond pairs with >=1 episode); by #replicas: "
           f"{comb.n_reps.value_counts().sort_index().to_dict()}; size: "
           f"{comb['size'].value_counts().sort_index().to_dict()}")
    if a.dms:
        msg += f"; containing a DMS pair: {(comb.dms_pairs != '').sum()}"
    print(msg + f" -> {outdir}")
    with open(os.path.join(outdir, "params.json"), "w") as fh:
        json.dump(dict(args={k: v for k, v in vars(a).items()}, hbond_settings=meta0,
                       meta=metas), fh, indent=1, default=str)


if __name__ == "__main__":
    main()
