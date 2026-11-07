#!/usr/bin/env python
"""
kras_hmr_wco.py  --  Stage 2: wavelet coherence on continuous chi angles.

Input : wco_hmr/{rep}_chi.npz   (from kras_hmr_prep.py)
Output: wco_hmr/P{band}_thr{}_ep{}_ts{}/   (one dir per parameter set)
          {rep}_pairs.csv          all inter-residue chi pairs
          {rep}_chi_quality.csv    per-chi circ. variance, jump count, band power
          episodes_all_reps.csv    chi pairs with >=1 episode in any replica
          residue_pairs.csv        residue-pair aggregation (rule in aggregate_residues)
          params.json

Method
------
* Each chi -> 2-vector signal (cos chi, sin chi); no rotamer classification.
  Symmetric sp2 terminals (ASP chi2, GLU chi3, PHE chi2, TYR chi2) use 2*chi
  so that 0 == 180 deg.
* Morlet CWT (w0 = 6) of cos and sin; Torrence & Webster smoothing
  (Gaussian in time, sigma = scale; boxcar 0.6 octave in scale).
* Pair coherence = wavelet RV coefficient of the smoothed 2x2 cross-spectral
  matrices:  R = sum_kl |S(W_ik W_jl*)|^2 / sqrt(D_i D_j),
  D_i = sum_kl |S(W_ik W_il*)|^2.  Bounded [0,1]; =1 for identical signals;
  invariant to a constant angle offset.
* Band-collapsed coherence R_band(t) = mean over scales with period in band.
* Edge exclusion (operational rule, reduces but does not eliminate boundary
  influence): sqrt(2)*s + 2*tsmooth*s, with s = largest scale entering the
  band through the scale boxcar.
* Episode = contiguous run with R_band(t) >= thr lasting >= min_ep ns.
* Replicas are analysed independently (never concatenated).

Run from the working directory:
  python kras_hmr_wco.py                         # 80-120 ns, thr 0.75, 250 ns
  python kras_hmr_wco.py --thr 0.8 --min-ep 300
  python kras_hmr_wco.py --selftest
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.fft as sfft
from scipy.ndimage import uniform_filter1d

W0 = 6.0
FF = 4 * np.pi / (W0 + np.sqrt(2 + W0 ** 2))  # Fourier factor: period = FF*scale
SYMMETRIC = {("ASP", 2), ("GLU", 3), ("PHE", 2), ("TYR", 2)}


def next_pow2(n):
    return 1 << int(np.ceil(np.log2(n)))


# ---------------------------------------------------------------- signals
def chi_to_signals(angles_deg, resname, chi):
    """(T, n) degrees -> (n, 2, T) [cos, sin]; symmetric chis doubled."""
    th = np.radians(angles_deg.astype(np.float64)).T.copy()  # (n, T)
    for k, (rn, c) in enumerate(zip(resname, chi)):
        if (str(rn), int(c)) in SYMMETRIC:
            th[k] *= 2.0
    return np.stack([np.cos(th), np.sin(th)], axis=1)


def quality(angles_deg, resname, chi, dt_ns=1.0, lag_ns=10.0):
    """Descriptors only (not rotamer-transition counts):
    circ_var : circular variance of the effective angle
    disp95   : 95th percentile of |chi(t+lag) - chi(t)| in original-angle
               degrees (max 180; max 90 for symmetry-doubled angles)."""
    lag = max(1, int(round(lag_ns / dt_ns)))
    th = np.radians(angles_deg.astype(np.float64))
    fac = np.ones(th.shape[1])
    for k, (rn, c) in enumerate(zip(resname, chi)):
        if (str(rn), int(c)) in SYMMETRIC:
            fac[k] = 2.0
    th = th * fac
    circ_var = 1.0 - np.abs(np.exp(1j * th).mean(axis=0))
    d = np.degrees(np.angle(np.exp(1j * (th[lag:] - th[:-lag])))) / fac
    disp95 = np.percentile(np.abs(d), 95, axis=0)
    return circ_var, disp95


# ---------------------------------------------------------------- CWT
def make_scales(pmin, pmax, dj, margin_oct):
    smin = pmin / FF / 2 ** margin_oct
    smax = pmax / FF * 2 ** margin_oct
    n = int(np.ceil(np.log2(smax / smin) / dj)) + 1
    return smin * 2.0 ** (np.arange(n) * dj)


def cwt_batch(x, dt, scales):
    """x: (m, T) real -> (m, S, T) complex Morlet CWT, scaled by s^-1/2."""
    m, T = x.shape
    npad = next_pow2(2 * T)
    xp = np.zeros((m, npad))
    xp[:, :T] = x - x.mean(axis=1, keepdims=True)
    X = np.fft.fft(xp, axis=1)
    w = 2 * np.pi * np.fft.fftfreq(npad, d=dt)
    out = np.empty((m, len(scales), T), dtype=np.complex64)
    for k, s in enumerate(scales):
        psi = (np.pi ** -0.25 * np.sqrt(2 * np.pi * s / dt)
               * np.exp(-0.5 * (s * w - W0) ** 2) * (w > 0))
        out[:, k, :] = np.fft.ifft(X * psi, axis=1)[:, :T] / np.sqrt(s)
    return out


class Smoother:
    """Torrence & Webster smoothing on decimated grid (dtd ns/sample)."""

    def __init__(self, scales, Td, dtd, dj, tfac=1.0):
        self.Td = Td
        sig = tfac * scales / dtd  # Gaussian sigma in samples
        self.npad = next_pow2(Td + int(6 * sig.max()) + 1)
        f = np.fft.fftfreq(self.npad)
        self.G = np.exp(-2 * np.pi ** 2 * sig[:, None] ** 2 * f[None, :] ** 2
                        ).astype(np.float32)
        self.nbox = max(1, int(round(0.6 / dj)))

    def __call__(self, A):
        """A: (..., S, Td) complex -> smoothed, same shape."""
        sh = A.shape
        P = sfft.fft(A.astype(np.complex64), n=self.npad, axis=-1, workers=-1)
        P = sfft.ifft(P * self.G, axis=-1, workers=-1)[..., :self.Td]
        re = uniform_filter1d(P.real, self.nbox, axis=-2, mode="nearest")
        im = uniform_filter1d(P.imag, self.nbox, axis=-2, mode="nearest")
        return re + 1j * im


def self_D(W, sm):
    """W: (n,2,S,Td) -> D = sum_kl |S(W_k W_l*)|^2 and power P = S|W_c|^2+S|W_s|^2."""
    c00 = sm(np.abs(W[:, 0]) ** 2)
    c11 = sm(np.abs(W[:, 1]) ** 2)
    c01 = sm(W[:, 0] * np.conj(W[:, 1]))
    D = (np.abs(c00) ** 2 + np.abs(c11) ** 2 + 2 * np.abs(c01) ** 2)
    return D.astype(np.float32), (c00.real + c11.real).astype(np.float32)


def pair_R(Wi, Wj, Di, Dj, sm):
    """Wi (2,S,T); Wj (m,2,S,T); Di (S,T); Dj (m,S,T) -> R (m,S,T)."""
    prod = Wi[None, :, None] * np.conj(Wj[:, None, :])  # (m,2,2,S,T)
    m = prod.shape[0]
    C = sm(prod.reshape((m, 4) + prod.shape[-2:]))
    num = (np.abs(C) ** 2).sum(axis=1)
    return num / np.sqrt(Di[None] * Dj + 1e-30)


# ---------------------------------------------------------------- episodes
def episodes(above, dtd, min_ep):
    """above: bool (T,) -> [(start_ns, end_ns, len_ns)] with len >= min_ep."""
    if not above.any():
        return []
    a = np.concatenate([[0], above.astype(np.int8), [0]])
    d = np.diff(a)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    L = (ends - starts) * dtd
    return [(s * dtd, e * dtd, l) for s, e, l in zip(starts, ends, L) if l >= min_ep]


# ---------------------------------------------------------------- core
def analyse(angles, resid, resname, chi, a, log=print):
    """Return list of row dicts for all inter-residue chi pairs."""
    if not np.isfinite(angles).all():
        raise ValueError("input chi angles contain NaN or infinite values")
    dt, tdec = a.dt_ns, a.tdec
    dtd = dt * tdec
    scales = make_scales(a.pmin, a.pmax, a.dj, 0.3)
    periods = scales * FF
    band = (periods >= a.pmin) & (periods <= a.pmax)
    sig = chi_to_signals(angles, resname, chi)           # (n,2,T)
    n, _, T = sig.shape
    W = cwt_batch(sig.reshape(2 * n, T), dt, scales)     # (2n,S,T)
    W = W[:, :, ::tdec].reshape(n, 2, len(scales), -1)
    Td = W.shape[-1]
    sm = Smoother(scales, Td, dtd, a.dj, a.tsmooth)
    DP = [self_D(W[k:k + 32], sm) for k in range(0, n, 32)]
    D = np.concatenate([x[0] for x in DP])
    Pb = np.concatenate([x[1] for x in DP])[:, band, :].mean(axis=1)  # (n,Td)

    # edge exclusion: COI (sqrt2*s) + 2 sigma of time smoothing, using the
    # largest scale that enters the band through the scale boxcar
    kmax = min(np.where(band)[0].max() + sm.nbox // 2, len(scales) - 1)
    s_edge = scales[kmax]
    edge = np.sqrt(2) * s_edge + 2 * a.tsmooth * s_edge
    t = np.arange(Td) * dtd
    valid = (t >= edge) & (t <= (T - 1) * dt - edge)
    tvalid_ns = valid.sum() * dtd

    cv, disp95 = quality(angles, resname, chi, dt_ns=dt)
    pow_all = np.array([Pb[k][valid].mean() for k in range(n)])
    rows = []
    t0 = time.time()
    for i in range(n - 1):
        js = np.array([j for j in range(i + 1, n) if resid[j] != resid[i]])
        if js.size == 0:
            continue
        for b0 in range(0, js.size, a.chunk):
            jb = js[b0:b0 + a.chunk]
            R = pair_R(W[i], W[jb], D[i], D[jb], sm)       # (m,S,Td)
            Rb = R[:, band, :].mean(axis=1)                # (m,Td)
            for m_, j in enumerate(jb):
                r = Rb[m_]
                ep = episodes((r >= a.thr) & valid, dtd, a.min_ep)
                inep = np.zeros(Td, bool)
                for s0, e0, _ in ep:
                    inep[int(s0 / dtd):int(e0 / dtd)] = True
                rows.append(dict(
                    resid_i=int(resid[i]), resname_i=str(resname[i]), chi_i=int(chi[i]),
                    resid_j=int(resid[j]), resname_j=str(resname[j]), chi_j=int(chi[j]),
                    seqsep=int(abs(resid[j] - resid[i])),
                    mean_R=float(r[valid].mean()),
                    frac_above=float(((r >= a.thr) & valid).sum() * dtd / tvalid_ns),
                    n_episodes=len(ep),
                    total_ep_ns=float(sum(l for *_, l in ep)),
                    max_ep_ns=float(max((l for *_, l in ep), default=0.0)),
                    episodes_ns=";".join(f"{s0:g}-{e0:g}" for s0, e0, _ in ep),
                    pow_i=float(pow_all[i]), pow_j=float(pow_all[j]),
                    pow_i_ep=float(Pb[i][inep].mean()) if ep else np.nan,
                    pow_j_ep=float(Pb[j][inep].mean()) if ep else np.nan,
                    ep_pow=";".join(
                        f"{Pb[i][int(s0/dtd):int(e0/dtd)].mean():.4g}/"
                        f"{Pb[j][int(s0/dtd):int(e0/dtd)].mean():.4g}"
                        for s0, e0, _ in ep),
                    circvar_i=float(cv[i]), circvar_j=float(cv[j]),
                    disp95_i=float(disp95[i]), disp95_j=float(disp95[j]),
                ))
        if i % 25 == 0:
            el = time.time() - t0
            log(f"    chi {i}/{n}  {el:.0f} s")
    chi_tab = dict(resid=resid, resname=resname, chi=chi, circvar=cv,
                   disp95=disp95, band_power=pow_all)
    meta = dict(scales=scales.tolist(), edge_scale=float(s_edge), periods_ns=periods.tolist(),
                band_periods_ns=periods[band].tolist(), coi_edge_ns=float(edge),
                valid_ns=float(tvalid_ns), n_chi=n, dt_dec_ns=dtd)
    return rows, meta, chi_tab


# ---------------------------------------------------------------- selftest
def telegraph(T, dwell, rng, states=(60.0, 180.0, 300.0)):
    x = np.empty(T)
    s = rng.integers(3)
    for t in range(T):
        if rng.random() < 1.0 / dwell:
            s = (s + rng.integers(1, 3)) % 3
        x[t] = states[s]
    return x


def selftest(a):
    rng = np.random.default_rng(1)
    T = 10000
    A = telegraph(T, 50, rng)
    C = telegraph(T, 50, rng)
    B = A + 120 + rng.normal(0, 15, T)          # coupled, offset 120 deg
    B[:3000] = telegraph(3000, 50, rng)          # decoupled for first 3 us
    B2 = B + 77                                  # offset invariance check
    ang = np.stack([A, B, C, B2], axis=1) + rng.normal(0, 10, (T, 4))
    resid = np.array([1, 2, 3, 4]); rn = np.array(["LYS"] * 4); ch = np.array([1] * 4)
    rows, meta, _ = analyse(ang.astype(np.float32), resid, rn, ch, a, log=lambda *x: None)
    get = {(r["resid_i"], r["resid_j"]): r for r in rows}
    ab, ac, ab2 = get[(1, 2)], get[(1, 3)], get[(1, 4)]
    print(f"A-B  coupled 3-10us : mean_R {ab['mean_R']:.2f}  episodes {ab['n_episodes']}"
          f"  total {ab['total_ep_ns']:.0f} ns  [{ab['episodes_ns']}]")
    print(f"A-C  independent    : mean_R {ac['mean_R']:.2f}  episodes {ac['n_episodes']}"
          f"  total {ac['total_ep_ns']:.0f} ns")
    print(f"A-B' offset +77 deg : mean_R {ab2['mean_R']:.2f}  (should equal A-B)")
    ok = (ab["total_ep_ns"] > 3000 and ac["total_ep_ns"] < ab["total_ep_ns"] / 5
          and abs(ab["mean_R"] - ab2["mean_R"]) < 0.02)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------- residues
def _union_ns(intervals):
    tot, cur_s, cur_e = 0.0, None, None
    for s0, e0 in sorted(intervals):
        if cur_e is None or s0 > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s0, e0
        else:
            cur_e = max(cur_e, e0)
    return tot + (cur_e - cur_s if cur_e is not None else 0.0)


def aggregate_residues(per_rep, dtd):
    """Rule: episodes are detected per chi pair first (never stitched across
    chi pairs). Residue pair hit in a replica = >=1 chi pair with an episode.
    Representative chi pair = largest total episode time, ties -> longest
    episode -> lowest chi indices. Coverage = union of intervals over chi pairs.
    Replication reported as any-chi (n_reps_any) and same-chi (n_reps_same_chi)."""
    import pandas as pd
    out = None
    chi_hits = {}
    for rep, df in per_rep.items():
        rows = []
        for (ri, rj), g in df.groupby(["resid_i", "resid_j"]):
            hit = g[g.n_episodes > 0]
            for _, h in hit.iterrows():
                chi_hits.setdefault((ri, rj, h.chi_i, h.chi_j), set()).add(rep)
            if len(hit):
                best = hit.sort_values(["total_ep_ns", "max_ep_ns", "chi_i", "chi_j"],
                                       ascending=[False, False, True, True]).iloc[0]
                iv = [tuple(map(float, x.split("-")))
                      for e in hit.episodes_ns for x in e.split(";")]
                rec = dict(rep_chi=f"chi{best.chi_i}-chi{best.chi_j}",
                           cov_ns=_union_ns(iv), n_chi_hit=len(hit))
            else:
                rec = dict(rep_chi="", cov_ns=0.0, n_chi_hit=0)
            rows.append(dict(resid_i=ri, resname_i=g.resname_i.iloc[0],
                             resid_j=rj, resname_j=g.resname_j.iloc[0],
                             seqsep=g.seqsep.iloc[0], n_chi_combos=len(g),
                             **{f"{k}_{rep}": v for k, v in rec.items()}))
        d = pd.DataFrame(rows)
        key = ["resid_i", "resname_i", "resid_j", "resname_j", "seqsep", "n_chi_combos"]
        out = d if out is None else out.merge(d, on=key, how="outer")
    reps = list(per_rep)
    out["n_reps_any"] = sum((out[f"n_chi_hit_{r}"] > 0).astype(int) for r in reps)
    same, same_pairs = {}, {}
    for (ri, rj, ci, cj), rs in chi_hits.items():
        k, n_ = (ri, rj), len(rs)
        if n_ > same.get(k, 0):
            same[k], same_pairs[k] = n_, [f"chi{ci}-chi{cj}"]
        elif n_ == same.get(k, 0):
            same_pairs[k].append(f"chi{ci}-chi{cj}")
    out["n_reps_same_chi"] = [same.get((a_, b_), 0)
                              for a_, b_ in zip(out.resid_i, out.resid_j)]
    out["same_chi_pairs"] = [";".join(sorted(same_pairs.get((a_, b_), [])))
                             for a_, b_ in zip(out.resid_i, out.resid_j)]
    return out.sort_values(["n_reps_same_chi", "n_reps_any", "resid_i", "resid_j"],
                           ascending=[False, False, True, True])


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="wco_hmr")
    p.add_argument("--reps", nargs="+", default=["namd_3", "namd_3_2", "namd_3_3"])
    p.add_argument("--pmin", type=float, default=80.0, help="band min period (ns)")
    p.add_argument("--pmax", type=float, default=120.0, help="band max period (ns)")
    p.add_argument("--thr", type=float, default=0.75, help="coherence threshold")
    p.add_argument("--min-ep", type=float, default=500.0, help="min episode (ns)")
    p.add_argument("--dt-ns", type=float, default=None,
                   help="frame spacing; default = read from time_ns in the chi npz")
    p.add_argument("--tdec", type=int, default=10,
                   help="time decimation after CWT (10 -> 10 ns grid)")
    p.add_argument("--dj", type=float, default=1 / 12, help="scale step (octaves)")
    p.add_argument("--tsmooth", type=float, default=2.0,
                   help="time-smoothing width in units of scale (T&W = 1)")
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()

    if a.selftest:
        a.dt_ns = a.dt_ns or 1.0
        sys.exit(0 if selftest(a) else 1)

    import pandas as pd
    if a.dt_ns is None:                       # resolve dt before naming the run
        _t = np.load(os.path.join(a.indir, f"{a.reps[0]}_chi.npz"))["time_ns"]
        if len(_t) < 2:
            sys.exit("time_ns too short")
        a.dt_ns = float(_t[1] - _t[0])
    tag = (f"P{a.pmin:g}-{a.pmax:g}_thr{a.thr:g}_ep{a.min_ep:g}_ts{a.tsmooth:g}"
           f"_dt{a.dt_ns:g}_dec{a.tdec}_dj{a.dj:.4g}")
    outdir = os.path.join(a.indir, tag)
    os.makedirs(outdir, exist_ok=True)
    per_rep = {}
    metas = {}
    ref_ids = None
    for rep in a.reps:
        f = os.path.join(a.indir, f"{rep}_chi.npz")
        z = np.load(f)
        tns = z["time_ns"]; d = np.diff(tns)
        if len(d) == 0 or (d <= 0).any() or not np.allclose(d, d[0], rtol=1e-6):
            sys.exit(f"{rep}: time_ns not uniformly increasing; aborting")
        if a.dt_ns is None:
            a.dt_ns = float(d[0])
        elif not np.isclose(a.dt_ns, d[0]):
            sys.exit(f"{rep}: --dt-ns {a.dt_ns} != time_ns spacing {d[0]}; aborting")
        ids = list(zip(z["resid"].tolist(), z["resname"].tolist(), z["chi"].tolist()))
        if ref_ids is None:
            ref_ids = ids
        elif ids != ref_ids:
            sys.exit(f"chi identities in {rep} differ from {a.reps[0]}; aborting")
        print(f"[{rep}] {z['angles'].shape[0]} frames, {z['angles'].shape[1]} chi")
        t0 = time.time()
        rows, meta, chi_tab = analyse(z["angles"], z["resid"], z["resname"], z["chi"], a)
        pd.DataFrame(chi_tab).to_csv(os.path.join(outdir, f"{rep}_chi_quality.csv"),
                                     index=False)
        df = pd.DataFrame(rows)
        out = os.path.join(outdir, f"{rep}_pairs.csv")
        df.to_csv(out, index=False)
        per_rep[rep] = df
        metas[rep] = meta
        print(f"[{rep}] {len(df)} pairs, {(df.n_episodes > 0).sum()} with episodes, "
              f"{time.time() - t0:.0f} s -> {out}")

    # residue-pair aggregation (prespecified rule, see aggregate_residues)
    res = aggregate_residues(per_rep, dtd=a.dt_ns * a.tdec)
    res.to_csv(os.path.join(outdir, "residue_pairs.csv"), index=False)
    print(f"{(res.n_reps_any > 0).sum()} residue pairs with a hit; "
          f"n_reps_any: {res.n_reps_any.value_counts().sort_index().to_dict()}; "
          f"n_reps_same_chi: {res.n_reps_same_chi.value_counts().sort_index().to_dict()}")

    # combined: pairs with >=1 episode in any replica
    key = ["resid_i", "resname_i", "chi_i", "resid_j", "resname_j", "chi_j", "seqsep"]
    cols = ["mean_R", "n_episodes", "total_ep_ns", "max_ep_ns", "episodes_ns"]
    comb = None
    for rep, df in per_rep.items():
        d = df[key + cols].rename(columns={c: f"{c}_{rep}" for c in cols})
        comb = d if comb is None else comb.merge(d, on=key, how="outer")
    comb["n_reps_with_episode"] = sum(
        (comb[f"n_episodes_{r}"] > 0).astype(int) for r in per_rep)
    comb = comb[comb.n_reps_with_episode > 0].sort_values(
        ["n_reps_with_episode", "resid_i", "resid_j"], ascending=[False, True, True])
    out = os.path.join(outdir, "episodes_all_reps.csv")
    comb.to_csv(out, index=False)
    print(f"\n{len(comb)} pairs with >=1 episode; by #replicas: "
          f"{comb.n_reps_with_episode.value_counts().sort_index().to_dict()} -> {out}")

    with open(os.path.join(outdir, "params.json"), "w") as fh:
        json.dump(dict(args=vars(a), meta=metas), fh, indent=1)


if __name__ == "__main__":
    main()
