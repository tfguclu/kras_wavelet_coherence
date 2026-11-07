#!/usr/bin/env python
"""
kras_hmr_phase.py -- relative phase (lead/lag) of reproducible SASA and H-bond associations.

For every association found in >= 2 replicas at baseline (80-120 ns, coherence 0.75,
500 ns episodes), the smoothed cross-wavelet spectrum C(s,t) = S(W1 W2*) is recomputed
with the unchanged pipeline functions (same scales, smoothing, H-bond power floor).
Its phase is read only inside coherent episodes and only in band cells (and, for H-bonds,
cells passing the power floor).

  phase  circular mean of the complex coherency C/sqrt(P1 P2) over the episode cells,
         per scale; the band value is the circular mean over scales
  lag    per scale: phase x period / 360; band value = median over scales (ns);
         for anti-phase pairs the lag is the offset from 180 deg (signal 1 vs inverted signal 2)
  sign   positive phase/lag = signal 1 leads signal 2
         (SASA: first residue of the pair; H-bond: first bond of "hb1 | hb2")
         -- verified by --selftest on a synthetic lagged pair
  class  in-phase |phase| < 45 deg, anti-phase |phase| > 135 deg, shifted otherwise
  Rbar   resultant length of the per-cell phases (1 = same phase in every cell);
         cells are not independent after smoothing, so Rbar is descriptive

The phase is defined modulo the period (80-120 ns): a lag is interpretable only when it
is well below half a period (about 40-60 ns). Consistency is judged across episodes
and replicas, not from Rbar.

Inputs : wco_hmr/{rep}_sasa.npz, {rep}_hbond.npz, wco_hmr/averages/associations.csv
         (from kras_hmr_averages.py)
Output : wco_hmr/phase/
  episodes.csv      association x replica x episode: start, end, phase, lag, Rbar, class
  replicas.csv      association x replica (all episodes pooled)
  associations.csv  per association: replicas with episodes, phase per replica, class
                    agreement, sign agreement of the lag, whole-window r sign per replica
  phase.npz         per-scale phase and lag per association x replica (for figures)

  python kras_hmr_phase.py
  python kras_hmr_phase.py --selftest
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kras_hmr_runcheck as rc  # noqa: E402
from kras_hmr_wco import cwt_batch, episodes  # noqa: E402
from kras_hmr_averages import band_setup, transitions, BASE, POW_FLOOR, OCC, MIN_TRANS  # noqa: E402

IN_PHASE, ANTI_PHASE = 45.0, 135.0


# ----------------------------------------------------------------------------- core
def cross_spectrum(x1, x2, B, tdec, floor=None):
    """-> band R(t), complex coherency G(s,t) = C/sqrt(P1 P2), cell mask (s,t) in band."""
    W = cwt_batch(np.vstack([x1, x2]).astype(np.float64), 1.0 * B["dt"], B["scales"])[:, :, ::tdec]
    sm = B["sm"]
    P = np.maximum(sm(np.abs(W) ** 2).real, 0.0)
    C = sm(W[0][None] * np.conj(W[1][None]))[0]
    G = C / np.sqrt(P[0] * P[1] + 1e-30)
    R = np.abs(G) ** 2
    ok = np.ones_like(R, bool)
    if floor is not None:      # same rule as pair_band_scalar: per signal over band cells
        Pb = P[:, B["band"], :]
        f = [floor * np.median(Pb[k][:, B["valid"]]) for k in range(2)]
        ok[B["band"]] = (Pb[0] >= f[0]) & (Pb[1] >= f[1])
    Rb = np.where(ok[B["band"]], R[B["band"]], 0.0).mean(axis=0)
    return Rb, G[B["band"]], ok[B["band"]]


def phase_stats(G, ok, tmask, periods):
    """G, ok: (S_band, Td); tmask: (Td,) -> band phase (deg), median lag (ns), Rbar, per-scale arrays."""
    m = ok & tmask[None, :]
    if not m.any():
        return np.nan, np.nan, np.nan, None, None
    U = np.where(m, G / (np.abs(G) + 1e-30), 0)
    n_s = m.sum(axis=1)
    zs = U.sum(axis=1) / np.maximum(n_s, 1)                 # per-scale mean unit phasor
    ph_s = np.degrees(np.angle(zs)); ph_s[n_s == 0] = np.nan
    lag_s = ph_s / 360.0 * periods
    z = U.sum() / m.sum()
    ph = float(np.degrees(np.angle(z)))
    if abs(ph) > ANTI_PHASE:   # anti-phase: lag = offset from 180 deg (sign of signal 1 lead kept)
        lag_s = ((ph_s % 360) - 180.0) / 360.0 * periods
    return ph, float(np.nanmedian(lag_s)), float(np.abs(z)), ph_s, lag_s


def pclass(ph):
    if not np.isfinite(ph):
        return ""
    a = abs(ph)
    return "in-phase" if a < IN_PHASE else ("anti-phase" if a > ANTI_PHASE else "shifted")


def circ_mean(deg):
    d = np.radians(np.asarray([v for v in deg if np.isfinite(v)]))
    if not len(d):
        return np.nan, np.nan
    z = np.exp(1j * d).mean()
    return float(np.degrees(np.angle(z))), float(np.abs(z))


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="wco_hmr")
    p.add_argument("--reps", nargs="+", default=["namd_3", "namd_3_2", "namd_3_3"])
    p.add_argument("--min-reps", type=int, default=2)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    out = os.path.join(a.indir, "phase"); os.makedirs(out, exist_ok=True)
    tdec = BASE["tdec"]; min_ep = rc.eff_ep(BASE)
    AS = pd.read_csv(os.path.join(a.indir, "averages", "associations.csv"))
    AS = AS[AS.observable.isin(["sasa", "hb"]) & (AS.replicas >= a.min_reps)]

    E, RR, blob = [], [], {}
    # ---------------- signals
    sigs = {}
    for r in a.reps:
        z = np.load(os.path.join(a.indir, f"{r}_sasa.npz"))
        x = z["sasa"].astype(np.float64); dt = float(np.diff(z["time_ns"])[0])
        B = band_setup(x.shape[0], dt, tdec); B["dt"] = dt
        sigs[("sasa", r)] = (x, {int(v): k for k, v in enumerate(z["resid"])}, B, float(z["time_ns"][0]))
        z = np.load(os.path.join(a.indir, f"{r}_hbond.npz"))
        M = z["hb"]; dt = float(np.diff(z["time_ns"])[0])
        occ = M.mean(axis=0); trn = np.array([transitions(M[:, k]) for k in range(M.shape[1])])
        use = (occ >= OCC[0]) & (occ <= OCC[1]) & (trn >= MIN_TRANS)
        lab = [tuple(sorted(t)) for t in zip(z["res_i"].tolist(), z["res_j"].tolist())]
        B = band_setup(M.shape[0], dt, tdec); B["dt"] = dt
        sigs[("hb", r)] = (M, {lab[k]: k for k in np.where(use)[0]}, B, float(z["time_ns"][0]))

    for e in AS.itertuples():
        o, key = e.observable, e.association
        for r in a.reps:
            X, idx, B, t0 = sigs[(o, r)]
            if o == "sasa":
                i, j = map(int, key.split("-"))
                x1, x2, floor = X[:, idx[i]], X[:, idx[j]], None
            else:
                h1, h2 = [tuple(sorted(map(int, s.strip().split("-")))) for s in key.split("|")]
                if h1 not in idx or h2 not in idx:
                    continue                                     # signal filtered out in this replica
                x1, x2, floor = X[:, idx[h1]], X[:, idx[h2]], POW_FLOOR
            Rb, G, ok = cross_spectrum(x1, x2, B, tdec, floor)
            per = B["periods"][B["band"]]
            ep = episodes((Rb >= BASE["thr"]) & B["valid"], B["dtd"], min_ep)
            fv = B["fvalid"]
            r_all = float(np.corrcoef(x1[fv].astype(float), x2[fv].astype(float))[0, 1])
            t = np.arange(B["Td"]) * B["dtd"]
            allmask = np.zeros(B["Td"], bool)
            for k, (s, en, L) in enumerate(ep):
                tm = (t >= s) & (t < en); allmask |= tm
                ph, lag, rb, _, _ = phase_stats(G, ok, tm, per)
                E.append(dict(observable=o, association=key, label=e.label, replica=r, episode=k + 1,
                              start_ns=s + t0, end_ns=en + t0, length_ns=L, phase_deg=ph, lag_ns=lag,
                              Rbar=rb, cls=pclass(ph)))
            ph, lag, rb, ph_s, lag_s = phase_stats(G, ok, allmask, per)
            RR.append(dict(observable=o, association=key, label=e.label, replica=r, n_episodes=len(ep),
                           episode_ns=float(sum(L for *_, L in ep)), phase_deg=ph, lag_ns=lag, Rbar=rb,
                           cls=pclass(ph), whole_r=r_all))
            if ph_s is not None:
                tag = f"{o}|{key}|{r}"
                blob[f"{tag}|phase_s"] = ph_s.astype(np.float32)
                blob[f"{tag}|lag_s"] = lag_s.astype(np.float32)
                blob[f"{tag}|periods"] = per.astype(np.float32)

    Ed, Rd = pd.DataFrame(E), pd.DataFrame(RR)
    Ed.to_csv(os.path.join(out, "episodes.csv"), index=False, float_format="%.4g")
    Rd.to_csv(os.path.join(out, "replicas.csv"), index=False, float_format="%.4g")

    S = []
    for (o, key), g in Rd.groupby(["observable", "association"], sort=False):
        w = g[g.n_episodes > 0]
        eg = Ed[(Ed.observable == o) & (Ed.association == key)]
        cm, cR = circ_mean(w.phase_deg)
        cls = sorted(set(w.cls))
        lags = w.lag_ns.values
        S.append(dict(observable=o, association=key, label=g.label.iloc[0], reps_with_episodes=len(w),
                      n_episodes=len(eg),
                      phase_per_replica="/".join(f"{v:.0f}" if np.isfinite(v) else "-" for v in g.phase_deg),
                      lag_per_replica_ns="/".join(f"{v:.1f}" if np.isfinite(v) else "-" for v in g.lag_ns),
                      phase_circmean_deg=cm, phase_between_rep_Rbar=cR,
                      cls="/".join(cls), class_agree=len(cls) == 1,
                      lag_sign_agree=bool(len(lags) >= 2 and (np.all(lags > 0) or np.all(lags < 0))),
                      episode_phase_range_deg=(f"{eg.phase_deg.min():.0f} to {eg.phase_deg.max():.0f}"
                                               if len(eg) else ""),
                      whole_r_per_replica="/".join(f"{v:.2f}" for v in g.whole_r)))
    S = pd.DataFrame(S)
    S.to_csv(os.path.join(out, "associations.csv"), index=False, float_format="%.4g")
    blob["meta"] = np.array(json.dumps(dict(base=BASE, pow_floor=POW_FLOOR, reps=a.reps,
                                            sign="positive = signal 1 leads signal 2",
                                            in_phase=IN_PHASE, anti_phase=ANTI_PHASE)))
    np.savez_compressed(os.path.join(out, "phase.npz"), **blob)
    with pd.option_context("display.width", 230, "display.max_rows", 100, "display.max_colwidth", 30):
        print(S.drop(columns=["label"]).to_string(index=False))
        print()
        print(S.groupby(["observable", "cls"]).size().to_string())
    print(f"wrote {out}/")


# ----------------------------------------------------------------------------- selftest
def selftest():
    """Shared 100 ns oscillation; signal 2 lags signal 1 by 12 ns -> phase ~ +43 deg, lag ~ +12 ns.
    Anti-phase pair -> |phase| ~ 180."""
    rng = np.random.default_rng(1)
    T, dt, per, tau = 10000, 1.0, 100.0, 12.0
    t = np.arange(T) * dt
    amp = 1 + 0.3 * np.sin(2 * np.pi * t / 3000)
    s1 = amp * np.sin(2 * np.pi * t / per) + rng.normal(0, 0.5, T)
    s2 = amp * np.sin(2 * np.pi * (t - tau) / per) + rng.normal(0, 0.5, T)
    s3 = -amp * np.sin(2 * np.pi * t / per) + rng.normal(0, 0.5, T)
    B = band_setup(T, dt, BASE["tdec"]); B["dt"] = dt
    per_b = B["periods"][B["band"]]
    ok_all = True
    for name, y, want_ph in [("lagged +12 ns", s2, 360 * tau / per), ("anti-phase", s3, 180.0)]:
        Rb, G, ok = cross_spectrum(s1, y, B, BASE["tdec"])
        ph, lag, rb, _, _ = phase_stats(G, ok, B["valid"], per_b)
        d = (ph - want_ph + 180) % 360 - 180
        good = abs(d) < 8 and rb > 0.9
        ok_all &= good
        print(f"{name:14s} phase {ph:7.1f} (expect {want_ph:5.1f})  lag {lag:6.1f} ns  Rbar {rb:.3f}  "
              f"mean R {Rb[B['valid']].mean():.2f}  {'ok' if good else 'FAIL'}")
    print("sign convention: positive phase = signal 1 leads signal 2")
    print("SELFTEST", "PASS" if ok_all else "FAIL")
    return ok_all


if __name__ == "__main__":
    main()
