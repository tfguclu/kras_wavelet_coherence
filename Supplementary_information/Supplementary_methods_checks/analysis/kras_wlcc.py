#!/usr/bin/env python
"""
kras_wlcc.py -- windowed lagged cross-correlation (WLCC) of residue signals, a
time-domain counterpart of the band wavelet coherence used in the manuscript.

WHAT IS COMPUTED
  For two signals x and y, in windows of 500 ns centred every 10 ns, the Pearson
  correlation r(tau) between x(t) and y(t + tau) for |tau| <= 25 ns (band mode) or
  50 ns (broad mode), in 1 ns steps.
  Per window it keeps q = r^2 at the lag with the largest |r| (chi angles: the
  lagged RV coefficient of the (cos chi, sin chi) signals, the same statistic that
  Equation 1 smooths in the wavelet domain), that lag, and the sign of r.
  A positive lag means the first signal leads (same convention as the phase in
  Fig. S2). A WLCC episode is a run of window centres with q >= theta lasting
  >= 500 ns, exactly as a coherence episode is a run of band coherence >= 0.75.

TWO MODES
  band   signals band-passed to 80-120 ns (zero-phase Butterworth); windows are
         mean-centred. Window, step, lag range and episode length follow from the
         wavelet setup. Use: validation of the coherence results in the time domain.
  broad  unfiltered signals; each window (and each lagged window) is linearly
         detrended. Adds lags across all timescales below the window.
  THRESHOLD (both modes): theta is calibrated so that the controls give the same
         rate of episodes per signal pair as the coherence controls at 0.75
         (Fig. S6: chi 0.091%, SASA 0.027%, H-bond 0.059% per control replica).
         The analytic value 0.75 (peak |r| >= 0.866) is reported for comparison only:
         a 500 ns window of 80-120 ns signals holds about 4 independent values and the
         maximum over the lags inflates q, so 0.75 accepts far more than coherence does
         (see selftest --examples).

UNITS (what is tested; one row each in units.csv)
  coherent    every coherent pair of the baseline coherence run
              (--coh-chi residue_pairs.csv, --coh-sasa edges.csv, --coh-hb hyperedges.csv)
  dms         DMS couplings (|eps| > RT and 95% width < 1 kcal/mol, as kras_overlap_null.py),
              for every observable in which the residue pair is testable
  background  for every DMS unit, --n-bg non-DMS residue pairs testable for the same
              observable and in the same 2 A bin of C-alpha distance
  calib       --calib random signal pairs per observable, used only to calibrate theta
  A residue-pair unit tests all its signal pairs (chi: every chi-angle pair; H-bond
  DMS/background units: every pair of used H-bond signals with one bond on each residue)
  and is detected in a replica when any of its signal pairs has an episode, as for chi
  in the coherence analysis. A coherent H-bond unit is its own two H-bond signals.

CONTROLS  (the manuscript's design, Section 5.4; run only where needed)
  original    kras_hmr_coherence_null.py (circshift-v2): SeedSequence([7, set, crc32(replica),
              observable index]), one offset in [1 us, T - 1 us] per signal, np.roll; sets 1..200.
              Per pair: k = sets in which the unit reaches its real number of replicas,
              p = (1 + k)/(1 + sets); a pair in >= 2 replicas is set aside when k >= 1.
              Screened set (DMS units): units in >= 1, >= 2 and 3 replicas, real against each control.
  >= 1 us     from the same runs, counting only signal pairs whose offsets are >= 1 us apart
              (screened-set counts rescaled by the fraction of signal pairs passing).
  seam masked from the same runs, +-722 ns around each signal's seam removed from control and real
              series; counted when the control reaches max(real masked replicas, 2).
  per residue kras_hmr_controls_review.py reswise: SeedSequence([7, set, crc32(replica), 99]),
              one offset per residue for all its chi angles; 200 sets, chi only.
  As in the manuscript, pairs are set aside on the original controls only; the others test
  robustness. band/broad run 20 original sets for the calibration pairs only (threshold);
  confirm runs everything above for the DMS units if any is found in >= 2 replicas.
  Every signal gets the same offset as in the coherence controls. One
  difference: which H-bond signals are used is taken from the real data, whereas the
  pipeline re-applied the occupancy/transition filter to each shifted copy (the shift
  can change a transition count by one at the joint).

INPUTS (outputs of the existing pipeline; nothing else is imported)
  <indir>/{rep}_chi.npz    angles (T, n) deg, resid, resname, chi, time_ns
  <indir>/{rep}_sasa.npz   sasa (T, n), resid, time_ns
  <indir>/{rep}_hbond.npz  hb (T, n) 0/1, res_i, res_j, time_ns
  coherent-pair tables, DMS table (kras_coupling_edges_full_canonical.csv) and a PDB
  for C-alpha distances (default <indir>/namd_3_2_wrapped.pdb, as in the DMS null).

USAGE  (from the analysis folder that holds wco_hmr/; inputs are found automatically)
  python kras_wlcc.py test            # self-test, incl. wco_hmr/pairsignals/signals.npz if present
  python kras_wlcc.py band            # band mode  -> wlcc_band/report.txt
  python kras_wlcc.py broad           # broad mode -> wlcc_broad/report.txt
  python kras_wlcc.py confirm broad   # the manuscript's control analyses for the DMS units, if needed
  band/broad: real data, 20 control sets for the calibration pairs only (for the
  threshold), summary. confirm: see CONTROLS.
  Every input can be overridden (--indir, --coh-chi, --dms, --pdb, ...); see -h of each command.
  Reruns reuse finished results; an interrupted run continues where it stopped.

Requires numpy, scipy, pandas. Uses (CPU count - 2) processes by default (--jobs).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")                  # parallel work is done by --jobs processes
import argparse
import glob
import hashlib
import json
import sys
import time
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt

OBS = ("chi", "sasa", "hb")
FILES = {"chi": ("chi", "angles"), "sasa": ("sasa", "sasa"), "hb": ("hbond", "hb")}
KIND_INDEX = {"chi": 0, "sasa": 1, "hb": 2}           # order of KINDS in kras_hmr_coherence_null.py
SYMMETRIC = {("ASP", 2), ("GLU", 3), ("PHE", 2), ("TYR", 2)}
RT = 0.001987 * 303
TARGET_RATE = {"chi": 0.00091, "sasa": 0.00027, "hb": 0.00059}   # Fig. S6, controls at 0.75
# parameters that change the numbers; a change invalidates stored results
COMPUTE_KEYS = ["mode", "pmin", "pmax", "order", "window_ns", "step_ns", "max_lag_ns", "lag_step_ns",
                "edge_ns", "pow_floor", "hb_occ", "hb_min_trans", "sasa_min_sd", "ctrl_seed",
                "min_shift_ns", "reps"]


# ============================================================================ signals
def load_replica(indir, rep):
    """raw signals of one replica, exactly as stored by the pipeline"""
    out = {}
    z = np.load(os.path.join(indir, f"{rep}_chi.npz"), allow_pickle=True)
    out["chi"] = z["angles"]
    out["chi_resid"] = z["resid"].astype(int)
    out["chi_resname"] = z["resname"].astype(str)
    out["chi_idx"] = z["chi"].astype(int)
    t = z["time_ns"]
    z = np.load(os.path.join(indir, f"{rep}_sasa.npz"), allow_pickle=True)
    out["sasa"] = z["sasa"]
    out["sasa_resid"] = z["resid"].astype(int)
    if not np.allclose(z["time_ns"], t):
        raise ValueError(f"{rep}: chi and SASA time axes differ")
    z = np.load(os.path.join(indir, f"{rep}_hbond.npz"), allow_pickle=True)
    out["hb"] = z["hb"]
    out["hb_i"], out["hb_j"] = z["res_i"].astype(int), z["res_j"].astype(int)
    if not np.allclose(z["time_ns"], t):
        raise ValueError(f"{rep}: chi and H-bond time axes differ")
    d = np.diff(t)
    if not np.allclose(d, d[0]):
        raise ValueError(f"{rep}: time axis is not uniform")
    out["dt"], out["T"] = float(d[0]), len(t)
    return out


def chi_signals(deg, resname, chi):
    """(T, n) degrees -> (T, n, 2) [cos, sin]; groups symmetric under a 180 deg flip are doubled"""
    th = np.radians(np.asarray(deg, float))
    dbl = np.array([(r, int(c)) in SYMMETRIC for r, c in zip(resname, chi)], bool)
    th[:, dbl] *= 2.0
    return np.stack([np.cos(th), np.sin(th)], axis=2)


def bandpass(X, dt, pmin, pmax, order):
    """zero-phase Butterworth band-pass along time (axis 0); zero phase so that no lag is added"""
    sos = butter(order, [1.0 / pmax, 1.0 / pmin], btype="band", fs=1.0 / dt, output="sos")
    return sosfiltfilt(sos, X, axis=0, padtype="odd", padlen=min(X.shape[0] - 1, 1000))


def prepare(obs, raw_cols, P, dt, resname=None, chi=None):
    """raw columns (T, m) -> analysis signals (T, m, d), d = 2 for chi, 1 otherwise"""
    if obs == "chi":
        Z = chi_signals(raw_cols, resname, chi)
    else:
        Z = np.asarray(raw_cols, float)[:, :, None]
    if P["mode"] == "band":
        Z = bandpass(Z, dt, P["pmin"], P["pmax"], P["order"])
    return Z - Z.mean(axis=0, keepdims=True)          # centring improves the precision of the sums


def ctrl_offsets(seed, k, rep, obs, T, n, ms, scheme="original", resid=None):
    """offsets of control set k, identical to the manuscript's controls:
    original  kras_hmr_coherence_null.make_surrogate: one offset per signal (column)
    reswise   kras_hmr_controls_review.make_reswise (chi only): one offset per residue"""
    if scheme == "reswise":
        rng = np.random.default_rng(np.random.SeedSequence([seed, k, zlib.crc32(rep.encode()), 99]))
        res = np.asarray(resid).astype(int)
        ures = np.unique(res)
        off_res = dict(zip(ures.tolist(), rng.integers(ms, T - ms, size=len(ures))))
        return np.array([off_res[r] for r in res.tolist()])
    rng = np.random.default_rng(np.random.SeedSequence([seed, k, zlib.crc32(rep.encode()), KIND_INDEX[obs]]))
    return rng.integers(ms, T - ms, size=n)


# ============================================================================ WLCC core
def make_geometry(T, dt, P, centers_ns=None):
    w = int(round(P["window_ns"] / dt))
    half = w // 2
    w = 2 * half
    L = int(round(P["max_lag_ns"] / dt))
    ls = max(1, int(round(P["lag_step_ns"] / dt)))
    if centers_ns is None:                          # same 10 ns grid and valid range as the coherence
        grid = np.arange(0.0, (T - 1) * dt + 1e-9, P["step_ns"])
        centers_ns = grid[(grid >= P["edge_ns"]) & (grid <= (T - 1) * dt - P["edge_ns"])]
    c = np.round(np.asarray(centers_ns, float) / dt).astype(int)
    if ((c - half - L) < 0).any() or ((c + half + L) > T).any():
        raise ValueError("window + lag reach beyond the record: raise edge_ns or shorten window/lag")
    lags = [0]
    for k in range(ls, L + 1, ls):                  # lags ordered by |tau|: ties go to the smaller lag
        lags += [k, -k]
    m = int(np.ceil(P["min_ep_ns"] / P["step_ns"] - 1e-9))
    return dict(starts=c - half, w=w, centers_ns=np.asarray(centers_ns, float), lags=np.array(lags),
                dt=dt, m=m, step_ns=float(P["step_ns"]), detrend=(P["mode"] == "broad"))


def _cs(a):
    out = np.zeros((a.shape[0] + 1,) + a.shape[1:])
    np.cumsum(a, axis=0, out=out[1:])
    return out


def wlcc(X, Y, G, gate=False, pow_floor=0.05):
    """
    X, Y: (T, nb, d) centred signals, column b of X is paired with column b of Y.
    Returns q (nb, nc), lag_ns (nb, nc), sign (nb, nc), all at the lag of maximal q.
    Window sums come from cumulative sums, so each lag costs O(T) per pair; with
    linear detrending (broad mode) the residual covariance is
        cov = Sxy - Sx Sy / w - Sx~ Sy~ / Stt,   Sz~ = sum (t - t_mean) z,  Stt = w (w^2 - 1) / 12.
    """
    T, nb, d = X.shape
    s0, w = G["starts"], G["w"]
    det = 1.0 if G["detrend"] else 0.0
    Stt = w * (w * w - 1) / 12.0
    t = np.arange(T, dtype=float)[:, None]

    def ws(C, s):
        return C[s + w] - C[s]

    Sx = [ws(_cs(X[:, :, k]), s0) for k in range(d)]
    tbx = (s0 + (w - 1) / 2.0)[:, None]
    Sxc = [ws(_cs(t * X[:, :, k]), s0) - tbx * Sx[k] for k in range(d)]
    Vx = {}
    for k in range(d):
        for l in range(k, d):
            Vx[k, l] = ws(_cs(X[:, :, k] * X[:, :, l]), s0) - Sx[k] * Sx[l] / w - det * Sxc[k] * Sxc[l] / Stt
    trx = Vx[0, 0] ** 2 + ((Vx[1, 1] ** 2 + 2 * Vx[0, 1] ** 2) if d == 2 else 0.0)

    CY = [_cs(Y[:, :, k]) for k in range(d)]
    CtY = [_cs(t * Y[:, :, k]) for k in range(d)]
    CYY = {(k, l): _cs(Y[:, :, k] * Y[:, :, l]) for k in range(d) for l in range(k, d)}

    def yside(s):
        Sy = [ws(CY[k], s) for k in range(d)]
        tby = (s + (w - 1) / 2.0)[:, None]
        Syc = [ws(CtY[k], s) - tby * Sy[k] for k in range(d)]
        Vy = {kl: ws(CYY[kl], s) - Sy[kl[0]] * Sy[kl[1]] / w - det * Syc[kl[0]] * Syc[kl[1]] / Stt
              for kl in CYY}
        return Sy, Syc, Vy

    if gate:                                        # H-bonds: both signals must change in the window
        fx = pow_floor * np.median(Vx[0, 0], axis=0)
        fy = pow_floor * np.median(yside(s0)[2][0, 0], axis=0)

    nc = len(s0)
    bq = np.zeros((nc, nb))
    bl = np.zeros((nc, nb), np.int16)
    bs = np.zeros((nc, nb), np.int8)
    # cross sums per lag: products are summed in blocks of B frames (B = window step) when every
    # window starts on a block boundary; the cumulative sum then runs over blocks, not frames
    lo, hi = int(s0.min()), int(s0.max()) + w
    B = int(round(G["step_ns"] / G["dt"]))
    blocks = B > 1 and (s0 - lo).min() >= 0 and ((s0 - lo) % B == 0).all() and w % B == 0 and (hi - lo) % B == 0
    if not blocks:
        B = 1
    ib, wb = (s0 - lo) // B, w // B
    Xr = [np.ascontiguousarray(X[lo:hi, :, k]) for k in range(d)]       # contiguous components
    Yc = [np.ascontiguousarray(Y[:, :, l]) for l in range(d)]
    for tau in G["lags"]:
        Sy, Syc, Vy = yside(s0 + tau)
        tr_y = Vy[0, 0] ** 2 + ((Vy[1, 1] ** 2 + 2 * Vy[0, 1] ** 2) if d == 2 else 0.0)
        num = np.zeros((nc, nb))
        c00 = None
        for l in range(d):
            for k in range(d):
                Yr = Yc[l][lo + tau:hi + tau]
                if B > 1:
                    prod = np.einsum("ijk,ijk->ik", Xr[k].reshape(-1, B, nb), Yr.reshape(-1, B, nb))
                else:
                    prod = Xr[k] * Yr
                C = _cs(prod)
                cov = (C[ib + wb] - C[ib]) - Sx[k] * Sy[l] / w - det * Sxc[k] * Syc[l] / Stt
                num += cov * cov
                if k == 0 and l == 0:
                    c00 = cov
        den = np.sqrt(np.maximum(trx * tr_y, 0.0))
        ok = den > 1e-12 * (1.0 + np.abs(num))
        q = np.where(ok, num / np.where(ok, den, 1.0), 0.0)
        q = np.clip(q, 0.0, 1.0)
        if gate:
            q = np.where((Vx[0, 0] >= fx) & (Vy[0, 0] >= fy), q, 0.0)
        upd = q > bq
        bq[upd] = q[upd]
        bl[upd] = tau
        if d == 1:
            bs[upd] = np.sign(c00[upd]).astype(np.int8)
    return bq.T.astype(np.float32), (bl.T * G["dt"]).astype(np.float32), bs.T


def sustained(q, m):
    """largest theta at which q contains an episode: max over runs of m points of the run minimum"""
    q = np.atleast_2d(q)
    if q.shape[1] < m:
        return np.full(q.shape[0], np.nan, np.float32)
    return sliding_window_view(q, m, axis=1).min(axis=2).max(axis=1).astype(np.float32)


def runs(mask, m):
    """[(start_idx, end_idx_exclusive)] of True runs with >= m points"""
    a = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(a)
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(s, e) for s, e in zip(st, en) if e - s >= m]


# ============================================================================ units
def read_ca(pdb):
    X = {}
    for line in open(pdb):
        if line.startswith("ENDMDL"):
            break
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            X[int(line[22:26])] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return X


def load_dms(path):
    d = pd.read_csv(path)
    d = d[(d["mean_kcal/mol"].abs() > RT) & (d["ci95_kcal/mol"] < 1)].copy()
    d["i"] = d[["pos_i_can", "pos_j_can"]].min(axis=1).astype(int)
    d["j"] = d[["pos_i_can", "pos_j_can"]].max(axis=1).astype(int)
    return {(i, j): "+".join(sorted(set(g.trait))) for (i, j), g in d.groupby(["i", "j"])}


def build_units(a, R):
    """units (list of dicts) and the registry of signal pairs per (observable, replica)"""
    reps = a.reps
    r0 = R[reps[0]]
    for rep in reps[1:]:
        r = R[rep]
        if (r["T"] != r0["T"] or not np.array_equal(r["chi_resid"], r0["chi_resid"])
                or not np.array_equal(r["chi_idx"], r0["chi_idx"])
                or not np.array_equal(r["sasa_resid"], r0["sasa_resid"])):
            raise ValueError(f"{rep}: chi/SASA columns or length differ from {reps[0]}")
    chi_cols = defaultdict(list)
    for c, r in enumerate(r0["chi_resid"]):
        chi_cols[int(r)].append(c)
    sasa_col = {int(r): c for c, r in enumerate(r0["sasa_resid"])}
    sd = np.min([R[rep]["sasa"].std(axis=0) for rep in reps], axis=0)
    sasa_ok = {int(r) for r, s in zip(r0["sasa_resid"], sd) if s >= a.sasa_min_sd}
    hb_col, hb_used, bonds_of = {}, {}, {}
    for rep in reps:
        M = R[rep]["hb"]
        occ = M.mean(axis=0)
        trn = ((M[1:] == 1) & (M[:-1] == 0)).sum(axis=0)
        use = (occ >= a.hb_occ[0]) & (occ <= a.hb_occ[1]) & (trn >= a.hb_min_trans)
        bonds = list(zip(R[rep]["hb_i"].tolist(), R[rep]["hb_j"].tolist()))
        hb_col[rep] = {b: c for c, b in enumerate(bonds)}
        hb_used[rep] = {b for b, u in zip(bonds, use) if u}
        bonds_of[rep] = defaultdict(list)
        for b in sorted(hb_used[rep]):
            bonds_of[rep][b[0]].append(b)
            bonds_of[rep][b[1]].append(b)

    reg = {(o, rep): {} for o in OBS for rep in reps}
    units = []

    def add(o, set_, i, j, sp, label, **extra):
        u = dict(uid=len(units), obs=o, set=set_, res_i=i, res_j=j, label=label)
        u.update(extra)
        for rep in reps:
            ids = [reg[(o, rep)].setdefault(p, len(reg[(o, rep)])) for p in sp.get(rep, [])]
            u[f"sp_{rep}"] = ";".join(map(str, ids))
        units.append(u)
        return u

    def chi_sp(i, j):
        p = [(x, y) for x in chi_cols.get(i, []) for y in chi_cols.get(j, [])]
        return {rep: p for rep in reps}

    def sasa_sp(i, j, check=True):
        if i not in sasa_col or j not in sasa_col:
            return {}
        if check and (i not in sasa_ok or j not in sasa_ok or abs(i - j) < 3):
            return {}
        return {rep: [(sasa_col[i], sasa_col[j])] for rep in reps}

    def hb_sp_res(i, j):
        return {rep: [(hb_col[rep][A], hb_col[rep][B]) for A in bonds_of[rep].get(i, [])
                      for B in bonds_of[rep].get(j, []) if A != B] for rep in reps}

    def n_rep(sp):
        return sum(len(v) > 0 for v in sp.values())

    # ---- coherent pairs of the baseline coherence run
    if a.coh_chi:
        df = pd.read_csv(a.coh_chi)
        for r in df[df.n_reps_any > 0].itertuples():
            add("chi", "coherent", int(r.resid_i), int(r.resid_j), chi_sp(int(r.resid_i), int(r.resid_j)),
                f"{r.resid_i}-{r.resid_j}", coh_reps=int(r.n_reps_any))
    if a.coh_sasa:
        df = pd.read_csv(a.coh_sasa)
        for r in df.itertuples():
            add("sasa", "coherent", int(r.resid_i), int(r.resid_j),
                sasa_sp(int(r.resid_i), int(r.resid_j), check=False), f"{r.resid_i}-{r.resid_j}",
                coh_reps=int(r.n_reps_any))
    if a.coh_hb:
        df = pd.read_csv(a.coh_hb)
        for r in df.itertuples():
            b1 = tuple(sorted(map(int, str(r.hb1).split("-"))))
            b2 = tuple(sorted(map(int, str(r.hb2).split("-"))))
            sp = {rep: ([(hb_col[rep][b1], hb_col[rep][b2])]
                        if b1 in hb_used[rep] and b2 in hb_used[rep] else []) for rep in reps}
            add("hb", "coherent", -1, -1, sp, f"{r.hb1} | {r.hb2}", coh_reps=int(r.n_reps))

    # ---- DMS couplings and their distance-matched background
    rng = np.random.default_rng(a.unit_seed)
    if a.dms:
        dms = load_dms(a.dms)
        CA = read_ca(a.pdb)
        prot = sorted(r for r in sasa_col if r in CA)
        dist = lambda i, j: float(np.linalg.norm(CA[i] - CA[j]))
        dms_units = []
        for (i, j), tr in sorted(dms.items()):
            if i not in CA or j not in CA:
                continue
            dd = dist(i, j)
            b = int(dd // a.bin_A)
            for o, sp in (("chi", chi_sp(i, j)), ("sasa", sasa_sp(i, j)), ("hb", hb_sp_res(i, j))):
                if n_rep(sp) >= 2 and sum(len(v) for v in sp.values()) > 0:
                    dms_units.append(add(o, "dms", i, j, sp, f"{i}-{j}", trait=tr, dist_A=round(dd, 2), bin=b))
        pool = {o: defaultdict(list) for o in OBS}
        for x in range(len(prot)):
            for y in range(x + 1, len(prot)):
                i, j = prot[x], prot[y]
                if (i, j) in dms:
                    continue
                b = int(dist(i, j) // a.bin_A)
                if chi_cols.get(i) and chi_cols.get(j):
                    pool["chi"][b].append((i, j))
                if i in sasa_ok and j in sasa_ok and j - i >= 3:
                    pool["sasa"][b].append((i, j))
                if n_rep(hb_sp_res(i, j)) >= 2:
                    pool["hb"][b].append((i, j))
        for u in dms_units:
            cand = pool[u["obs"]].get(u["bin"], [])
            if not cand:
                continue
            pick = rng.choice(len(cand), size=min(a.n_bg, len(cand)), replace=False)
            for k in sorted(pick):
                i, j = cand[k]
                sp = {"chi": chi_sp, "sasa": sasa_sp, "hb": hb_sp_res}[u["obs"]](i, j)
                add(u["obs"], "background", i, j, sp, f"{i}-{j}", bg_of=u["uid"],
                    dist_A=round(dist(i, j), 2), bin=u["bin"])

    # ---- calibration: random distinct single signal pairs
    def draw(n, make):
        seen, out, tries = set(), [], 0
        while len(out) < n and tries < 50 * n:
            tries += 1
            x = make()
            if x is not None and x[0] not in seen:
                seen.add(x[0])
                out.append(x)
        return out

    if a.calib > 0:
        allc = [(c, r) for r in sorted(chi_cols) for c in chi_cols[r]]

        def mk_chi():
            x, y = rng.choice(len(allc), 2, replace=False)
            (cx, rx), (cy, ry) = sorted([allc[x], allc[y]], key=lambda v: v[1])
            return None if rx == ry else ((cx, cy), rx, ry)
        for (cx, cy), rx, ry in draw(a.calib, mk_chi):
            add("chi", "calib", rx, ry, {rep: [(cx, cy)] for rep in reps}, f"{rx}-{ry}")
        ok = sorted(sasa_ok)

        def mk_sasa():
            i, j = sorted(rng.choice(ok, 2, replace=False).tolist())
            return None if j - i < 3 else ((i, j), i, j)
        for (i, j), _, _ in draw(a.calib, mk_sasa):
            add("sasa", "calib", i, j, sasa_sp(i, j), f"{i}-{j}")
        common = sorted(set.intersection(*[hb_used[rep] for rep in reps]))

        def mk_hb():
            x, y = sorted(rng.choice(len(common), 2, replace=False).tolist())
            return ((common[x], common[y]), 0, 0)
        for (b1, b2), _, _ in (draw(a.calib, mk_hb) if len(common) > 1 else []):
            add("hb", "calib", -1, -1, {rep: [(hb_col[rep][b1], hb_col[rep][b2])] for rep in reps},
                f"{b1[0]}-{b1[1]} | {b2[0]}-{b2[1]}")

    names = {("chi", rep): [f"{r}:chi{c}" for r, c in zip(r0["chi_resid"], r0["chi_idx"])] for rep in reps}
    names.update({("sasa", rep): [f"{r}" for r in r0["sasa_resid"]] for rep in reps})
    names.update({("hb", rep): [f"{i}-{j}" for i, j in zip(R[rep]["hb_i"], R[rep]["hb_j"])] for rep in reps})
    return units, reg, names


# ============================================================================ computing one data set
def compute_obs_rep(obs, rep, raw, pairs, P, G, ctrl_set=None, detail=True, batch=128):
    """WLCC for the signal pairs (n, 2) of one observable and replica (real data)"""
    X0 = raw[obs]
    T = X0.shape[0]
    cols = np.unique(pairs)
    if ctrl_set is None:
        sub = X0[:, cols]
    else:
        offs = ctrl_offsets(P["ctrl_seed"], ctrl_set, rep, obs, T, X0.shape[1],
                            int(round(P["min_shift_ns"] / raw["dt"])))
        sub = np.stack([np.roll(X0[:, c], offs[c]) for c in cols], axis=1)
    if obs == "chi":
        Z = prepare(obs, sub, P, raw["dt"], raw["chi_resname"][cols], raw["chi_idx"][cols])
    else:
        Z = prepare(obs, sub, P, raw["dt"])
    pos = {c: k for k, c in enumerate(cols)}
    ia = np.array([pos[c] for c in pairs[:, 0]])
    ib = np.array([pos[c] for c in pairs[:, 1]])
    n, nc = len(pairs), len(G["starts"])
    S = np.empty(n, np.float32)
    if detail:
        Q = np.empty((n, nc), np.float32)
        LAG = np.empty((n, nc), np.float32)
        SG = np.empty((n, nc), np.int8)
    for b0 in range(0, n, batch):
        sl = slice(b0, min(n, b0 + batch))
        q, lag, sg = wlcc(Z[:, ia[sl], :], Z[:, ib[sl], :], G, gate=(obs == "hb"), pow_floor=P["pow_floor"])
        S[sl] = sustained(q, G["m"])
        if detail:
            Q[sl], LAG[sl], SG[sl] = q, lag, sg
    return (S, Q, LAG, SG) if detail else S


def control_obs_rep(obs, rep, raw, pairs, P, G, k, scheme, realQ=None, batch=128):
    """
    One control set for the signal pairs (n, 2) of one observable and replica. Returns per signal pair:
      S     sustained level in the control
      rel   circular distance between the offsets of the two signals (ns)       -> >= 1 us filter
      Smc   sustained level of the control with +-edge_ns around both seams set to 0   -> seam masking
      Smr   the same zones removed from the real series (original scheme only)
    """
    X0 = raw[obs]
    T, dt = X0.shape[0], raw["dt"]
    ms = int(round(P["min_shift_ns"] / dt))
    offs = ctrl_offsets(P["ctrl_seed"], k, rep, obs, T, X0.shape[1], ms, scheme,
                        raw["chi_resid"] if obs == "chi" else None)
    cols = np.unique(pairs)
    sub = np.stack([np.roll(X0[:, c], offs[c]) for c in cols], axis=1)
    if obs == "chi":
        Z = prepare(obs, sub, P, dt, raw["chi_resname"][cols], raw["chi_idx"][cols])
    else:
        Z = prepare(obs, sub, P, dt)
    pos = {c: i for i, c in enumerate(cols)}
    ia = np.array([pos[c] for c in pairs[:, 0]])
    ib = np.array([pos[c] for c in pairs[:, 1]])
    d = np.abs(offs[pairs[:, 0]] - offs[pairs[:, 1]]) % T
    out = dict(S=np.empty(len(pairs), np.float32), rel=(np.minimum(d, T - d) * dt).astype(np.float32))
    seam = offs * dt                                 # np.roll puts original frame 0 at index offset
    cen = G["centers_ns"]
    if scheme == "original":
        out["Smc"] = np.empty(len(pairs), np.float32)
        out["Smr"] = np.empty(len(pairs), np.float32)
    for b0 in range(0, len(pairs), batch):
        sl = slice(b0, min(len(pairs), b0 + batch))
        q, _, _ = wlcc(Z[:, ia[sl], :], Z[:, ib[sl], :], G, gate=(obs == "hb"), pow_floor=P["pow_floor"])
        out["S"][sl] = sustained(q, G["m"])
        if scheme == "original":
            zone = ((np.abs(cen[None, :] - seam[pairs[sl, 0]][:, None]) < P["edge_ns"])
                    | (np.abs(cen[None, :] - seam[pairs[sl, 1]][:, None]) < P["edge_ns"]))
            out["Smc"][sl] = sustained(np.where(zone, 0.0, q), G["m"])
            out["Smr"][sl] = sustained(np.where(zone, 0.0, realQ[sl]), G["m"])
    return out


def read_units(path, reps):
    return pd.read_csv(path, dtype={f"sp_{rep}": str for rep in reps})


def unit_ids(units, rep):
    return [np.array([int(x) for x in str(s).split(";") if x not in ("", "nan")], int)
            for s in units[f"sp_{rep}"].fillna("")]


# ---- worker state for control sets (each process loads its own copy)
_W = {}


def needed_ids(units, uids, reps, obs=OBS):
    """signal-pair ids per (observable, replica) used by the given units"""
    sel = units[units.uid.isin(uids)]
    need = {}
    for rep in reps:
        ids_list = unit_ids(sel, rep)
        for o in obs:
            m = (sel.obs == o).values
            need[(o, rep)] = np.array(sorted({x for mm, ids in zip(m, ids_list) if mm for x in ids}), int)
    return need


def _init_worker(cfg):
    _W["cfg"] = cfg
    _W["R"] = {rep: load_replica(cfg["indir"], rep) for rep in cfg["reps"]}
    reg = np.load(os.path.join(cfg["out"], "registry.npz"))
    _W["pairs"] = {(o, rep): reg[f"{o}|{rep}"] for o in OBS for rep in cfg["reps"]}
    _W["Q"] = {}


def _realQ(o, rep):
    if (o, rep) not in _W["Q"]:
        _W["Q"][(o, rep)] = np.load(os.path.join(_W["cfg"]["out"], f"real_{o}_{rep}_Q.npy"), mmap_mode="r")
    return _W["Q"][(o, rep)]


def _run_ctrl(job):
    k, scheme = job
    cfg = _W["cfg"]
    P, reps, R = cfg["P"], cfg["reps"], _W["R"]
    G = make_geometry(R[reps[0]]["T"], R[reps[0]]["dt"], P)
    t0 = time.time()
    res = {}
    for (o, rep), ids in cfg["need"][scheme].items():
        if not len(ids):
            continue
        rq = np.asarray(_realQ(o, rep)[ids]) if scheme == "original" else None
        out = control_obs_rep(o, rep, R[rep], _W["pairs"][(o, rep)][ids], P, G, k, scheme, rq, cfg["batch"])
        res[f"{o}|{rep}|ids"] = np.asarray(ids)
        for f, v in out.items():
            res[f"{o}|{rep}|{f}"] = v
    path = os.path.join(cfg["out"], "controls", scheme, f"s{k:04d}.npz")
    if os.path.exists(path):                        # keep signal pairs computed earlier for this set
        z = np.load(path)
        if str(z["phash"]) == cfg["phash"]:
            for key in [x[:-4] for x in z.files if x.endswith("|ids")]:
                fields = [x[len(key) + 1:] for x in z.files if x.startswith(key + "|") and not x.endswith("|ids")]
                if f"{key}|ids" not in res:
                    for f in ["ids"] + fields:
                        res[f"{key}|{f}"] = z[f"{key}|{f}"]
                    continue
                keep = ~np.isin(z[f"{key}|ids"], res[f"{key}|ids"])
                for f in ["ids"] + fields:
                    if f"{key}|{f}" in res:
                        res[f"{key}|{f}"] = np.concatenate([z[f"{key}|{f}"][keep], res[f"{key}|{f}"]])
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, set=k, scheme=scheme, phash=cfg["phash"], **res)
    os.replace(tmp, path)                           # an interrupted write never leaves a broken file
    return k, scheme, time.time() - t0


# ============================================================================ commands
DMS_NAME = "kras_coupling_edges_full_canonical.csv"


def newest(pattern):
    c = glob.glob(pattern)
    return max(c, key=os.path.getmtime) if c else None


def discover(a):
    """fill unset inputs from the standard layout of the analysis folder (the one holding wco_hmr/)"""
    d = a.indir
    for rep in a.reps:
        for kind in ("chi", "sasa", "hbond"):
            if not os.path.exists(os.path.join(d, f"{rep}_{kind}.npz")):
                sys.exit(f"missing {os.path.join(d, f'{rep}_{kind}.npz')}: run from the folder that holds {d}/ "
                         "or pass --indir")
    a.coh_chi = a.coh_chi or newest(os.path.join(d, "P80-120_thr0.75_ep500_ts2_*", "residue_pairs.csv"))
    a.coh_sasa = a.coh_sasa or newest(os.path.join(d, "SASA_P80-120_thr0.75_ep500_ts2_*sep3*", "edges.csv"))
    a.coh_hb = a.coh_hb or newest(os.path.join(d, "HB_P80-120_thr0.75_ep500_ts2_*pf0.05*", "hyperedges.csv"))
    if a.dms is None:
        a.dms = next((x for x in (DMS_NAME, os.path.join(d, DMS_NAME)) if os.path.exists(x)), None)
    a.pdb = a.pdb or os.path.join(d, "namd_3_2_wrapped.pdb")
    if a.dms and not os.path.exists(a.pdb):
        sys.exit(f"missing {a.pdb} (C-alpha distances for the DMS background); pass --pdb")
    for k, v in (("chi coherent pairs", a.coh_chi), ("SASA coherent pairs", a.coh_sasa),
                 ("H-bond coherent pairs", a.coh_hb), ("DMS couplings", a.dms), ("distances", a.pdb)):
        print(f"  {k:22s} {v if v else 'NOT FOUND (skipped)'}")


def parse_sets(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            x, y = map(int, part.split("-"))
            out += list(range(x, y + 1))
        elif part.strip():
            out.append(int(part))
    return out


def params_from(a):
    if a.max_lag_ns is None:
        a.max_lag_ns = 0.25 * (a.pmin + a.pmax) / 2 if a.mode == "band" else 50.0
    P = {k: getattr(a, k) for k in ["mode", "pmin", "pmax", "order", "window_ns", "step_ns", "max_lag_ns",
                                    "lag_step_ns", "min_ep_ns", "edge_ns", "pow_floor", "sasa_min_sd",
                                    "hb_min_trans", "ctrl_seed", "min_shift_ns", "theta_band"]}
    P["hb_occ"] = list(a.hb_occ)
    P["reps"] = list(a.reps)
    return P


def phash(P, units_path):
    h = hashlib.sha256(json.dumps({k: P[k] for k in COMPUTE_KEYS}, sort_keys=True).encode())
    h.update(open(units_path, "rb").read())
    return h.hexdigest()[:16]


def cmd_run(a):
    a.out = a.out or f"wlcc_{a.mode}"
    os.makedirs(os.path.join(a.out, "controls"), exist_ok=True)
    print("inputs:")
    discover(a)
    P = params_from(a)
    ppath = os.path.join(a.out, "params.json")
    if os.path.exists(ppath):
        old = json.load(open(ppath))
        diff = [k for k in COMPUTE_KEYS if old["P"].get(k) != P.get(k)]
        if diff and not a.force:
            sys.exit(f"{a.out} holds results computed with other settings ({diff}); use another --out or --force")
    t0 = time.time()
    R = {rep: load_replica(a.indir, rep) for rep in a.reps}
    T, dt = R[a.reps[0]]["T"], R[a.reps[0]]["dt"]
    G = make_geometry(T, dt, P)
    print(f"{T} frames x {dt:g} ns; {len(G['starts'])} windows of {G['w'] * dt:g} ns every {P['step_ns']:g} ns; "
          f"lags {G['lags'].min() * dt:g}..{G['lags'].max() * dt:g} ns; mode {P['mode']}", flush=True)

    upath = os.path.join(a.out, "units.csv")
    units_l, reg, names = build_units(a, R)
    units = pd.DataFrame(units_l)
    if os.path.exists(upath) and not a.force:
        old_u = pd.read_csv(upath)
        if len(old_u) != len(units) or not (old_u.label.astype(str).values == units.label.astype(str).values).all():
            sys.exit("units differ from the stored units.csv (inputs or unit settings changed); use --force")
    else:
        units.to_csv(upath, index=False)
    units = read_units(upath, a.reps)
    regarr = {f"{o}|{rep}": np.array(list(reg[(o, rep)].keys()), int).reshape(-1, 2) for (o, rep) in reg}
    np.savez_compressed(os.path.join(a.out, "registry.npz"), **regarr,
                        **{f"names|{o}|{rep}": np.array(v) for (o, rep), v in names.items()})
    ph = phash(P, upath)
    json.dump(dict(P=P, phash=ph, inputs=dict(indir=a.indir, coh_chi=a.coh_chi, coh_sasa=a.coh_sasa,
                                              coh_hb=a.coh_hb, dms=a.dms, pdb=a.pdb, n_bg=a.n_bg,
                                              bin_A=a.bin_A, calib=a.calib, unit_seed=a.unit_seed)),
              open(ppath, "w"), indent=1)
    print(units.groupby(["obs", "set"]).size().unstack(fill_value=0).to_string(), flush=True)
    print("signal pairs per replica: " + ", ".join(f"{o} {len(reg[(o, a.reps[0])])}" for o in OBS), flush=True)

    # ---- real data
    for rep in a.reps:
        for o in OBS:
            path = os.path.join(a.out, f"real_{o}_{rep}.npz")
            qpath = os.path.join(a.out, f"real_{o}_{rep}_Q.npy")     # uncompressed copy, memory-mapped by controls
            pairs = regarr[f"{o}|{rep}"]
            if os.path.exists(path) and str(np.load(path)["phash"]) == ph:
                if not os.path.exists(qpath):
                    np.save(qpath, np.load(path)["Q"])
                continue
            if len(pairs) == 0:
                continue
            t1 = time.time()
            S, Q, LAG, SG = compute_obs_rep(o, rep, R[rep], pairs, P, G, None, True, a.batch)
            np.save(qpath, Q)
            np.savez_compressed(path, S=S, Q=Q, LAG=LAG, SG=SG, pairs=pairs, centers_ns=G["centers_ns"], phash=ph)
            print(f"[real] {rep} {o}: {len(pairs)} signal pairs, {time.time() - t1:.0f} s", flush=True)

    # ---- controls (same schemes as the manuscript; see the header)
    sets = parse_sets(a.sets) if a.sets else []
    if not sets:
        print(f"done (no controls requested), {time.time() - t0:.0f} s")
        return
    if a.only_units == "calib":
        uids = units.uid[units.set == "calib"].tolist()
    elif a.only_units:
        uids = sorted({int(x) for x in open(a.only_units).read().split()})
    else:
        uids = units.uid.tolist()
    schemes = ["original"] + (["reswise"] if getattr(a, "reswise", False) else [])
    need = {"original": needed_ids(units, uids, a.reps),
            "reswise": needed_ids(units, uids, a.reps, obs=("chi",))}
    todo = []
    for sc in schemes:
        os.makedirs(os.path.join(a.out, "controls", sc), exist_ok=True)
        for k in sets:
            p = os.path.join(a.out, "controls", sc, f"s{k:04d}.npz")
            if os.path.exists(p):
                z = np.load(p)
                if str(z["phash"]) == ph and all(
                        not len(ids) or (f"{o}|{rep}|ids" in z.files and set(ids) <= set(z[f"{o}|{rep}|ids"].tolist()))
                        for (o, rep), ids in need[sc].items()):
                    continue
            todo.append((k, sc))
    print(f"controls: {len(todo)} runs to do ({', '.join(schemes)} scheme; sets {a.sets}; {len(uids)} units; "
          f"{a.jobs} processes)", flush=True)
    cfg = dict(indir=a.indir, reps=a.reps, out=a.out, P=P, need=need, batch=a.batch, phash=ph)
    if a.jobs <= 1:
        _init_worker(cfg)
        done = map(_run_ctrl, todo)
    else:
        ex = ProcessPoolExecutor(max_workers=a.jobs, initializer=_init_worker, initargs=(cfg,))
        done = ex.map(_run_ctrl, todo)
    for n_, (k_, sc_, sec) in enumerate(done, 1):
        print(f"[{n_}/{len(todo)}] {sc_} control {k_}: {sec:.0f} s", flush=True)
    if a.jobs > 1:
        ex.shutdown()
    print(f"done, {time.time() - t0:.0f} s")


def cmd_summarize(a):
    meta = json.load(open(os.path.join(a.out, "params.json")))
    P, ph = meta["P"], meta["phash"]
    reps = P["reps"]
    units = read_units(os.path.join(a.out, "units.csv"), reps)
    reg = np.load(os.path.join(a.out, "registry.npz"))
    real = {}
    for o in OBS:
        for rep in reps:
            p = os.path.join(a.out, f"real_{o}_{rep}.npz")
            if os.path.exists(p):
                z = np.load(p)
                if str(z["phash"]) != ph:
                    sys.exit(f"{p} was computed with other settings")
                real[(o, rep)] = {k: z[k] for k in ("S", "Q", "LAG", "SG")}
                centers = z["centers_ns"]
    step = P["step_ns"]
    m = int(np.ceil(P["min_ep_ns"] / step - 1e-9))
    ids = {rep: unit_ids(units, rep) for rep in reps}
    for r_i, rep in enumerate(reps):
        units[f"S_{rep}"] = [np.nanmax(real[(o, rep)]["S"][x]) if len(x) and (o, rep) in real else np.nan
                             for o, x in zip(units.obs, ids[rep])]

    # ---- controls: signal-pair results -> unit results per control set
    has_ids = np.array([[len(ids[rep][u]) > 0 for rep in reps] for u in range(len(units))], bool)
    uobs = units.obs.values
    nsig = {(o, rep): len(reg[f"{o}|{rep}"]) for o in OBS for rep in reps}

    def load_scheme(scheme):
        out = {}
        for f in sorted(glob.glob(os.path.join(a.out, "controls", scheme, "s*.npz"))):
            z = np.load(f)
            if str(z["phash"]) != ph:
                print(f"skipping {f}: other settings")
                continue
            arr = {fld: np.full((len(units), len(reps)), np.nan) for fld in ("S", "Soff", "Smc", "Smr")}
            rel_pass = {}
            for r_i, rep in enumerate(reps):
                for o in OBS:
                    key = f"{o}|{rep}"
                    if f"{key}|ids" not in z.files:
                        continue
                    ii = z[f"{key}|ids"]
                    full = {fld: np.full(nsig[(o, rep)], np.nan) for fld in arr}
                    full["S"][ii] = z[f"{key}|S"]
                    full["Soff"][ii] = np.where(z[f"{key}|rel"] >= a.rel_min_ns, z[f"{key}|S"], -1.0)
                    rel_pass.setdefault(o, []).append(float((z[f"{key}|rel"] >= a.rel_min_ns).mean()))
                    if f"{key}|Smc" in z.files:
                        full["Smc"][ii] = z[f"{key}|Smc"]
                        full["Smr"][ii] = z[f"{key}|Smr"]
                    for u_i in np.where(uobs == o)[0]:
                        x = ids[rep][u_i]
                        if len(x) == 0 or np.isnan(full["S"][x]).any():
                            continue
                        for fld in arr:
                            v = full[fld][x]
                            if not np.isnan(v).any():
                                arr[fld][u_i, r_i] = v.max()
            covered = (np.isfinite(arr["S"]) | ~has_ids).all(axis=1) & has_ids.any(axis=1)
            out[int(z["set"])] = dict(arr=arr, covered=covered,
                                      rel_pass={o: float(np.mean(v)) for o, v in rel_pass.items()})
        return out

    ctrl = {sc: load_scheme(sc) for sc in ("original", "reswise")}
    n_sets_total = len(ctrl["original"])

    # ---- threshold per observable
    theta, rows = {}, []
    for o in OBS:
        derived = P["theta_band"] if P["mode"] == "band" else np.nan
        # only control sets that contain the whole calibration set (the band/broad sets): later
        # confirm sets cover a few calibration pairs only by coincidence, and those would bias theta
        cal = ((units.set == "calib") & (units.obs == o)).values
        full = [c for c in ctrl["original"].values() if cal.any() and c["covered"][cal].all()]
        vals = np.concatenate([c["arr"]["S"][cal].ravel() for c in full]) if full else np.zeros(0)
        vals = vals[np.isfinite(vals)]
        rate = a.target_rate.get(o, TARGET_RATE[o])
        cal = float(np.quantile(vals, 1 - rate)) if len(vals) and len(vals) * rate >= 1 else np.nan
        if a.theta is not None:
            th, src = a.theta, "user"
        elif (a.theta_source or "calibrated") == "derived":
            th, src = derived, "derived"
        else:
            th, src = cal, "calibrated"
        if not np.isfinite(th):
            print(f"WARNING {o}: no threshold, so no {o} unit is counted. Calibration has {len(vals)} control "
                  f"values and a rate of {rate} needs >= {int(np.ceil(1 / rate))} (more --calib units or control "
                  f"sets), or pass --theta.")
            th, src = np.inf, "unavailable"
        theta[o] = th
        rows.append(dict(observable=o, theta=th, source=src, theta_derived=derived, theta_calibrated=cal,
                         target_rate=rate, calib_sets=len(full), calib_values=len(vals),
                         calib_rate_at_theta=float((vals >= th).mean()) if len(vals) else np.nan,
                         peak_abs_r_at_theta=float(np.sqrt(th))))
    thr = pd.DataFrame(rows)
    thr.to_csv(os.path.join(a.out, "thresholds.csv"), index=False)

    # ---- per unit: replicas, episodes, lags
    th_u = units.obs.map(theta).values
    Sm = units[[f"S_{r}" for r in reps]].values
    units["theta"] = th_u
    units["testable_reps"] = np.isfinite(Sm).sum(axis=1)
    units["reps"] = (np.nan_to_num(Sm, nan=-1) >= th_u[:, None]).sum(axis=1)
    for r_i, rep in enumerate(reps):
        cov, lag_med, lag_sd, pos, best, eps = [], [], [], [], [], []
        for u_i, (o, x) in enumerate(zip(units.obs, ids[rep])):
            if not len(x) or not np.isfinite(Sm[u_i, r_i]) or Sm[u_i, r_i] < th_u[u_i] or (o, rep) not in real:
                cov.append(0.0); lag_med.append(np.nan); lag_sd.append(np.nan); pos.append(np.nan)
                best.append(""); eps.append("")
                continue
            d = real[(o, rep)]
            mask = np.zeros(len(centers), bool)
            for sp in x:
                for s, e in runs(d["Q"][sp] >= th_u[u_i], m):
                    mask[s:e] = True
            cov.append(float(mask.sum() * step))
            b = x[np.nanargmax(d["S"][x])]
            ep = runs(d["Q"][b] >= th_u[u_i], m)
            pts = np.concatenate([np.arange(s, e) for s, e in ep])
            lag_med.append(float(np.median(d["LAG"][b][pts])))
            lag_sd.append(float(np.std(d["LAG"][b][pts])))
            pos.append(float((d["SG"][b][pts] > 0).mean()) if o != "chi" else np.nan)
            pa, pb = reg[f"{o}|{rep}"][b]
            nm = reg[f"names|{o}|{rep}"]
            best.append(f"{nm[pa]} / {nm[pb]}")
            eps.append(";".join(f"{centers[s]:g}-{centers[e - 1] + step:g}" for s, e in ep))
        units[f"cov_ns_{rep}"] = cov
        units[f"lag_ns_{rep}"] = lag_med
        units[f"lag_sd_ns_{rep}"] = lag_sd
        units[f"frac_pos_r_{rep}"] = pos
        units[f"best_pair_{rep}"] = best
        units[f"episodes_{rep}"] = eps
    lagcols = units[[f"lag_ns_{r}" for r in reps]].values
    sg = np.sign(np.where(np.abs(lagcols) < a.lag_zero_ns, 0, lagcols))
    units["lag_sign_consistent"] = [bool(len(set(v[np.isfinite(v)])) <= 1) if np.isfinite(v).sum() >= 2 else np.nan
                                    for v in sg]

    # ---- controls per unit, as in the manuscript (Section 5.4)
    # original  k = control sets in which the unit reached at least its real number of replicas;
    #           p = (1 + k) / (1 + sets); a pair in >= 2 replicas is set aside when k >= 1
    # >=1 us    the same, counting only signal pairs whose two offsets are >= rel_min_ns apart
    # seam      the same zones (+-edge_ns around each seam) removed from control and real series;
    #           counted when the control reaches max(real masked replicas, 2)
    # residue   chi only: all chi angles of a residue share one offset (separate controls)
    real_reps = units.reps.values
    thc = th_u[:, None]

    def reps_of(x):
        return (np.nan_to_num(x, nan=-1.0) >= thc).sum(axis=1)

    def tally(scheme, rule):
        k, n = np.zeros(len(units)), np.zeros(len(units))
        for c in ctrl[scheme].values():
            hit = rule(c["arr"])
            k += c["covered"] & hit
            n += c["covered"]
        return k, n

    k0, n0 = tally("original", lambda A: reps_of(A["S"]) >= real_reps)
    k1, n1 = tally("original", lambda A: reps_of(A["Soff"]) >= real_reps)
    k2, n2 = tally("original", lambda A: reps_of(A["Smc"]) >= np.maximum(reps_of(A["Smr"]), 2))
    k3, n3 = tally("reswise", lambda A: reps_of(A["S"]) >= real_reps)
    pos = real_reps >= 1
    units["ctrl_sets"] = n0.astype(int)
    units["ctrl_k"] = np.where(pos & (n0 > 0), k0, np.nan)
    units["ctrl_p"] = np.where(pos & (n0 > 0), (1 + k0) / (1 + n0), np.nan)
    units["set_aside"] = np.where((real_reps >= 2) & (n0 > 0), k0 >= 1, np.nan)
    units["k_rel_1us"] = np.where(pos & (n1 > 0), k1, np.nan)
    units["k_seam_masked"] = np.where((real_reps >= 2) & (n2 > 0), k2, np.nan)
    units["k_per_residue"] = np.where(pos & (n3 > 0), k3, np.nan)
    units["per_residue_sets"] = n3.astype(int)

    # ---- controls for the screened set (DMS units), as the network counts of Section 2.1
    rows_c = []
    dms = (units.set == "dms").values
    for o in OBS:
        msk = dms & (units.obs == o).values

        def counts(r, cov):
            r = r[msk & cov]
            return int((r >= 1).sum()), int((r >= 2).sum()), int((r >= 3).sum())
        for scheme, label in (("original", "original"), ("original", "rel >= 1 us"),
                              ("original", "seam masked"), ("reswise", "per residue")):
            if scheme == "reswise" and o != "chi":
                continue
            # only sets in which every DMS unit of this observable was run (confirm), as for the
            # network counts of the manuscript; earlier sets cover a few units only by coincidence
            sets_ = {k_: c for k_, c in ctrl[scheme].items() if msk.any() and c["covered"][msk].all()}
            if not sets_:
                continue
            real_c, ctrl_c = [], []
            for c in sets_.values():
                A, cov = c["arr"], c["covered"]
                if label == "seam masked":
                    real_c.append(counts(reps_of(A["Smr"]), cov))
                    ctrl_c.append(counts(reps_of(A["Smc"]), cov))
                elif label == "rel >= 1 us":
                    f = c["rel_pass"].get(o, 1.0) or 1.0
                    real_c.append(counts(real_reps, cov))
                    ctrl_c.append(tuple(v / f for v in counts(reps_of(A["Soff"]), cov)))
                else:
                    real_c.append(counts(real_reps, cov))
                    ctrl_c.append(counts(reps_of(A["S"]), cov))
            R_, C_ = np.array(real_c, float), np.array(ctrl_c, float)
            rows_c.append(dict(observable=o, controls=label, sets=len(C_), units=int(msk.sum()),
                               real_ge1=f"{R_[:, 0].min():g}-{R_[:, 0].max():g}",
                               real_ge2=f"{R_[:, 1].min():g}-{R_[:, 1].max():g}",
                               real_3=f"{R_[:, 2].min():g}-{R_[:, 2].max():g}",
                               ctrl_ge1_mean=C_[:, 0].mean(), ctrl_ge1_range=f"{C_[:, 0].min():.3g}-{C_[:, 0].max():.3g}",
                               ctrl_ge2_mean=C_[:, 1].mean(), ctrl_ge2_range=f"{C_[:, 1].min():.3g}-{C_[:, 1].max():.3g}",
                               ctrl_3_max=C_[:, 2].max(),
                               sets_ge2_reaching_real=int((C_[:, 1] >= R_[:, 1]).sum())))
    ctrl_tab = pd.DataFrame(rows_c)
    if len(ctrl_tab):
        ctrl_tab.to_csv(os.path.join(a.out, "controls_summary.csv"), index=False, float_format="%.3g")
    units.to_csv(os.path.join(a.out, "units_results.csv"), index=False)
    sel = units[(units.set != "calib") & (units.reps >= a.select_min_reps)].uid
    open(os.path.join(a.out, "units_reproducible.txt"), "w").write("\n".join(map(str, sel)) + "\n")

    # ---- coherent pairs: coherence replicas against WLCC replicas
    coh = units[units.set == "coherent"]
    ct = []
    for o in OBS:
        c = coh[coh.obs == o]
        if len(c):
            tab = pd.crosstab(c.coh_reps, c.reps).reindex(columns=range(len(reps) + 1), fill_value=0)
            tab.insert(0, "observable", o)
            ct.append(tab.reset_index())
    if ct:
        pd.concat(ct).rename(columns={"coh_reps": "coherence_replicas"}).to_csv(
            os.path.join(a.out, "coherent_crosstab.csv"), index=False)

    # ---- DMS units against distance-matched background
    rng = np.random.default_rng(a.seed)
    rows = []
    for o in OBS if "bg_of" in units else ():
        D = units[(units.set == "dms") & (units.obs == o) & (units.testable_reps >= 2)]
        Bs, keep = [], []
        for u in D.itertuples():
            B = units[(units.set == "background") & (units.bg_of == u.uid) & (units.testable_reps >= 2)]
            if len(B):
                Bs.append((B.reps.values >= 2).astype(int))
                keep.append(u.uid)
        if not keep:
            continue
        Dk = D[D.uid.isin(keep)]
        obs_n = int((Dk.reps >= 2).sum())
        sims = np.zeros(a.n_resample, int)
        for b in Bs:
            sims += b[rng.integers(0, len(b), a.n_resample)]
        allB = np.concatenate(Bs)
        rows.append(dict(observable=o, dms_units=len(Dk), dms_reproducible=obs_n,
                         dms_reproducible_not_set_aside=int(((Dk.reps >= 2) & (Dk.set_aside == 0)).sum()),
                         background_units=len(allB), background_fraction=float(allB.mean()),
                         matched_mean=float(sims.mean()), matched_2p5=float(np.percentile(sims, 2.5)),
                         matched_97p5=float(np.percentile(sims, 97.5)),
                         frac_matched_ge_observed=float((sims >= obs_n).mean()),
                         p=(1 + int((sims >= obs_n).sum())) / (1 + a.n_resample)))
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "dms_vs_background.csv"), index=False)

    # ---- short report
    L = [f"mode {P['mode']}; window {P['window_ns']} ns, step {step} ns, lags +-{P['max_lag_ns']} ns, "
         f"episode >= {P['min_ep_ns']} ns; control sets read: {n_sets_total}", "", thr.to_string(index=False), ""]
    for o in OBS:
        c = coh[coh.obs == o]
        if len(c):
            r2 = c[c.coh_reps >= 2]
            L.append(f"{o}: coherent pairs {len(c)}; WLCC in >= 2 replicas: {int((c.reps >= 2).sum())}; "
                     f"of the {len(r2)} reproducible coherent pairs, {int((r2.reps >= 2).sum())} in >= 2 WLCC replicas")
    if rows:
        L += ["", pd.DataFrame(rows).to_string(index=False)]
    if len(ctrl_tab):
        L += ["", "controls for the DMS units (screened set), as in the manuscript:", ctrl_tab.to_string(index=False)]
        new = units[(units.set == "dms") & (units.reps >= 2)]
        if len(new):
            L += ["", "DMS units in >= 2 replicas, with the per-pair control checks:",
                  new[["uid", "obs", "label", "trait", "reps", "ctrl_sets", "ctrl_k", "ctrl_p", "set_aside",
                       "k_rel_1us", "k_seam_masked", "k_per_residue"]].to_string(index=False)]
    open(os.path.join(a.out, "report.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


# ============================================================================ self-tests
def _direct(x, y, s, w, tau, detrend):
    """reference: Pearson r of one (lagged) window pair by explicit regression"""
    xs, ys = x[s:s + w], y[s + tau:s + tau + w]
    tt = np.arange(w, dtype=float)
    A = np.vstack([np.ones(w), tt]).T if detrend else np.ones((w, 1))
    rx = xs - A @ np.linalg.lstsq(A, xs, rcond=None)[0]
    ry = ys - A @ np.linalg.lstsq(A, ys, rcond=None)[0]
    return rx, ry


def cmd_selftest(a):
    ok = True
    rng = np.random.default_rng(0)
    # 1. window sums against explicit regression, scalar and chi (RV), with and without detrending
    T = 3000
    tt = np.arange(T)
    x = rng.normal(size=T) + 0.003 * tt
    y = 0.6 * np.roll(x, 7) + rng.normal(size=T) - 0.002 * tt
    for mode in ("band", "broad"):
        P = dict(window_ns=200, step_ns=10, max_lag_ns=15, lag_step_ns=1, min_ep_ns=100, edge_ns=300, mode=mode)
        G = make_geometry(T, 1.0, P)
        q, lag, sg = wlcc(x[:, None, None], y[:, None, None], G)
        worst = 0.0
        for wi in range(0, len(G["starts"]), 37):
            best = max(((np.corrcoef(*_direct(x, y, G["starts"][wi], G["w"], t_, mode == "broad"))[0, 1]) ** 2, -abs(t_), t_)
                       for t_ in G["lags"])
            worst = max(worst, abs(best[0] - q[0, wi]))
            if best[2] != lag[0, wi]:
                worst = max(worst, 1.0)
        print(f"[1] scalar, {mode}: max |q - direct| = {worst:.2e}")
        ok &= worst < 1e-6
        a2 = rng.uniform(-180, 180, size=(T, 2))
        Xc = chi_signals(a2, ["LEU", "LEU"], [1, 1])
        Xc[:, 1] = 0.7 * np.roll(Xc[:, 0], -4, axis=0) + 0.5 * Xc[:, 1]
        Xc = Xc - Xc.mean(axis=0)
        q2, lag2, _ = wlcc(Xc[:, :1], Xc[:, 1:], G)
        worst = 0.0
        for wi in range(0, len(G["starts"]), 41):
            vals = []
            for t_ in G["lags"]:
                rx0, ry0 = _direct(Xc[:, 0, 0], Xc[:, 1, 0], G["starts"][wi], G["w"], t_, mode == "broad")
                rx1, ry1 = _direct(Xc[:, 0, 1], Xc[:, 1, 1], G["starts"][wi], G["w"], t_, mode == "broad")
                RX, RY = np.vstack([rx0, rx1]), np.vstack([ry0, ry1])
                Cxy, Cxx, Cyy = RX @ RY.T, RX @ RX.T, RY @ RY.T
                vals.append(((Cxy ** 2).sum() / np.sqrt((Cxx ** 2).sum() * (Cyy ** 2).sum()), -abs(t_), t_))
            best = max(vals)
            worst = max(worst, abs(best[0] - q2[0, wi]))
        print(f"[1] chi RV, {mode}: max |q - direct| = {worst:.2e}")
        ok &= worst < 1e-6

    # 2. synthetic signals as in Fig. 5: shared 100 ns oscillation in three stretches, signal 2 trails by 3 ns
    T = 10000
    t = np.arange(T, dtype=float)
    stretches = [(2400, 4300, 1.0), (6000, 7100, 1.0), (8000, 8200, 2.0)]
    o1, o2 = np.zeros(T), np.zeros(T)
    for s, e, A in stretches:
        o1[s:e] += A * np.sin(2 * np.pi * t[s:e] / 100)
        o2[s:e] += A * np.sin(2 * np.pi * (t[s:e] - 3) / 100)
    drift = lambda: np.convolve(rng.normal(size=T), np.ones(500) / 500, "same") * 20
    s1 = o1 + rng.normal(size=T) + drift()
    s2 = o2 + rng.normal(size=T) + drift()
    P = dict(mode="band", pmin=80, pmax=120, order=4, window_ns=500, step_ns=10, max_lag_ns=25, lag_step_ns=1,
             min_ep_ns=500, edge_ns=722)
    G = make_geometry(T, 1.0, P)
    Z = prepare("sasa", np.stack([s1, s2], 1), P, 1.0)
    q, lag, _ = wlcc(Z[:, :1], Z[:, 1:], G)
    ep = runs(q[0] >= 0.75, G["m"])
    spans = [(G["centers_ns"][s], G["centers_ns"][e - 1] + 10) for s, e in ep]
    pts = np.concatenate([np.arange(s, e) for s, e in ep]) if ep else np.array([], int)
    hit = [any(sp[0] < en and sp[1] > st for sp in spans) for st, en, _ in stretches]
    med = float(np.median(lag[0][pts])) if len(pts) else np.nan
    print(f"[2] synthetic: episodes {[(int(s), int(e)) for s, e in spans]} ns; stretches hit {hit}; "
          f"median lag {med:+.0f} ns (imposed +3)")
    ok &= hit[0] and hit[1] and abs(med - 3) <= 3
    if hit[2]:
        print("    note: the 200 ns burst of double amplitude also makes an episode; a 500 ns boxcar window plus "
              "band-pass ringing spreads a short strong burst over > 500 ns of window centres")
    Zn = prepare("sasa", np.stack([o1, o2], 1), P, 1.0)        # same signals without noise and drift
    qn, lagn, _ = wlcc(Zn[:, :1], Zn[:, 1:], G)
    inside = (G["centers_ns"] > 2700) & (G["centers_ns"] < 4000)
    print(f"[2] noise-free: median lag {np.median(lagn[0][inside]):+.0f} ns, median q {np.median(qn[0][inside]):.4f}")
    ok &= np.median(lagn[0][inside]) == 3 and np.median(qn[0][inside]) > 0.999
    off = ctrl_offsets(7, 1, "namd_3", "sasa", T, 2, 1000)
    Zc = prepare("sasa", np.stack([s1, np.roll(s2, off[1])], 1), P, 1.0)
    qc, _, _ = wlcc(Zc[:, :1], Zc[:, 1:], G)
    print(f"[2] same pair, second signal shifted by {off[1]} ns: sustained level {sustained(qc, G['m'])[0]:.2f} "
          f"(real {sustained(q, G['m'])[0]:.2f})")

    # 3. optional: real example pairs exported by the pipeline (raw signals + band coherence)
    if a.examples:
        z = np.load(a.examples, allow_pickle=True)
        keys = sorted({k.rsplit("|", 1)[0] for k in z.files if k.endswith("|raw")})
        out = []
        for key in keys:
            parts = key.split("|")
            pair, obs, rep, lab = parts[0], parts[1], parts[-1], "|".join(parts[2:-1])
            raw = z[key + "|raw"].T
            Rb, valid, td = z[key + "|Rb"], z[key + "|valid"].astype(bool), z[key + "|t_dec"]
            if obs == "chi":
                sym = z[f"{pair}|chi|{lab}|symmetric"].astype(bool)
                th = np.radians(raw.astype(float))
                th[:, sym] *= 2
                Zx = np.stack([np.cos(th), np.sin(th)], 2)
                Zx = bandpass(Zx, 1.0, 80, 120, 4)
            else:
                Zx = bandpass(raw.astype(float)[:, :, None], 1.0, 80, 120, 4)
            Zx = Zx - Zx.mean(0)
            G = make_geometry(raw.shape[0], 1.0, P, centers_ns=td[valid])
            q, lag, _ = wlcc(Zx[:, :1], Zx[:, 1:], G, gate=(obs == "hb"))
            Rv = Rb[valid]
            eq = np.zeros(len(Rv), bool)
            er = np.zeros(len(Rv), bool)
            for s, e in runs(q[0] >= 0.75, G["m"]):
                eq[s:e] = True
            for s, e in runs(Rv >= 0.75, G["m"]):
                er[s:e] = True
            out.append(dict(pair=pair, obs=obs, signals=lab, rep=rep, r_q_coherence=np.corrcoef(q[0], Rv)[0, 1],
                            S_wlcc=sustained(q, G["m"])[0], S_coherence=sustained(Rv[None], G["m"])[0],
                            ep_ns_wlcc=eq.sum() * 10, ep_ns_coherence=er.sum() * 10, overlap_ns=(eq & er).sum() * 10))
        df = pd.DataFrame(out)
        pd.set_option("display.width", 200)
        print("[3] example pairs, band mode vs band coherence (r between the two time series; S = sustained level):")
        print(df.round(2).to_string(index=False))
        df.to_csv("wlcc_selftest_examples.csv", index=False)
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


# ============================================================================ main
def summarize_defaults(out):
    return argparse.Namespace(out=out, theta=None, theta_source=None, target_rate={}, select_min_reps=2,
                              lag_zero_ns=5.0, n_resample=10000, seed=1, rel_min_ns=1000.0)


def cmd_confirm(a):
    """
    The manuscript's control analyses for the DMS units, run only when WLCC finds a DMS unit in
    >= 2 replicas: 200 original controls for every DMS unit (the screened set, so that counts of
    recurrent units can be compared as in Section 2.1), from which the >= 1 us and seam-masked
    checks are derived, and 200 per-residue chi controls. Then a new summary.
    """
    a.out = a.out or f"wlcc_{a.mode}"
    res = os.path.join(a.out, "units_results.csv")
    if not os.path.exists(res):
        sys.exit(f"{res} not found: run 'python kras_wlcc.py {a.mode}' first")
    d = pd.read_csv(res)
    dms = d[d.set == "dms"]
    new = dms[dms.reps >= 2]
    if new.empty:
        print("no DMS unit is found in >= 2 replicas: no controls needed")
        return
    print(f"{len(new)} DMS units in >= 2 replicas:")
    print(new[["uid", "obs", "label", "trait", "reps"]].to_string(index=False))
    path = os.path.join(a.out, "units_dms.txt")
    open(path, "w").write("\n".join(map(str, dms.uid)) + "\n")
    print(f"controls for all {len(dms)} DMS units (the screened set): original and, for chi, per residue")
    a.only_units, a.reswise = path, True
    cmd_run(a)
    cmd_summarize(summarize_defaults(a.out))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # options shared by every command that computes (defaults = the standard analysis folder)
    c = argparse.ArgumentParser(add_help=False)
    c.add_argument("--indir", default="wco_hmr", help="folder with {rep}_chi/sasa/hbond.npz")
    c.add_argument("--reps", nargs="+", default=["namd_3", "namd_3_2", "namd_3_3"])
    c.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="parallel processes")
    c.add_argument("--out", help="default wlcc_<mode>")
    c.add_argument("--coh-chi", help="default: newest <indir>/P80-120_thr0.75_ep500_ts2_*/residue_pairs.csv")
    c.add_argument("--coh-sasa", help="default: newest <indir>/SASA_P80-120_thr0.75_ep500_ts2_*sep3*/edges.csv")
    c.add_argument("--coh-hb", help="default: newest <indir>/HB_P80-120_thr0.75_ep500_ts2_*pf0.05*/hyperedges.csv")
    c.add_argument("--dms", help=f"default: ./{DMS_NAME} or <indir>/{DMS_NAME}")
    c.add_argument("--pdb", help="default <indir>/namd_3_2_wrapped.pdb (as in the DMS null)")
    c.add_argument("--n-bg", type=int, default=5, help="background pairs per DMS unit")
    c.add_argument("--bin-A", type=float, default=2.0)
    c.add_argument("--calib", type=int, default=2000, help="random signal pairs per observable for calibration")
    c.add_argument("--unit-seed", type=int, default=1)
    c.add_argument("--pmin", type=float, default=80.0)
    c.add_argument("--pmax", type=float, default=120.0)
    c.add_argument("--order", type=int, default=4)
    c.add_argument("--window-ns", type=float, default=500.0)
    c.add_argument("--step-ns", type=float, default=10.0)
    c.add_argument("--max-lag-ns", type=float, default=None,
                   help="default band: 25 ns = P/4 at the band centre (with |r|, every phase maps to one sign and "
                        "one lag in +-P/4); broad: 50 ns")
    c.add_argument("--lag-step-ns", type=float, default=1.0)
    c.add_argument("--min-ep-ns", type=float, default=500.0)
    c.add_argument("--edge-ns", type=float, default=722.0)
    c.add_argument("--theta-band", type=float, default=0.75)
    c.add_argument("--pow-floor", type=float, default=0.05)
    c.add_argument("--sasa-min-sd", type=float, default=1.0)
    c.add_argument("--hb-occ", type=float, nargs=2, default=[0.05, 0.95])
    c.add_argument("--hb-min-trans", type=int, default=10)
    c.add_argument("--ctrl-seed", type=int, default=7)
    c.add_argument("--min-shift-ns", type=float, default=1000.0)
    c.add_argument("--batch", type=int, default=128)
    c.add_argument("--force", action="store_true", help="overwrite results computed with other settings")

    t = sub.add_parser("test", help="self-test; uses <indir>/pairsignals/signals.npz if present")
    t.add_argument("--indir", default="wco_hmr")
    t.add_argument("--examples")
    for mode in ("band", "broad"):
        b = sub.add_parser(mode, parents=[c], help=f"{mode} mode: real data, 20 control sets for the "
                                                   "calibration pairs, summary")
        b.add_argument("--sets", default="1-20")
    cf = sub.add_parser("confirm", parents=[c], help="the manuscript's control analyses (200 sets) for the DMS "
                                                     "units, if any is found in >= 2 replicas; then summary")
    cf.add_argument("mode", choices=["band", "broad"])
    cf.add_argument("--sets", default="1-200")

    # advanced: the individual steps
    r = sub.add_parser("run", parents=[c], help="advanced: compute only")
    r.add_argument("--mode", choices=["band", "broad"], default="band")
    r.add_argument("--sets", default="1-20", help="control sets, e.g. 1-20; empty = none")
    r.add_argument("--only-units", help="'calib' or a file of unit ids; controls are run for these units only")
    r.add_argument("--reswise", action="store_true", help="also the per-residue chi controls")
    s = sub.add_parser("summarize", help="advanced: summary only")
    s.add_argument("--out", required=True)
    s.add_argument("--theta", type=float, help="one threshold for all observables (overrides)")
    s.add_argument("--theta-source", choices=["derived", "calibrated"])
    s.add_argument("--target-rate", nargs="*", default=[], help="e.g. chi=0.00091 sasa=0.00027 hb=0.00059")
    s.add_argument("--select-min-reps", type=int, default=2)
    s.add_argument("--lag-zero-ns", type=float, default=5.0,
                   help="|lag| below this counts as zero when comparing lag signs (lag SD ~5 ns at Rbar 0.95)")
    s.add_argument("--n-resample", type=int, default=10000)
    s.add_argument("--rel-min-ns", type=float, default=1000.0, help="relative-offset filter (manuscript: 1 us)")
    s.add_argument("--seed", type=int, default=1)

    a = p.parse_args()
    if a.cmd == "test":
        if a.examples is None:
            x = os.path.join(a.indir, "pairsignals", "signals.npz")
            a.examples = x if os.path.exists(x) else None
            print(f"examples: {a.examples or 'not found (only synthetic tests)'}")
        sys.exit(0 if cmd_selftest(a) else 1)
    if a.cmd in ("band", "broad"):
        a.mode, a.only_units = a.cmd, "calib"
        cmd_run(a)
        cmd_summarize(summarize_defaults(a.out))
    elif a.cmd == "confirm":
        cmd_confirm(a)
    elif a.cmd == "run":
        cmd_run(a)
    else:
        a.target_rate = {k: float(v) for k, v in (x.split("=") for x in a.target_rate)}
        cmd_summarize(a)


if __name__ == "__main__":
    main()
