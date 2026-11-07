#!/usr/bin/env python
"""
kras_hmr_pairsignals.py -- every observable for selected residue pairs (SI figures).

For each residue pair (default Ser17-Tyr32, Asp69-Arg102, Gln99-Arg102) and each
replica, whether or not the pair is an association, the script recomputes with the
unchanged pipeline functions (80-120 ns, 0.75, 500 ns, smoothing 2 x scale):
  chi     every inter-residue chi-angle pair: band coherence R(t), episodes
  SASA    the two residue SASA series: R(t), episodes (weak residues flagged)
  H-bond  every pair of filtered H-bond signals with one signal involving residue i
          and the other residue j (or one signal joining i and j): R(t), episodes
together with the raw signals, whole-window Pearson r and r within each episode.

Output: wco_hmr/pairsignals/
  summary.csv   pair x observable x sub-pair x replica: mean_R, coverage, episodes,
                whole-window r (r2-type for chi: zero-lag RV), r within episodes
  signals.npz   raw signals, R(t), valid mask and episodes for every row of summary.csv

  python kras_hmr_pairsignals.py
  python kras_hmr_pairsignals.py --pairs 17-32 69-102 99-102 73-103
  # K88 figure: K88-E91 and K88-D92 in all observables, plus the K88-D92 backbone / side-chain split
  python kras_hmr_pairsignals.py --pairs 88-91 88-92 --k88-split --outdir wco_hmr/pairsignals_k88

--k88-split adds, per replica, the K88-D92 H-bond split into a backbone-backbone signal (D92bb) and a
signal with a side-chain atom (D92sc), as in kras_k88_split.py (needs <indir>/<rep>_wrapped.pdb/.dcd), and
stores E91|D92sc and E91|D92bb (tag 88-92|hbsplit|...) with the same functions and settings.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kras_hmr_wco import chi_to_signals, episodes, SYMMETRIC  # noqa: E402
from kras_hmr_averages import (band_setup, pair_band_chi, pair_band_scalar, rv0_matrix,  # noqa: E402
                               transitions, BASE, POW_FLOOR, WEAK_SD, OCC, MIN_TRANS)
import kras_hmr_runcheck as rc  # noqa: E402


def ep_stats(Rb, B, t0):
    ep = episodes((Rb >= BASE["thr"]) & B["valid"], B["dtd"], rc.eff_ep(BASE))
    cov = sum(l for *_, l in ep) / (B["valid"].sum() * B["dtd"])
    return ep, cov


def corr_in(x, y, ep, t0, fvalid):
    t = np.arange(len(x)) + t0
    r_all = float(np.corrcoef(x[fvalid], y[fvalid])[0, 1])
    r_ep = [float(np.corrcoef(x[(t >= s + t0) & (t < e + t0)], y[(t >= s + t0) & (t < e + t0)])[0, 1])
            for s, e, _ in ep]
    return r_all, r_ep


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="wco_hmr")
    p.add_argument("--reps", nargs="+", default=["namd_3", "namd_3_2", "namd_3_3"])
    p.add_argument("--pairs", nargs="+", default=["17-32", "69-102", "99-102"])
    p.add_argument("--outdir", default=None, help="default: <indir>/pairsignals")
    p.add_argument("--k88-split", action="store_true", help="add the K88-D92 backbone / side-chain split")
    a = p.parse_args()
    out = a.outdir or os.path.join(a.indir, "pairsignals"); os.makedirs(out, exist_ok=True)
    pairs = [tuple(sorted(map(int, s.split("-")))) for s in a.pairs]
    rows, blob, tdec = [], {}, BASE["tdec"]

    def store(tag, r, raw, Rb, B, t0, ep):
        blob[f"{tag}|{r}|raw"] = raw
        blob[f"{tag}|{r}|Rb"] = Rb.astype(np.float32)
        blob[f"{tag}|{r}|valid"] = B["valid"]
        blob[f"{tag}|{r}|t_dec"] = (t0 + np.arange(B["Td"]) * B["dtd"]).astype(np.float32)
        blob[f"{tag}|{r}|episodes"] = np.array([[s + t0, e + t0] for s, e, _ in ep]).reshape(-1, 2)

    for r in a.reps:
        # ---------------- chi
        z = np.load(os.path.join(a.indir, f"{r}_chi.npz"))
        ang, resid, resname, chi = z["angles"], z["resid"], z["resname"], z["chi"]
        T = ang.shape[0]; dt = float(np.diff(z["time_ns"])[0]); t0 = float(z["time_ns"][0])
        B = band_setup(T, dt, tdec); B["dt"] = dt
        sig = chi_to_signals(ang, resname, chi)
        for i, j in pairs:
            for ki in np.where(resid == i)[0]:
                for kj in np.where(resid == j)[0]:
                    Rb, _ = pair_band_chi(sig[ki], sig[kj], B, tdec)
                    ep, cov = ep_stats(Rb, B, t0)
                    rv = float(rv0_matrix(np.stack([sig[ki][:, B["fvalid"]], sig[kj][:, B["fvalid"]]]))[0, 1])
                    sub = f"chi{chi[ki]}-chi{chi[kj]}"
                    tag = f"{i}-{j}|chi|{sub}"
                    store(tag, r, ang[:, [ki, kj]].T.astype(np.float32), Rb, B, t0, ep)
                    blob[f"{tag}|symmetric"] = np.array([(str(resname[ki]), int(chi[ki])) in SYMMETRIC,
                                                         (str(resname[kj]), int(chi[kj])) in SYMMETRIC])
                    rows.append(dict(pair=f"{i}-{j}", observable="chi", sub=sub, replica=r,
                                     names=f"{resname[ki]}{i} chi{chi[ki]} / {resname[kj]}{j} chi{chi[kj]}",
                                     mean_R=float(Rb[B["valid"]].mean()), coverage=cov, n_episodes=len(ep),
                                     episodes=";".join(f"{s + t0:.0f}-{e + t0:.0f}" for s, e, _ in ep),
                                     whole_r2_type=rv, r_episodes=""))
        # ---------------- SASA
        z = np.load(os.path.join(a.indir, f"{r}_sasa.npz"))
        x, resid, resname = z["sasa"].astype(np.float64), z["resid"], z["resname"]
        T = x.shape[0]; dt = float(np.diff(z["time_ns"])[0]); t0 = float(z["time_ns"][0])
        B = band_setup(T, dt, tdec); B["dt"] = dt
        for i, j in pairs:
            ki, kj = np.where(resid == i)[0][0], np.where(resid == j)[0][0]
            Rb, _ = pair_band_scalar(x[:, ki], x[:, kj], B, tdec)
            ep, cov = ep_stats(Rb, B, t0)
            r_all, r_ep = corr_in(x[:, ki], x[:, kj], ep, t0, B["fvalid"])
            weak = [bool(x[:, k].std() < WEAK_SD) for k in (ki, kj)]
            tag = f"{i}-{j}|sasa|sasa"
            store(tag, r, x[:, [ki, kj]].T.astype(np.float32), Rb, B, t0, ep)
            rows.append(dict(pair=f"{i}-{j}", observable="sasa", sub="sasa", replica=r,
                             names=f"{resname[ki]}{i} / {resname[kj]}{j}" + (" (weak signal)" if any(weak) else ""),
                             mean_R=float(Rb[B["valid"]].mean()), coverage=cov, n_episodes=len(ep),
                             episodes=";".join(f"{s + t0:.0f}-{e + t0:.0f}" for s, e, _ in ep),
                             whole_r=r_all, r_episodes=";".join(f"{v:.3f}" for v in r_ep)))
        # ---------------- H-bond
        z = np.load(os.path.join(a.indir, f"{r}_hbond.npz"))
        M = z["hb"]; T = M.shape[0]; dt = float(np.diff(z["time_ns"])[0]); t0 = float(z["time_ns"][0])
        B = band_setup(T, dt, tdec); B["dt"] = dt
        occ = M.mean(axis=0); trn = np.array([transitions(M[:, k]) for k in range(M.shape[1])])
        use = (occ >= OCC[0]) & (occ <= OCC[1]) & (trn >= MIN_TRANS)
        lab = [tuple(sorted(t)) for t in zip(z["res_i"].tolist(), z["res_j"].tolist())]
        names = {lab[k]: f"{z['name_i'][k]}{lab[k][0]}-{z['name_j'][k]}{lab[k][1]}" for k in range(len(lab))}
        idx = [k for k in np.where(use)[0]]
        for i, j in pairs:
            Ii = [k for k in idx if i in lab[k]]; Ij = [k for k in idx if j in lab[k]]
            cand = set()
            for k1 in Ii:
                for k2 in Ij:
                    if k1 != k2:
                        cand.add(tuple(sorted((k1, k2), key=lambda k: lab[k])))
            for k1, k2 in sorted(cand, key=lambda kk: (lab[kk[0]], lab[kk[1]])):
                Rb, _ = pair_band_scalar(M[:, k1], M[:, k2], B, tdec, floor=POW_FLOOR)
                ep, cov = ep_stats(Rb, B, t0)
                r_all, r_ep = corr_in(M[:, k1].astype(float), M[:, k2].astype(float), ep, t0, B["fvalid"])
                sub = f"{lab[k1][0]}-{lab[k1][1]} | {lab[k2][0]}-{lab[k2][1]}"
                tag = f"{i}-{j}|hb|{sub}"
                store(tag, r, np.stack([M[:, k1], M[:, k2]]).astype(np.uint8), Rb, B, t0, ep)
                rows.append(dict(pair=f"{i}-{j}", observable="hb", sub=sub, replica=r,
                                 names=f"{names[lab[k1]]} / {names[lab[k2]]}",
                                 occupancy=f"{occ[k1]:.2f}/{occ[k2]:.2f}",
                                 mean_R=float(Rb[B["valid"]].mean()), coverage=cov, n_episodes=len(ep),
                                 episodes=";".join(f"{s + t0:.0f}-{e + t0:.0f}" for s, e, _ in ep),
                                 whole_r=r_all, r_episodes=";".join(f"{v:.3f}" for v in r_ep)))
        # ---------------- K88-D92 split (backbone vs side chain), same H-bond settings and coherence functions
        if a.k88_split:
            from kras_k88_split import atom_signals
            Sk, nF = atom_signals(r, argparse.Namespace(indir=a.indir, d_a=3.2, angle=150.0))
            if nF != T:
                sys.exit(f"[{r}] K88 split: {nF} frames, H-bond matrix {T}")
            for A, Bn in (("E91", "D92sc"), ("E91", "D92bb")):
                x1, x2 = Sk[A], Sk[Bn]
                Rb, _ = pair_band_scalar(x1, x2, B, tdec, floor=POW_FLOOR)
                ep, cov = ep_stats(Rb, B, t0)
                r_all, r_ep = corr_in(x1.astype(float), x2.astype(float), ep, t0, B["fvalid"])
                sub = f"{A} | {Bn}"; tag = f"88-92|hbsplit|{sub}"
                store(tag, r, np.stack([x1, x2]).astype(np.uint8), Rb, B, t0, ep)
                rows.append(dict(pair="88-92", observable="hbsplit", sub=sub, replica=r, names=sub,
                                 occupancy=f"{x1.mean():.2f}/{x2.mean():.2f}",
                                 mean_R=float(Rb[B["valid"]].mean()), coverage=cov, n_episodes=len(ep),
                                 episodes=";".join(f"{s + t0:.0f}-{e + t0:.0f}" for s, e, _ in ep),
                                 whole_r=r_all, r_episodes=";".join(f"{v:.3f}" for v in r_ep)))
        print(f"[{r}] done")
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(out, "summary.csv"), index=False, float_format="%.4g")
    blob["meta"] = np.array(json.dumps(dict(base=BASE, pairs=a.pairs, reps=a.reps)))
    np.savez_compressed(os.path.join(out, "signals.npz"), **blob)
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        g = S.groupby(["pair", "observable", "sub"]).agg(n_reps=("n_episodes", lambda v: int((v > 0).sum())),
                                                         cov_max=("coverage", "max"), mean_R=("mean_R", "mean"))
        print(g.sort_values(["n_reps", "cov_max"], ascending=False).to_string())
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
