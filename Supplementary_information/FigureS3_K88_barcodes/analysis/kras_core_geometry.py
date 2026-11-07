#!/usr/bin/env python
"""
kras_core_geometry.py -- does the Switch II-alpha3 geometry differ with the state of the core H-bond pairs?

For each core H-bond pair (R102, R73, K88, T144) every frame is put in one of four states:
  both   both bonds formed        first   only the first bond     second   only the second bond
  neither                         (same atom-level H-bond signals as the coherence, 1 frame per ns)
For each state the interface coordinates from kras_switch_states.py are summarized:
  CA_62_99   E62-Q99 Calpha distance across the switch-II pocket (main coordinate)
  CA_61_92, CA_62_92, CA_64_95, CA_62_95   further Switch II-alpha3 Calpha distances
  Y71OH_V9CA, Q61CD_PG                     Switch II position (Y71 over the P-loop, Q61 at the gamma-phosphate)
  SB_E62_K88, SB_K88_D92                   K88 salt bridges to E62 (Switch II) and D92 (alpha3); for the K88
                                           pair, SB_K88_D92 partly restates the state itself (a check)
Contrasts per replica:  both - neither  and  first - second  (difference of means, in A),
with a 95% interval from a moving-block bootstrap over time (blocks of --block-ns, default 100 ns),
so that the strong autocorrelation of the signals is kept. Frames of the analysed window only
(730-9270 ns), as for the coherence. Optionally only frames with T35 off Mg (--switch1 off), or on.

Run from the analysis folder after kras_switch_states.py (same inputs as kras_core_by_switch1.py):
  python kras_core_geometry.py                     # all frames of the window
  python kras_core_geometry.py --switch1 off       # T35 off Mg only (removes the Switch I confound)
Reads   switch_states_hmr/{rep}_measures.csv  and  wco_hmr/{rep}_hbond.npz
Writes  switch_states_hmr/core_geometry[_T35on|_T35off]/
  by_state.csv    replica, pair, state, frames, coordinate, mean, median, sd
  contrasts.csv   replica, pair, contrast, coordinate, frames_a, frames_b, diff_mean, ci_low, ci_high
  report.txt      the main coordinate per pair and replica, and contrasts whose interval excludes 0
                  with the same sign in every replica with enough frames
  geometry_by_state.png   E62-Q99 distance by state, per pair and replica
  geometry_K88.png        K88 NZ-E62 and E62-Q99 distances by K88 state, per replica
  k88_frames.csv          with --dump: per-frame K88 state and distances (for the figure)
  r102_frames.csv         with --dump: per-frame R102 state, E62-Q99 and E62-D92 (for the figure)
The analysis is descriptive: a difference between states does not say which causes which, and states
that occur at different times of a run can differ for other reasons (e.g. the one Switch I opening).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

REPS = ["namd_3", "namd_3_2", "namd_3_3"]
WINDOW = (730.0, 9270.0)
SHELL = 2.6
CORE = {"R102 (D69 | Q99)": ((69, 102), (99, 102)), "R73 (V103 | D105)": ((73, 103), (73, 105)),
        "K88 (E91 | D92)": ((88, 91), (88, 92)), "T144 (N116 | Q150)": ((116, 144), (144, 150))}
COORDS = ["CA_62_99", "CA_61_92", "CA_62_92", "CA_64_95", "CA_62_95", "Y71OH_V9CA", "Q61CD_PG",
          "SB_E62_K88", "SB_K88_D92"]
MAIN = "CA_62_99"
STATES = ["both", "first", "second", "neither"]
CONTRASTS = [("both", "neither"), ("first", "second")]


def states_of(A, B):
    s = np.full(len(A), "neither", dtype=object)
    s[A & B] = "both"
    s[A & ~B] = "first"
    s[~A & B] = "second"
    return s


def block_boot(x, s, a, b, block, n_boot, rng):
    """moving-block bootstrap over time of mean(x | s==a) - mean(x | s==b)"""
    n = len(x)
    ia, ib = s == a, s == b
    if ia.sum() == 0 or ib.sum() == 0:
        return np.nan, np.nan, np.nan
    d0 = x[ia].mean() - x[ib].mean()
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    out = np.empty(n_boot)
    for k in range(n_boot):
        idx = (starts[k][:, None] + np.arange(block)[None, :]).ravel()[:n]
        xa, xb = x[idx][ia[idx]], x[idx][ib[idx]]
        out[k] = xa.mean() - xb.mean() if len(xa) and len(xb) else np.nan
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return d0, lo, hi


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="wco_hmr")
    p.add_argument("--states", default="switch_states_hmr")
    p.add_argument("--reps", nargs="+", default=REPS)
    p.add_argument("--switch1", choices=["all", "on", "off"], default="all",
                   help="use all frames, or only frames with T35 on (<= 2.6 A) or off Mg")
    p.add_argument("--block-ns", type=float, default=100.0, help="bootstrap block length")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--min-frames", type=int, default=200,
                   help="a state enters a contrast only with this many frames")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dump", action="store_true",
                   help="also write k88_frames.csv: time, replica, K88 state, K88 NZ-E62 and E62-Q99 per frame")
    a = p.parse_args()
    tag = "" if a.switch1 == "all" else f"_T35{a.switch1}"
    out = os.path.join(a.states, "core_geometry" + tag)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    rows, crow, plotdata, dump, dump102 = [], [], {}, [], []
    for rep in a.reps:
        m = pd.read_csv(os.path.join(a.states, f"{rep}_measures.csv"))
        z = np.load(os.path.join(a.indir, f"{rep}_hbond.npz"), allow_pickle=True)
        hb, t = z["hb"].astype(bool), z["time_ns"].astype(float)
        if len(m) != len(t):
            sys.exit(f"{rep}: {len(m)} rows of measures but {len(t)} H-bond frames (rerun kras_switch_states.py "
                     f"with --stride 1)")
        missing = [c for c in COORDS if c not in m.columns]
        if missing:
            sys.exit(f"{rep}: measures file lacks {missing}")
        dt = float(np.median(np.diff(t)))
        block = max(1, int(round(a.block_ns / dt)))
        keys = [tuple(sorted((int(i), int(j)))) for i, j in zip(z["res_i"], z["res_j"])]
        col = {k: n for n, k in enumerate(keys)}
        sel = (t >= WINDOW[0]) & (t <= WINDOW[1])
        if a.switch1 == "on":
            sel &= m.T35OG1_Mg.values <= SHELL
        elif a.switch1 == "off":
            sel &= m.T35OG1_Mg.values > SHELL
        if sel.sum() < block * 2:
            print(f"{rep}: only {int(sel.sum())} frames selected, skipped")
            continue
        # contiguous only when no Switch I selection; with a selection the bootstrap resamples the kept
        # frames in their time order, which joins the stretches (noted in the report)
        X = {c: m[c].values[sel].astype(float) for c in COORDS}
        for name, (b1, b2) in CORE.items():
            if b1 not in col or b2 not in col:
                print(f"{rep}: {name} bond signal missing, skipped")
                continue
            s = states_of(hb[sel, col[b1]], hb[sel, col[b2]])
            plotdata[(rep, name)] = (s, X[MAIN], X["SB_E62_K88"])
            if a.dump and name.startswith("K88"):
                dump.append(pd.DataFrame(dict(replica=rep, time_ns=t[sel], k88_state=s,
                                              SB_E62_K88=X["SB_E62_K88"], CA_62_99=X[MAIN],
                                              CA_62_92=X["CA_62_92"], SB_K88_D92=X["SB_K88_D92"])))
            if a.dump and name.startswith("R102"):
                dump102.append(pd.DataFrame(dict(replica=rep, time_ns=t[sel], r102_state=s,
                                                 CA_62_99=X[MAIN], CA_62_92=X["CA_62_92"])))
            for st in STATES:
                k = s == st
                for c in COORDS:
                    x = X[c][k]
                    rows.append(dict(replica=rep, pair=name, state=st, frames=int(k.sum()), coordinate=c,
                                     mean=x.mean() if len(x) else np.nan,
                                     median=np.median(x) if len(x) else np.nan,
                                     sd=x.std() if len(x) > 1 else np.nan))
            for sa, sb in CONTRASTS:
                na, nb = int((s == sa).sum()), int((s == sb).sum())
                for c in COORDS:
                    if na < a.min_frames or nb < a.min_frames:
                        d = lo = hi = np.nan
                    else:
                        d, lo, hi = block_boot(X[c], s, sa, sb, block, a.n_boot, rng)
                    crow.append(dict(replica=rep, pair=name, contrast=f"{sa} - {sb}", coordinate=c,
                                     frames_a=na, frames_b=nb, diff_mean=d, ci_low=lo, ci_high=hi))

    if dump:
        pd.concat(dump).to_csv(os.path.join(out, "k88_frames.csv"), index=False, float_format="%.3f")
    if dump102:
        pd.concat(dump102).to_csv(os.path.join(out, "r102_frames.csv"), index=False, float_format="%.3f")
    S = pd.DataFrame(rows)
    C = pd.DataFrame(crow)
    S.to_csv(os.path.join(out, "by_state.csv"), index=False, float_format="%.4f")
    C.to_csv(os.path.join(out, "contrasts.csv"), index=False, float_format="%.4f")

    lines = [f"Switch II-alpha3 geometry by core H-bond state; frames {WINDOW[0]:.0f}-{WINDOW[1]:.0f} ns, "
             f"Switch I selection: {a.switch1}; block bootstrap {a.block_ns:g} ns x {a.n_boot}", ""]
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.max_rows", 500):
        w = S[S.coordinate == MAIN].pivot_table(index=["pair", "state"], columns="replica",
                                                values=["frames", "mean"], aggfunc="first")
        lines += [f"{MAIN} (A), mean per state, and frames:", w.round(2).to_string(), ""]
        C["excl0"] = (C.ci_low > 0) | (C.ci_high < 0)
        g = C.dropna(subset=["diff_mean"]).groupby(["pair", "contrast", "coordinate"])
        consistent = []
        for (pair, con, coord), d in g:
            if len(d) >= 2 and d.excl0.all() and (np.sign(d.diff_mean).nunique() == 1):
                consistent.append(dict(pair=pair, contrast=con, coordinate=coord, replicas=len(d),
                                       diffs=", ".join(f"{v:+.2f}" for v in d.diff_mean),
                                       cis=", ".join(f"[{l:+.2f}, {h:+.2f}]" for l, h in zip(d.ci_low, d.ci_high))))
        lines += ["Contrasts whose 95% interval excludes 0, with the same sign, in every replica that has",
                  f">= {a.min_frames} frames in both states (and at least two replicas):"]
        lines += [pd.DataFrame(consistent).to_string(index=False) if consistent else "  none", ""]
        lines += ["All contrasts for the main coordinate:",
                  C[C.coordinate == MAIN].drop(columns="excl0").round(2).to_string(index=False), ""]
    lines += ["Notes: differences are between frame sets, not effects of one bond on the geometry. States",
              "that occur at different times can differ for other reasons (relaxation, the single Switch I",
              "opening per run); rerun with --switch1 off to compare within the open Switch I state."]
    if a.switch1 != "all":
        lines += ["With a Switch I selection the bootstrap blocks run over the kept frames in time order,",
                  "so a block can join two separate stretches."]
    open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        reps = [r for r in a.reps if any(k[0] == r for k in plotdata)]
        fig, axes = plt.subplots(len(CORE), len(reps), figsize=(2.6 * len(reps), 2.0 * len(CORE)),
                                 sharey=True, squeeze=False)
        for i, name in enumerate(CORE):
            for j, rep in enumerate(reps):
                ax = axes[i][j]
                if (rep, name) not in plotdata:
                    ax.axis("off")
                    continue
                s, x, _ = plotdata[(rep, name)]
                data = [x[s == st] for st in STATES]
                keep = [k for k, d in enumerate(data) if len(d) >= 20]
                if keep:
                    ax.boxplot([data[k] for k in keep], positions=keep, widths=0.6, showfliers=False)
                ax.set_xticks(range(4))
                ax.set_xticklabels([f"{st}\n{len(d)}" for st, d in zip(STATES, data)], fontsize=6)
                if i == 0:
                    ax.set_title(rep, fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"{name}\nE62-Q99 CA (A)", fontsize=7)
                ax.tick_params(labelsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "geometry_by_state.png"), dpi=200)
        # panel for the K88 alternation: K88 NZ to the E62 carboxylate and E62-Q99 CA, by K88 state
        k88 = "K88 (E91 | D92)"
        fig, axes = plt.subplots(2, len(reps), figsize=(2.6 * len(reps), 4.0), squeeze=False)
        for j, rep in enumerate(reps):
            if (rep, k88) not in plotdata:
                continue
            s, xq, xk = plotdata[(rep, k88)]
            for i, (x, lab) in enumerate([(xk, "K88 NZ - E62 carboxylate (A)"), (xq, "E62-Q99 CA (A)")]):
                ax = axes[i][j]
                data = [x[s == st] for st in STATES]
                keep = [k for k, d in enumerate(data) if len(d) >= 20]
                if keep:
                    ax.boxplot([data[k] for k in keep], positions=keep, widths=0.6, showfliers=False)
                if i == 0:
                    ax.axhline(3.2, ls="--", lw=0.6, color="0.5")
                    ax.set_title(rep, fontsize=8)
                ax.set_xticks(range(4))
                ax.set_xticklabels(["both", "E91 only", "D92 only", "neither"], fontsize=6)
                if j == 0:
                    ax.set_ylabel(lab, fontsize=7)
                ax.tick_params(labelsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "geometry_K88.png"), dpi=200)
    except Exception as e:  # the tables are the result; the figure is a convenience
        print(f"figure skipped: {e}")


if __name__ == "__main__":
    main()
