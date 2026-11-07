#!/usr/bin/env python
"""
kras_core_by_switch1.py -- do the core H-bonds depend on the Switch I state?

Frames are split by the T35 OG1-Mg distance alone (no duration rule):
  T35 on  : T35OG1_Mg <= 2.6 A (state 2)      T35 off : > 2.6 A
Run from the analysis folder after kras_switch_states.py:
  python kras_core_by_switch1.py
Reads   switch_states_hmr/{rep}_measures.csv  and  wco_hmr/{rep}_hbond.npz  (both 1 frame per ns)
Writes  switch_states_hmr/core_by_switch1/
  core_hbonds.csv     per replica and state: frames, occupancy of each bond of the four core pairs,
                      both bonds together, together relative to independent bonds, neither bond
  all_pairs.csv       every H-bonded residue pair: occupancy with T35 on and off, per replica
  report.txt          the core table, and the residue pairs whose occupancy changes most between
                      the two states in the same direction in every replica with enough frames
Frames of the analysed window only (730-9270 ns), as for the coherence.
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="wco_hmr")
    p.add_argument("--states", default="switch_states_hmr")
    p.add_argument("--reps", nargs="+", default=REPS)
    p.add_argument("--min-frames", type=int, default=200,
                   help="a replica enters the all-pairs comparison only with this many frames in each state")
    a = p.parse_args()
    out = os.path.join(a.states, "core_by_switch1")
    os.makedirs(out, exist_ok=True)

    core_rows, pair_rows = [], []
    for rep in a.reps:
        m = pd.read_csv(os.path.join(a.states, f"{rep}_measures.csv"))
        z = np.load(os.path.join(a.indir, f"{rep}_hbond.npz"), allow_pickle=True)
        hb, t = z["hb"].astype(bool), z["time_ns"].astype(float)
        if len(m) != len(t):
            sys.exit(f"{rep}: {len(m)} rows of measures but {len(t)} H-bond frames (rerun kras_switch_states.py "
                     f"with --stride 1)")
        keys = [tuple(sorted((int(i), int(j)))) for i, j in zip(z["res_i"], z["res_j"])]
        col = {k: n for n, k in enumerate(keys)}
        win = (t >= WINDOW[0]) & (t <= WINDOW[1])
        on = win & (m.T35OG1_Mg.values <= SHELL)
        off = win & (m.T35OG1_Mg.values > SHELL)
        for state, sel in (("T35 on", on), ("T35 off", off)):
            n = int(sel.sum())
            for name, (b1, b2) in CORE.items():
                row = dict(replica=rep, state=state, frames=n, pair=name)
                if n == 0 or b1 not in col or b2 not in col:
                    core_rows.append(row)
                    continue
                A, B = hb[sel, col[b1]], hb[sel, col[b2]]
                oa, ob, both = A.mean(), B.mean(), (A & B).mean()
                row.update(occ_first=oa, occ_second=ob, both=both,
                           both_vs_independent=both / (oa * ob) if oa * ob > 0 else np.nan,
                           neither=(~A & ~B).mean())
                core_rows.append(row)
        if on.sum() and off.sum():
            for k, c in col.items():
                pair_rows.append(dict(replica=rep, pair=f"{k[0]}-{k[1]}", frames_on=int(on.sum()),
                                      frames_off=int(off.sum()), occ_on=hb[on, c].mean(), occ_off=hb[off, c].mean()))

    C = pd.DataFrame(core_rows)
    C.to_csv(os.path.join(out, "core_hbonds.csv"), index=False, float_format="%.4f")
    P = pd.DataFrame(pair_rows)
    P["delta_off_minus_on"] = P.occ_off - P.occ_on
    P.to_csv(os.path.join(out, "all_pairs.csv"), index=False, float_format="%.4f")

    lines = ["Core H-bonds with T35 on Mg (<= 2.6 A) and off, frames of 730-9270 ns", ""]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        lines += [C.round(3).to_string(index=False), ""]
        ok = P[(P.frames_on >= a.min_frames) & (P.frames_off >= a.min_frames)]
        reps_ok = sorted(ok.replica.unique())
        lines += [f"Replicas with >= {a.min_frames} frames in each state: {', '.join(reps_ok) or 'none'}"]
        if reps_ok:
            w = ok.pivot_table(index="pair", columns="replica", values="delta_off_minus_on")
            occ = ok.groupby("pair")[["occ_on", "occ_off"]].mean()
            same = (np.sign(w).nunique(axis=1) == 1) & w.notna().all(axis=1)
            w = w[same]
            w["mean_delta"] = w.mean(axis=1)
            w = w.join(occ).reindex(w.mean_delta.abs().sort_values(ascending=False).index)
            lines += ["Residue pairs whose occupancy changes in the same direction in each of these replicas,",
                      "largest mean change first (delta = off - on):", w.head(25).round(3).to_string(), ""]
            core_bonds = [f"{x[0]}-{x[1]}" for b in CORE.values() for x in b]
            rank = w.mean_delta.abs().rank(ascending=False)
            lines += ["Rank of the core bonds in that list (of %d pairs):" % len(w)]
            for cb in core_bonds:
                lines += [f"  {cb}: " + (f"rank {int(rank[cb])}, mean delta {w.loc[cb, 'mean_delta']:+.3f}"
                                         if cb in rank.index else "not in the same-direction list")]
    lines += ["", "Frames are split by distance only. A replica with few frames in one state is shown but its",
              "values there are not reliable; frames in the two states may also differ in when they occur."]
    open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
