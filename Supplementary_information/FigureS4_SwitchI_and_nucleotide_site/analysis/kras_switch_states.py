#!/usr/bin/env python
"""
kras_switch_states.py -- Mg2+ coordination, Switch I and Switch II states along the HMR runs.

Run from the analysis folder that holds wco_hmr/ (reads only the dry wrapped trajectories the
pipeline wrote: wco_hmr/{namd_3,namd_3_2,namd_3_3}_wrapped.pdb/.dcd, protein + GTP + Mg, 1 ns).
  python kras_switch_states.py                 # all replicas, every frame
  python kras_switch_states.py --stride 2      # every 2nd frame
  python kras_switch_states.py --from-csv      # redo summary and plots from the saved measures
All states are assigned frame by frame from distances and angles; no minimum duration is used.
Outputs (new folder, nothing else is written): switch_states_hmr/
  {rep}_measures.csv   one row per frame: time_ns and every measure below
  summary.csv          per replica and measure: mean, SD, 5/50/95 percentiles, fraction in state;
                       for the whole run and for the analysed window of the coherence (730-9270 ns)
  histograms.png, timeseries.png, report.txt

MEASURES (atom names as in the CHARMM topology; distances in A, angle in rad)
 Mg2+ first shell (protein/GTP oxygens; the two water ligands are not in the dry trajectory)
  mg_n_shell        number of protein/GTP O within 2.6 A of Mg (start model: S17 OG, T35 OG1, one
                    beta-O, one gamma-O -> 4)
  mg_S17, mg_T35, mg_betaO, mg_gammaO   distances of each expected ligand
 Switch I
  T35OG1_Mg         state 2 (T35 coordinates Mg) <= 2.6 A, open > 2.6 A, frame by frame
                    (state 1/state 2: Spoerner et al. 2001)
  sw1_phi           angle between Q25 CA->S17 CA and Q25 CA->T35 CA (Hu et al., JACS Au 2026,
                    6:830; CV1). WT-GTP basins: ~0.25 rad closed, ~1.23 moderately open, ~3.0 open
  Y32OH_G12CA       Y32 in (< 10.5 A) / out (Parker et al., Cancer Res 2022, 82:2485; Rascore)
  Y32OH_gammaO      Y32 OH to the nearest gamma-phosphate O; direct H-bond <= 3.2 A (Rascore
                    subdivides GTP-bound Switch I by this H-bond: direct, water-mediated, none;
                    water-mediated cannot be seen in the dry trajectory)
 Switch II
  G60N_gammaO       G60 backbone N to the nearest gamma-O; H-bond <= 3.2 A ("loaded spring",
                    Vetter & Wittinghofer 2001)
  Q61CD_PG          Hu et al. JACS Au 2026 CV2 (d1). WT-GTP basins: ~4.5 A (Q61 H-bonded to the
                    gamma-phosphate), ~6.8 A (bridging water, hydrolysis-competent), ~12.8 A (open)
  Y71OH_V9CA        Y71 in (< 8.75 A) / out (Parker et al. 2022)
  CA_61_92, CA_62_92, CA_64_95, CA_62_95   Switch II-alpha3 C-alpha distances (Vatansever et al.,
                    Sci Rep 2019, 9:11730; WT-GTP, AMBER ff99SB): "first" conformation peaks
                    8.3 / 10 / 8.8 A (61-92 / 62-92 / 64-95), "second" 16 / 15.8 / 11.6 A
                    (their Fig. 3 labels the middle pair E62-H95; the text gives 62-92, and
                    lists 62-95 too: both are reported here)
  CA_62_99          E62-Q99 C-alpha, Switch II pocket opening (manuscript ref 16)
  SB_E62_K88, SB_K88_D92   salt bridges of the "first" (close) conformation (Vatansever 2019)
  HB_R68_D92        H-bond of the "second" (distant) conformation (Vatansever 2019)
  SB_E63_R68        intra-Switch II salt bridge of active WT (Vatansever 2019)
  SB_K16_E63        P-loop-Switch II salt bridge, two states in WT-GTP (Vatansever 2019)
                    (salt bridge / H-bond formed: closest O-N <= 3.2 A, the VMD plugin default)
 GTP
  G12CA_PG          Hu et al. JACS Au 2026 CV3 (d2): ~4.3 A phosphates bound to the P-loop,
                    ~7 A detached
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

REPS = ["namd_3", "namd_3_2", "namd_3_3"]
WINDOW = (730.0, 9270.0)
SHELL = 2.6
HB = 3.2
GAMMA_O = "resname GTP and name O1G O2G O3G"
BETA_O = "resname GTP and name O1B O2B O3B"

# (column, kind, selection A, selection B, state rule, label for plots)
#   kind: "d" min distance between A and B; "phi": angle; rule: (op, value, state name) or None
MEASURES = [
    ("mg_S17", "d", "resid 17 and name OG", "resname MG", ("<=", SHELL, "in shell")),
    ("mg_T35", "d", "resid 35 and name OG1", "resname MG", ("<=", SHELL, "in shell")),
    ("mg_betaO", "d", BETA_O, "resname MG", ("<=", SHELL, "in shell")),
    ("mg_gammaO", "d", GAMMA_O, "resname MG", ("<=", SHELL, "in shell")),
    ("T35OG1_Mg", "d", "resid 35 and name OG1", "resname MG", ("<=", SHELL, "state 2 (T35 on Mg)")),
    ("sw1_phi", "phi", "resid 25 and name CA", ("resid 17 and name CA", "resid 35 and name CA"), None),
    ("Y32OH_G12CA", "d", "resid 32 and name OH", "resid 12 and name CA", ("<", 10.5, "Y32 in")),
    ("Y32OH_gammaO", "d", "resid 32 and name OH", GAMMA_O, ("<=", HB, "direct H-bond")),
    ("G60N_gammaO", "d", "resid 60 and name N", GAMMA_O, ("<=", HB, "H-bond")),
    ("Q61CD_PG", "d", "resid 61 and name CD", "resname GTP and name PG", None),
    ("Y71OH_V9CA", "d", "resid 71 and name OH", "resid 9 and name CA", ("<", 8.75, "Y71 in")),
    ("CA_61_92", "d", "resid 61 and name CA", "resid 92 and name CA", None),
    ("CA_62_92", "d", "resid 62 and name CA", "resid 92 and name CA", None),
    ("CA_64_95", "d", "resid 64 and name CA", "resid 95 and name CA", None),
    ("CA_62_95", "d", "resid 62 and name CA", "resid 95 and name CA", None),
    ("CA_62_99", "d", "resid 62 and name CA", "resid 99 and name CA", None),
    ("SB_E62_K88", "d", "resid 62 and name OE1 OE2", "resid 88 and name NZ", ("<=", HB, "formed")),
    ("SB_K88_D92", "d", "resid 88 and name NZ", "resid 92 and name OD1 OD2", ("<=", HB, "formed")),
    ("HB_R68_D92", "d", "resid 68 and name NE NH1 NH2", "resid 92 and name OD1 OD2", ("<=", HB, "formed")),
    ("SB_E63_R68", "d", "resid 63 and name OE1 OE2", "resid 68 and name NE NH1 NH2", ("<=", HB, "formed")),
    ("SB_K16_E63", "d", "resid 16 and name NZ", "resid 63 and name OE1 OE2", ("<=", HB, "formed")),
    ("G12CA_PG", "d", "resid 12 and name CA", "resname GTP and name PG", None),
]
# literature marks drawn on the histograms (value, short label)
MARKS = {
    "T35OG1_Mg": [(SHELL, "2.6")], "mg_S17": [(SHELL, "2.6")], "mg_betaO": [(SHELL, "2.6")],
    "mg_gammaO": [(SHELL, "2.6")], "mg_T35": [(SHELL, "2.6")],
    "sw1_phi": [(0.25, "closed"), (1.23, "mod. open"), (3.0, "open")],
    "Y32OH_G12CA": [(10.5, "10.5")], "Y32OH_gammaO": [(HB, "3.2")], "G60N_gammaO": [(HB, "3.2")],
    "Q61CD_PG": [(4.5, "H-bond"), (6.8, "water"), (12.8, "open")], "Y71OH_V9CA": [(8.75, "8.75")],
    "CA_61_92": [(8.3, "1st"), (16.0, "2nd")], "CA_62_92": [(10.0, "1st"), (15.8, "2nd")],
    "CA_64_95": [(8.8, "1st"), (11.6, "2nd")], "CA_62_99": [(12.0, "12.0"), (22.4, "22.4")],
    "SB_E62_K88": [(HB, "3.2")], "SB_K88_D92": [(HB, "3.2")], "HB_R68_D92": [(HB, "3.2")],
    "SB_E63_R68": [(HB, "3.2")], "SB_K16_E63": [(HB, "3.2")], "G12CA_PG": [(4.3, "bound"), (7.0, "detached")],
}


def compute(top, traj, stride, dt_ns):
    import warnings
    warnings.filterwarnings("ignore")
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import distance_array
    u = mda.Universe(top, traj) if traj else mda.Universe(top)
    groups = {}
    for name, kind, a, b, _ in MEASURES:
        if kind == "d":
            A, B = u.select_atoms(a), u.select_atoms(b)
            if not A.n_atoms or not B.n_atoms:
                sys.exit(f"{name}: empty selection ({a!r}: {A.n_atoms}, {b!r}: {B.n_atoms})")
            groups[name] = (A, B)
        else:
            A, B1, B2 = u.select_atoms(a), u.select_atoms(b[0]), u.select_atoms(b[1])
            if not (A.n_atoms == B1.n_atoms == B2.n_atoms == 1):
                sys.exit(f"{name}: each angle atom must select exactly one atom")
            groups[name] = (A, B1, B2)
    mg = u.select_atoms("resname MG")
    shell_O = u.select_atoms("(protein or resname GTP) and name O*")
    rows = []
    for ts in u.trajectory[::stride]:
        box = ts.dimensions if ts.dimensions is not None and np.all(ts.dimensions[:3] > 0) else None
        r = {"time_ns": ts.frame * dt_ns}
        for name, kind, *_ in MEASURES:
            g = groups[name]
            if kind == "d":
                r[name] = float(distance_array(g[0].positions, g[1].positions, box=box).min())
            else:
                v1 = g[1].positions[0] - g[0].positions[0]
                v2 = g[2].positions[0] - g[0].positions[0]
                c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                r[name] = float(np.arccos(np.clip(c, -1, 1)))
        r["mg_n_shell"] = int((distance_array(mg.positions, shell_O.positions, box=box) <= SHELL).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def in_state(x, rule):
    op, v, _ = rule
    return x <= v if op == "<=" else x < v


def summarize(D, out):
    rows = []
    for rep, d in D.items():
        for part, sel in (("whole run", np.ones(len(d), bool)),
                          ("analysed window", d.time_ns.between(*WINDOW).values)):
            x = d[sel]
            if not len(x):
                continue
            for name, kind, a, b, rule in MEASURES + [("mg_n_shell", "n", None, None, None)]:
                v = x[name].values
                row = dict(replica=rep, part=part, measure=name, mean=v.mean(), sd=v.std(),
                           p05=np.percentile(v, 5), p50=np.median(v), p95=np.percentile(v, 95))
                if rule:
                    row["state"] = rule[2]
                    row["fraction_in_state"] = float(in_state(v, rule).mean())
                if name == "mg_n_shell":
                    for k in range(0, 7):
                        row[f"frac_n{k}"] = float((v == k).mean())
                rows.append(row)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(out, "summary.csv"), index=False, float_format="%.4g")
    return S


def plots(D, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"namd_3": "#1f5aa6", "namd_3_2": "#d08c1f", "namd_3_3": "#2a9d8f"}
    names = [m[0] for m in MEASURES] + ["mg_n_shell"]
    n = len(names)
    nc = 4
    nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.3 * nr))
    for ax, name in zip(axs.ravel(), names):
        for rep, d in D.items():
            v = d[name].values
            if name == "mg_n_shell":
                ks = np.arange(0, 7)
                ax.plot(ks, [(v == k).mean() for k in ks], "o-", ms=3, color=cols[rep], lw=1, label=rep)
            else:
                ax.hist(v, bins=80, histtype="step", density=True, color=cols[rep], lw=1, label=rep)
        for val, lab in MARKS.get(name, []):
            ax.axvline(val, color="#666", lw=0.7, ls="--")
            ax.text(val, ax.get_ylim()[1] * 0.95, lab, fontsize=6, rotation=90, va="top", ha="right", color="#444")
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    axs.ravel()[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "histograms.png"), dpi=160)
    plt.close(fig)

    key = ["T35OG1_Mg", "sw1_phi", "Y32OH_gammaO", "Y32OH_G12CA", "G60N_gammaO", "Q61CD_PG",
           "Y71OH_V9CA", "CA_61_92", "CA_62_99", "mg_n_shell"]
    fig, axs = plt.subplots(len(key), 3, figsize=(14, 1.3 * len(key)), sharex=True)
    for j, (rep, d) in enumerate(D.items()):
        for i, name in enumerate(key):
            ax = axs[i, j]
            ax.plot(d.time_ns / 1000, d[name], lw=0.3, color=cols[rep])
            for val, _ in MARKS.get(name, []):
                ax.axhline(val, color="#888", lw=0.5, ls="--")
            ax.axvspan(0, WINDOW[0] / 1000, color="#eee", lw=0)
            ax.axvspan(WINDOW[1] / 1000, d.time_ns.max() / 1000, color="#eee", lw=0)
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_ylabel(name, fontsize=6)
            if i == 0:
                ax.set_title(rep, fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel("time (µs)", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "timeseries.png"), dpi=160)
    plt.close(fig)


def report(S, out):
    w = S[S.part == "analysed window"]
    if w.empty:
        w = S[S.part == "whole run"]
    lines = ["Switch I / Switch II / Mg states, HMR runs (analysed window 730-9270 ns unless stated)", ""]
    st = w.dropna(subset=["fraction_in_state"]).pivot_table(index=["measure", "state"], columns="replica",
                                                              values="fraction_in_state")
    lines += ["Fraction of frames in state:", st.round(3).to_string(), ""]
    md = w.pivot_table(index="measure", columns="replica", values="p50")
    lines += ["Medians (A; sw1_phi in rad):", md.round(2).to_string(), ""]
    ns = w[w.measure == "mg_n_shell"].set_index("replica")[[f"frac_n{k}" for k in range(0, 7)]]
    lines += ["Mg first shell (protein/GTP O within 2.6 A), fraction of frames with n ligands:",
              ns.round(3).to_string(), ""]
    lines += ["Literature values are references, not cutoffs for this force field: check that a",
              "measure is bimodal in histograms.png before reading it as two states."]
    open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="wco_hmr")
    p.add_argument("--reps", nargs="+", default=REPS)
    p.add_argument("--out", default="switch_states_hmr")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--dt-ns", type=float, default=1.0, help="time between saved frames in the input")
    p.add_argument("--pdb-only", action="store_true", help="test: measure the PDB frame only")
    p.add_argument("--from-csv", action="store_true", help="reuse {rep}_measures.csv in --out (no trajectory reading)")
    a = p.parse_args()
    if os.path.realpath(a.out) == os.path.realpath(a.indir) or os.path.realpath(a.out).startswith(
            os.path.realpath(a.indir) + os.sep):
        sys.exit("--out must not be inside --indir")
    os.makedirs(a.out, exist_ok=True)
    D = {}
    for rep in a.reps:
        if a.from_csv:
            D[rep] = pd.read_csv(os.path.join(a.out, f"{rep}_measures.csv"))
            continue
        top = os.path.join(a.indir, f"{rep}_wrapped.pdb")
        traj = None if a.pdb_only else os.path.join(a.indir, f"{rep}_wrapped.dcd")
        t0 = time.time()
        d = compute(top, traj, a.stride, a.dt_ns)
        d.to_csv(os.path.join(a.out, f"{rep}_measures.csv"), index=False, float_format="%.3f")
        D[rep] = d
        print(f"[{rep}] {len(d)} frames, {time.time() - t0:.0f} s", flush=True)
    if a.pdb_only:
        print(pd.concat(D).T.round(2).to_string())
        return
    S = summarize(D, a.out)
    plots(D, a.out)
    report(S, a.out)


if __name__ == "__main__":
    main()
