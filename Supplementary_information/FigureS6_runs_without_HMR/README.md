# Figure S6. Runs with HMR at 4 fs and without HMR at 2 fs

## Reproduce the figure

From the repository root (or run `../../run_all.sh`):

```
python Supplementary_information/FigureS6_runs_without_HMR/figure/figS6_hmr_vs_2fs.py
```

Output is written to `figure/`.

## data/

| File | Content | Produced by (`analysis/`) |
|---|---|---|
| `per_signal_replica.csv` | level and autocorrelation at 10 ns of each signal, per replica (first 1.5 µs) | kras_vanilla_check.py |
| `hmr_corr_summary.csv` | correlations shown in the figure (also rewritten by the script) | figure script |
| `kinetics_report.txt` | report of the comparison, including H-bond formation rates | kras_vanilla_check.py |

## analysis/

The scripts that produce `data/` from the trajectories, with the modules they import.
They expect the trajectories and the `wco_hmr/` working folder described in the top-level README.

Steps that make the inputs of every figure from the trajectories (run in this order, from the `work/` folder
described in the top-level README; outputs go to `wco_hmr/`):

1. `kras_hmr_prep.py --from-wrapped wco_hmr/namd_3 wco_hmr/namd_3_2 wco_hmr/namd_3_3` — χ angles (`{rep}_chi.npz`) from the deposited wrapped trajectories.
2. `kras_hmr_sasa.py`, `kras_hmr_hbond.py` — per-residue SASA and residue-pair H-bond signals.
3. `kras_hmr_wco.py`, `kras_hmr_sasa_wco.py`, `kras_hmr_hbond_wco.py` — wavelet coherence and coherent pairs (baseline: 80–120 ns, 0.75, 500 ns).
4. `kras_hmr_runcheck.py` — shared run construction used by the wrappers below (not run directly).
