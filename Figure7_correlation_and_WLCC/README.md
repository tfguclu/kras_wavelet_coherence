# Figure 7. Coherence compared with whole-trajectory correlation and windowed lagged cross-correlation (WLCC)

## Reproduce the figure

From the repository root (or run `../run_all.sh`):

```
python Figure7_correlation_and_WLCC/figure/fig7_correlation_wlcc.py
```

Output is written to `figure/`.

## data/

| File | Content | Produced by (`analysis/`) |
|---|---|---|
| `pairs_r2_chi.csv, pairs_r2_sasa.csv, pairs_r2_hb.csv` | whole-trajectory correlation of every tested signal pair, per replica | kras_hmr_averages.py |
| `avg_associations.csv` | reproducible pairs with coverage | kras_hmr_averages.py |
| `wlcc_broad_units_results.csv` | WLCC result for every coherent pair | kras_wlcc.py broad |
| `wlcc_band/, wlcc_broad/` | full WLCC outputs, thresholds and controls | kras_wlcc.py (see `analysis/kras_wlcc_explained.md`) |

## analysis/

The scripts that produce `data/` from the trajectories, with the modules they import.
They expect the trajectories and the `wco_hmr/` working folder described in the top-level README.

Steps that make the inputs of every figure from the trajectories (run in this order, from the `work/` folder
described in the top-level README; outputs go to `wco_hmr/`):

1. `kras_hmr_prep.py --from-wrapped wco_hmr/namd_3 wco_hmr/namd_3_2 wco_hmr/namd_3_3` — χ angles (`{rep}_chi.npz`) from the deposited wrapped trajectories.
2. `kras_hmr_sasa.py`, `kras_hmr_hbond.py` — per-residue SASA and residue-pair H-bond signals.
3. `kras_hmr_wco.py`, `kras_hmr_sasa_wco.py`, `kras_hmr_hbond_wco.py` — wavelet coherence and coherent pairs (baseline: 80–120 ns, 0.75, 500 ns).
4. `kras_hmr_runcheck.py` — shared run construction used by the wrappers below (not run directly).
