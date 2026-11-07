# Figure S2. Time-shifted controls and sensitivity analysis

## Reproduce the figure

From the repository root (or run `../../run_all.sh`):

```
python Supplementary_information/FigureS2_controls_and_sensitivity/figure/figS2_controls_sensitivity.py
```

Output is written to `figure/`.

## data/

| File | Content | Produced by (`analysis/`) |
|---|---|---|
| `cnull_summary.csv, cnull_vs_real.csv` | coherent pairs, real and 200 controls | kras_hmr_coherence_null.py, kras_hmr_stats.py |
| `sens_items.csv, sens_summary.csv` | replicas per pair under the nine settings | kras_hmr_sensitivity.py |
| `blocks_summary.csv` | replicas per pair in each half of the trajectory | kras_hmr_md_checks.py |
| `ctrl_*_per_assoc.csv, ctrl_reswise_per_control.csv` | controls reaching each pair under each control scheme | kras_hmr_stats.py, kras_hmr_controls_review.py |
| `smooth_*.csv` | smoothing 1s and 3s, real and controls | kras_hmr_smoothing.py |
| `*_network_edges.csv, hbond_hyperedges.csv, dms_couplings_canonical.csv` | read by `common.py` (coherent pairs and residue labels) | see Figure 1 |

## analysis/

The scripts that produce `data/` from the trajectories, with the modules they import.
They expect the trajectories and the `wco_hmr/` working folder described in the top-level README.

Steps that make the inputs of every figure from the trajectories (run in this order, from the `work/` folder
described in the top-level README; outputs go to `wco_hmr/`):

1. `kras_hmr_prep.py --from-wrapped wco_hmr/namd_3 wco_hmr/namd_3_2 wco_hmr/namd_3_3` — χ angles (`{rep}_chi.npz`) from the deposited wrapped trajectories.
2. `kras_hmr_sasa.py`, `kras_hmr_hbond.py` — per-residue SASA and residue-pair H-bond signals.
3. `kras_hmr_wco.py`, `kras_hmr_sasa_wco.py`, `kras_hmr_hbond_wco.py` — wavelet coherence and coherent pairs (baseline: 80–120 ns, 0.75, 500 ns).
4. `kras_hmr_runcheck.py` — shared run construction used by the wrappers below (not run directly).
