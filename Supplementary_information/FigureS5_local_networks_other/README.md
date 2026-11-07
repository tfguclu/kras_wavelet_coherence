# Figure S5. Local networks of the other 18 reproducible pairs

## Reproduce the figure

From the repository root (or run `../../run_all.sh`):

```
python Supplementary_information/FigureS5_local_networks_other/figure/fig_network_episodes.py --part B
```

Output is written to `figure/`.

## data/

| File | Content | Produced by (`analysis/`) |
|---|---|---|
| `episodes_all_reps_rotamers.csv, episodes_all_reps_SASA.csv` | coherent episodes (start–end, ns) of every χ and SASA pair in each replica | kras_hmr_wco.py, kras_hmr_sasa_wco.py |
| `*_network_edges.csv, hbond_hyperedges.csv, dms_couplings_canonical.csv` | read by `common.py` (coherent pairs and residue labels) | see Figure 1 |

Same script as Figure 4 (`--part A`).

## analysis/

The scripts that produce `data/` from the trajectories, with the modules they import.
They expect the trajectories and the `wco_hmr/` working folder described in the top-level README.

Steps that make the inputs of every figure from the trajectories (run in this order, from the `work/` folder
described in the top-level README; outputs go to `wco_hmr/`):

1. `kras_hmr_prep.py --from-wrapped wco_hmr/namd_3 wco_hmr/namd_3_2 wco_hmr/namd_3_3` — χ angles (`{rep}_chi.npz`) from the deposited wrapped trajectories.
2. `kras_hmr_sasa.py`, `kras_hmr_hbond.py` — per-residue SASA and residue-pair H-bond signals.
3. `kras_hmr_wco.py`, `kras_hmr_sasa_wco.py`, `kras_hmr_hbond_wco.py` — wavelet coherence and coherent pairs (baseline: 80–120 ns, 0.75, 500 ns).
4. `kras_hmr_runcheck.py` — shared run construction used by the wrappers below (not run directly).
