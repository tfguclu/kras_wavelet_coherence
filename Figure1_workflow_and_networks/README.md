# Figure 1. Workflow and coherent pairs of each observable

## Reproduce the figure

From the repository root (or run `../run_all.sh`):

```
python Figure1_workflow_and_networks/figure/fig1_workflow.py
python Figure1_workflow_and_networks/figure/fig1_networks.py
python Figure1_workflow_and_networks/figure/compose_figure1.py
```

Output is written to `figure/`.

## data/

| File | Content | Produced by (`analysis/`) |
|---|---|---|
| `chi_network_edges.csv` | χ coherent pairs (residue pairs, replicas, coverage) | kras_hmr_wco.py |
| `sasa_network_edges.csv` | SASA coherent pairs | kras_hmr_sasa_wco.py |
| `hbond_hyperedges.csv` | H-bond coherent pairs (pairs of H-bond signals) with their episodes | kras_hmr_hbond_wco.py |
| `dms_couplings_canonical.csv` | DMS coupling terms refitted with MoCHI, canonical numbering | MoCHI refit (Zenodo) |

The counts in panel h (350 → 42 → 16) come from `kras_hmr_coherence_null.py` (200 time-shifted controls), `kras_hmr_stats.py` and `kras_hmr_sensitivity.py`.

## analysis/

The scripts that produce `data/` from the trajectories, with the modules they import.
They expect the trajectories and the `wco_hmr/` working folder described in the top-level README.

Steps that make the inputs of every figure from the trajectories (run in this order, from the `work/` folder
described in the top-level README; outputs go to `wco_hmr/`):

1. `kras_hmr_prep.py --from-wrapped wco_hmr/namd_3 wco_hmr/namd_3_2 wco_hmr/namd_3_3` — χ angles (`{rep}_chi.npz`) from the deposited wrapped trajectories.
2. `kras_hmr_sasa.py`, `kras_hmr_hbond.py` — per-residue SASA and residue-pair H-bond signals.
3. `kras_hmr_wco.py`, `kras_hmr_sasa_wco.py`, `kras_hmr_hbond_wco.py` — wavelet coherence and coherent pairs (baseline: 80–120 ns, 0.75, 500 ns).
4. `kras_hmr_runcheck.py` — shared run construction used by the wrappers below (not run directly).
