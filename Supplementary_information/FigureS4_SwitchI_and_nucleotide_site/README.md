# Figure S4. Switch I and nucleotide-site distances along the three replicas

## Reproduce the figure

From the repository root (or run `../../run_all.sh`):

```
python Supplementary_information/FigureS4_SwitchI_and_nucleotide_site/figure/figS4_switch_traces.py
```

Output is written to `figure/`.

## data/

| File | Content | Produced by (`analysis/`) |
|---|---|---|
| `switch_states_hmr/` | per frame distances and the Switch I angle φ, with summaries | kras_switch_states.py, kras_core_by_switch1.py |

## analysis/

The scripts that produce `data/` from the trajectories, with the modules they import.
They expect the trajectories and the `wco_hmr/` working folder described in the top-level README.

Steps that make the inputs of every figure from the trajectories (run in this order, from the `work/` folder
described in the top-level README; outputs go to `wco_hmr/`):

1. `kras_hmr_prep.py --from-wrapped wco_hmr/namd_3 wco_hmr/namd_3_2 wco_hmr/namd_3_3` — χ angles (`{rep}_chi.npz`) from the deposited wrapped trajectories.
2. `kras_hmr_sasa.py`, `kras_hmr_hbond.py` — per-residue SASA and residue-pair H-bond signals.
3. `kras_hmr_wco.py`, `kras_hmr_sasa_wco.py`, `kras_hmr_hbond_wco.py` — wavelet coherence and coherent pairs (baseline: 80–120 ns, 0.75, 500 ns).
4. `kras_hmr_runcheck.py` — shared run construction used by the wrappers below (not run directly).
