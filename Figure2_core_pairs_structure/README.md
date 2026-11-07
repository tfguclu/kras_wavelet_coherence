# Figure 2. Core coherent pairs on the structure and sequence

## Reproduce the figure

From the repository root (or run `../run_all.sh`):

```
python Figure2_core_pairs_structure/figure/fig2_structure_panel.py
python Figure2_core_pairs_structure/figure/fig2_sequence_map.py
python Figure2_core_pairs_structure/figure/compose_figure2.py
```

Output is written to `figure/`.

## data/

| File | Content | Produced by (`analysis/`) |
|---|---|---|
| `view_front.png, view_back.png` | VMD renders of the starting model (re-render with `figure/render/kras_views_vmd.tcl`) | VMD |
| `fit.json` | projection used to place the residue labels on the renders | `figure/fit_projection.py` |
| `kras_start_model.pdb` | starting model (Boltz-2, with the two Mg²⁺ waters) | — |

## analysis/

The scripts that produce `data/` from the trajectories, with the modules they import.
They expect the trajectories and the `wco_hmr/` working folder described in the top-level README.

Steps that make the inputs of every figure from the trajectories (run in this order, from the `work/` folder
described in the top-level README; outputs go to `wco_hmr/`):

1. `kras_hmr_prep.py --from-wrapped wco_hmr/namd_3 wco_hmr/namd_3_2 wco_hmr/namd_3_3` — χ angles (`{rep}_chi.npz`) from the deposited wrapped trajectories.
2. `kras_hmr_sasa.py`, `kras_hmr_hbond.py` — per-residue SASA and residue-pair H-bond signals.
3. `kras_hmr_wco.py`, `kras_hmr_sasa_wco.py`, `kras_hmr_hbond_wco.py` — wavelet coherence and coherent pairs (baseline: 80–120 ns, 0.75, 500 ns).
4. `kras_hmr_runcheck.py` — shared run construction used by the wrappers below (not run directly).
