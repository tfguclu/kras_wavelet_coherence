# Figure S3. K88 H-bonds, Switch II opening and Switch II–α3 distance over time

## Reproduce the figure

From the repository root (or run `../../run_all.sh`):

```
python Supplementary_information/FigureS3_K88_barcodes/figure/figS3_k88_barcodes.py
```

Output is written to `figure/`.

## data/

| File | Content | Produced by (`analysis/`) |
|---|---|---|
| `k88_signals.npz` | K88 H-bond signals (side chain and backbone) | kras_hmr_pairsignals.py --k88-split |
| `k88_frames.csv` | per frame: K88 H-bond states and the K88–E62 H-bond | kras_core_geometry.py --dump |
| `namd_3_measures.csv, namd_3_2_measures.csv, namd_3_3_measures.csv` | per frame: Switch I/II and nucleotide-site distances | kras_switch_states.py |
| `k88_e62_by_state.csv` | K88–E62 H-bond fraction by K88 state (Section 2.2) | kras_core_geometry.py |

`k88_barcode_occupancy.csv` (occupancy per row and replica) is written by the figure script.

## analysis/

The scripts that produce `data/` from the trajectories, with the modules they import.
They expect the trajectories and the `wco_hmr/` working folder described in the top-level README.

Steps that make the inputs of every figure from the trajectories (run in this order, from the `work/` folder
described in the top-level README; outputs go to `wco_hmr/`):

1. `kras_hmr_prep.py --from-wrapped wco_hmr/namd_3 wco_hmr/namd_3_2 wco_hmr/namd_3_3` — χ angles (`{rep}_chi.npz`) from the deposited wrapped trajectories.
2. `kras_hmr_sasa.py`, `kras_hmr_hbond.py` — per-residue SASA and residue-pair H-bond signals.
3. `kras_hmr_wco.py`, `kras_hmr_sasa_wco.py`, `kras_hmr_hbond_wco.py` — wavelet coherence and coherent pairs (baseline: 80–120 ns, 0.75, 500 ns).
4. `kras_hmr_runcheck.py` — shared run construction used by the wrappers below (not run directly).
