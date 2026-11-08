# Wavelet coherence captures time-resolved coordination at the Switch II–α3 interface of unbound KRAS

Code and data behind the figures of

> Tandac F. Guclu. *Wavelet coherence captures time-resolved coordination at the Switch II–α3 interface of unbound KRAS.* (submitted)

The molecular dynamics trajectories (3 × 10 µs, KRAS4B·GTP·Mg²⁺, NAMD 3.0.1, CHARMM36m, HMR, 1 ns frames), the
three 1.5 µs runs without HMR and the MoCHI refit of the DMS data are on Zenodo: **https://doi.org/10.5281/zenodo.24052273** (to be added).

## Layout

One folder per main-text figure and one folder for the Supporting Information. Every figure folder has the same structure:

```
FigureN_.../
  data/       inputs of the figure (outputs of the analysis, small enough for GitHub)
  figure/     plotting script(s) and shared style (pub_style.py); the figure is written here when the scripts run
  analysis/   scripts that produce data/ from the trajectories, with the modules they import
  README.md   what each data file contains and which analysis script writes it
```

| Folder | Content |
|---|---|
| `Figure1_workflow_and_networks` | Figure 1: workflow, coherent pairs of each observable, reduction to the core (350 → 42 → 16) |
| `Figure2_core_pairs_structure` | Figure 2: core pairs on the structure and along the sequence |
| `Figure3_phase_and_joint_states` | Figure 3: relative phase, joint states and co-occupancy of the core H-bond pairs |
| `Figure4_local_networks` | Figure 4: local networks of reproducible pairs and their episode lengths |
| `Figure5_DMS_and_evolutionary_couplings` | Figure 5: DMS and evolutionary couplings and their overlap with the reproducible pairs |
| `Figure6_timescales` | Figure 6: period-band scan |
| `Figure7_correlation_and_WLCC` | Figure 7: coherence versus whole-trajectory correlation and WLCC |
| `Figure8_wavelet_coherence_method` | Figure 8: method on synthetic signals |
| `Supplementary_information/FigureS1_…` – `FigureS7_…` | Figures S1–S7 |
| `Supplementary_information/Tables_S1_S3` | Tables S1–S3 as CSV |
| `Supplementary_information/Supplementary_methods_checks` | step test, pycwt comparison and other checks cited in the Supplementary methods |
| `TOC_graphic` | table of contents graphic |

The analysis scripts are copied into each folder that needs them, so every folder can be used on its own; the copies are identical.

## Reproduce the figures from the data in this repository

```bash
pip install -r requirements.txt
bash run_all.sh          # writes every figure into its folder's figure/ (PNG files are not tracked)
```

or run the scripts listed in each folder's README. The figures use the font **Inter** (SIL Open Font License,
https://rsms.me/inter/); without it matplotlib falls back to its default font.

## Reproduce data/ from the trajectories

The Zenodo record holds the dry, wrapped trajectories (protein, GTP and Mg²⁺; protein whole and centred; 1 ns
frames) written by `kras_hmr_prep.py`, and the MoCHI refit of the DMS data (script, input and
`kras_coupling_edges_full_canonical.csv`). Place the trajectories as follows and copy the scripts of `analysis/`
next to the two folders:

```
work/
  wco_hmr/                    namd_3_wrapped.pdb    namd_3_wrapped.dcd      (HMR, 3 × 10 µs)
                              namd_3_2_wrapped.pdb  namd_3_2_wrapped.dcd
                              namd_3_3_wrapped.pdb  namd_3_3_wrapped.dcd
  wco_vanilla_first1500ns/    vanilla_1_wrapped.pdb vanilla_1_wrapped.dcd   (no HMR, 2 fs, 3 × 1.5 µs)
                              vanilla_2_wrapped.pdb ... vanilla_3_wrapped.dcd
  kras_coupling_edges_full_canonical.csv
```

Run from `work/`. The core pipeline (all figures depend on it):

```bash
python kras_hmr_prep.py --from-wrapped wco_hmr/namd_3 wco_hmr/namd_3_2 wco_hmr/namd_3_3   # χ angles
python kras_hmr_sasa.py               # per-residue SASA (FreeSASA)
python kras_hmr_hbond.py              # residue-pair H-bond signals (MDAnalysis)
python kras_hmr_wco.py                # χ coherence (RV coefficient)
python kras_hmr_sasa_wco.py --min-seqsep 3
python kras_hmr_hbond_wco.py --dms kras_coupling_edges_full_canonical.csv
```

`--from-wrapped` computes the χ angles from the deposited files; they are identical to those computed from the
raw production trajectories, because the protein is whole in the wrapped files. For the runs without HMR
(Figure S6):

```bash
python kras_hmr_prep.py --from-wrapped wco_vanilla_first1500ns/vanilla_1 wco_vanilla_first1500ns/vanilla_2 wco_vanilla_first1500ns/vanilla_3
python kras_vanilla_check.py all
```

Time-shifted controls are not deposited. They are regenerated deterministically from the signals with the seeds
in `kras_hmr_coherence_null.py` (one seed per control set, replica and observable), and their summaries are in
`Supplementary_information/FigureS1_time_shifted_controls/data`, `Supplementary_information/FigureS2_controls_and_sensitivity/data` and `Supplementary_information/Tables_S1_S3/data`.

Then run the analysis named in each folder's README (controls, sensitivity, period bands, averages, phase, joint
states, pair signals, WLCC, threshold scan). The evolutionary couplings (Figure 5) come from the EVcouplings server
(https://v2.evcouplings.org, default settings); its output is included in `Figure5_DMS_and_evolutionary_couplings/data`. Most scripts have `--help` and `--selftest`. The control runs
(200 time-shifted controls per scheme) are long; the wrappers accept `--jobs N --threads 1`.

## Requirements

Python ≥ 3.10 with numpy, scipy, pandas, matplotlib, networkx and Pillow for the figures; MDAnalysis and
freesasa in addition for the analysis; pycwt only for `pycwt_check.py`. See `requirements.txt`.

## License

Code: MIT License (see `LICENSE`). Data in the `data/` folders: CC0 1.0 Universal (public domain dedication,
https://creativecommons.org/publicdomain/zero/1.0/). Both allow any use, including commercial use and
modification. Please cite the article if you use this material.
