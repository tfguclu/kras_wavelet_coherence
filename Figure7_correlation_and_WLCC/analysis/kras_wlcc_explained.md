# kras_wlcc.py — windowed lagged cross-correlation, explained

## 1. What the script is for

The manuscript detects coordination with band wavelet coherence: two signals form a coherent pair when their coherence, averaged over periods of 80–120 ns, stays ≥ 0.75 for ≥ 500 ns. `kras_wlcc.py` asks the same question in the time domain. It correlates two signals inside sliding windows, shifts one signal against the other to find the best lag, and calls an episode when the correlation stays high for ≥ 500 ns.

It runs in two modes:

| Mode | Signals | Purpose |
|---|---|---|
| `band` | band-passed to 80–120 ns | Check that the coherence results reappear in the time domain, and read each pair's lag directly in ns. |
| `broad` | unfiltered, linearly detrended in each window | Look for coordination and lags that the 80–120 ns band does not cover. This is where new information can appear. |

It is applied to four sets of residue pairs, called units:

- every coherent pair of the baseline run;
- the DMS couplings;
- distance-matched non-DMS pairs as a background for the DMS couplings;
- random signal pairs used only to set the threshold.

The script imports nothing from the pipeline; it reads only the pipeline's output files.

## 2. Inputs

| Input | Content |
|---|---|
| `wco_hmr/{rep}_chi.npz` | `angles` (T × n, degrees), `resid`, `resname`, `chi`, `time_ns` |
| `wco_hmr/{rep}_sasa.npz` | `sasa` (T × residues), `resid`, `time_ns` |
| `wco_hmr/{rep}_hbond.npz` | `hb` (T × residue pairs, 0/1), `res_i`, `res_j`, `time_ns` |
| `--coh-chi` | `residue_pairs.csv` of the baseline χ run (`resid_i`, `resid_j`, `n_reps_any`) |
| `--coh-sasa` | `edges.csv` of the baseline SASA run (`resid_i`, `resid_j`, `n_reps_any`) |
| `--coh-hb` | `hyperedges.csv` of the baseline H-bond run (`hb1`, `hb2`, `n_reps`) |
| `--dms` | `kras_coupling_edges_full_canonical.csv` |
| `--pdb` | structure for Cα distances; default `wco_hmr/namd_3_2_wrapped.pdb`, the file the DMS null used |

The signals are prepared exactly as in the pipeline:
- χ angles become (cos χ, sin χ), with Asp χ2, Glu χ3, Phe χ2 and Tyr χ2 doubled.
- SASA residues with SD < 1 Å² in any replica are not testable.
- An H-bond signal is used in a replica if its occupancy is 5–95% and it forms at least 10 times there.

## 3. The statistic

### 3.1 One window, one lag

Take signals x and y and a window of w = 500 frames (500 ns) starting at frame a. For a lag τ, the script correlates x(t) for t in [a, a + w) with y(t + τ). It uses the Pearson correlation r(τ):
- In band mode, each window is mean-centred.
- In broad mode, each window is linearly detrended, so slow drifts inside the window do not count as coordination.

For χ, each angle is a two-component signal. The script then uses the RV coefficient of the two 2 × 2 covariance matrices:

  q = Σₖₗ cov(xₖ, yₗ)² / √( Σₖₗ cov(xₖ, xₗ)² · Σₖₗ cov(yₖ, yₗ)² )

This is the statistic that Equation 1 smooths in the wavelet domain. For one-dimensional signals it reduces to q = r². It does not change when an angle is offset by a constant.

### 3.2 Best lag

For every window, the script tries every lag in the range and keeps three things at the lag with the largest q (equivalently the largest |r|):
- q;
- the lag;
- for SASA and H-bonds, the sign of r.

The sign is ignored for detection, as in coherence, so a pair that changes in opposite directions counts. The sign is reported, and it separates in-phase (+) from antiphase (−) pairs.

Convention: a positive lag means the first signal leads, the same convention as the phase in Fig. S2. The order of the two signals is the order in the unit (lower residue first; for an H-bond unit, the first bond as listed).

When two lags give equal q, the smaller |lag| wins, because lags are tried in the order 0, +1, −1, +2, …

### 3.3 Computing it fast

Every window sum comes from cumulative sums, so the cost per lag is linear in the trajectory length. With detrending, the covariance of the residuals in a window is:

  cov = Σxy − Σx·Σy / w − Σx̃·Σỹ / Stt,  with Σz̃ = Σ(t − t̄)z and Stt = w(w² − 1)/12

This works because the lagged window of y has the same centred time axis as the window of x. The cross products for each lag are summed in 10-frame blocks, the step between windows, before the cumulative sum. That is exact because every window starts on a block boundary, and it makes the run about 3–4 times faster.

The self-test compares the result with an explicit regression for every window and lag, for scalar signals and for χ, in both modes. The largest difference was 3 × 10⁻⁸, which is float32 rounding.

### 3.4 From windows to episodes

The script slides the window in 10 ns steps over the same grid and the same analysed time as the coherence: window centres from 730 to 9270 ns, after excluding 722 ns at each end. This gives a series q(t), one value per 10 ns.

An episode is a run of centres with q ≥ θ lasting ≥ 500 ns, i.e. ≥ 50 consecutive centres. This is the same rule as a coherence episode.

The script also stores a threshold-free number per signal pair and replica, the sustained level S. S is the largest value that q stays at or above for 500 ns, i.e. the maximum over all 50-centre runs of the run minimum. An episode exists at threshold θ exactly when S ≥ θ. This lets `summarize` apply any threshold without recomputing anything.

### 3.5 Signal pairs and units

A unit is what gets a yes/no per replica:

| Unit | Signal pairs tested | Detected in a replica when |
|---|---|---|
| χ residue pair | every χ-angle pair of the two residues | any angle pair has an episode (as in the coherence analysis) |
| SASA residue pair | one pair (residues ≥ 3 apart in sequence) | it has an episode |
| coherent H-bond pair | its own two H-bond signals | it has an episode |
| DMS or background H-bond pair (i, j) | every pair of used H-bond signals with one bond containing i and the other containing j | any of them has an episode |

Taking the maximum over a unit's signal pairs works the same way in the controls, so units with many signal pairs are judged against their own baseline.

## 4. Parameters and where they come from

| Parameter | Value | Source |
|---|---|---|
| band (band mode) | 80–120 ns, zero-phase Butterworth, order 4 | the baseline band. Zero phase is essential: a causal filter would itself shift the lag. |
| window | 500 ns | The coherence smooths in time with a Gaussian of SD 2s. At the band centre (s = 100/1.033 ns) its equivalent rectangular width is √(2π)·2s = 485 ns. 500 ns is also the minimum episode (five periods). |
| step | 10 ns | the time grid of the coherence output, so episodes can be compared point by point |
| lag range, band mode | ±25 ns | Because \|r\| is used, every phase maps to one sign and one lag within ±P/4. At P = 100 ns that is ±25 ns. A wider range finds the same relation again, half a period away, with the opposite sign. |
| lag range, broad mode | ±50 ns | a choice (half a band-centre period); raise it with `--max-lag-ns` |
| lag step | 1 ns | the frame spacing; 3.6° of phase at 100 ns |
| minimum episode | 500 ns | same rule as coherence |
| excluded ends | 722 ns | same analysed time as coherence; windows and lags may reach into the excluded ends, which hold data |
| H-bond gate | a window counts only if both signals' window variance ≥ 5% of their median | the analogue of the coherence rule that band power must be ≥ 5% of its median; elsewhere q = 0 |
| control offsets | seed 7, sets 1…N, ≥ 1 µs | identical to the coherence controls (Section 6) |

## 5. The threshold

### 5.1 Why 0.75 cannot simply be reused

For band-limited signals, q at the best lag approximates the magnitude-squared coherence. So θ = 0.75 (peak |r| ≥ 0.866) was my first derivation. The self-test on your real example pairs shows it accepts far too much. It used `signals.npz`: S17–Y32, D69–R102 and Q99–R102 in all observables and replicas, 147 signal-pair series.

- **The time course matches.** WLCC q(t) followed band coherence R̄(t). The correlation between the two series was 0.70 for χ, 0.74 for H-bonds and 0.81 for SASA.
- **Every coherence episode was found.** All 18 signal-pair series with a coherence episode also had a WLCC episode at q ≥ 0.75. WLCC covered 87% of the coherence episode time. For the R102 bridge, coherence gave 5.6–7.1 µs and WLCC 6.4–7.6 µs per replica.
- **But 27 more series had WLCC episodes without any coherence episode.** Among series without a coherence episode, S reached 0.83 at the 95th percentile and 0.91 at most.

[My interpretation of the reason]
- A 500 ns window of 80–120 ns signals contains only about 4 independent values (2 × bandwidth × window = 2 × 1/240 ns⁻¹ × 500 ns).
- Taking the best of 51 lags raises q further.
- Coherence also averages over 0.6 octave of scales and uses a Gaussian rather than a boxcar window.

### 5.2 What the script does instead

θ is calibrated separately for each observable so that WLCC gives the same rate of episodes per signal pair in the controls as coherence did at 0.75. The target rates come from Fig. S6:
- χ-angle pairs: 0.091%;
- SASA pairs: 0.027%;
- H-bond signal pairs: 0.059% per control replica.

The calibration uses `--calib` random signal pairs per observable (default 2000) in every control set. θ is the (1 − rate) quantile of their sustained levels.

With 2000 pairs × 3 replicas × 20 sets there are 120,000 values per observable, so about 109, 32 and 71 exceed θ. That is enough for a stable estimate. With fewer values, `summarize` stops and says how many it needs.

The derived 0.75 and the calibrated θ are both written to `thresholds.csv`. `--theta-source derived` or `--theta X` override the calibration for comparisons.

## 6. Controls

The control analyses follow the manuscript's design (Sections 2.1 and 5.4), with the same offsets, rules and counts. They run only where they are needed.

| Analysis | Implementation | As in the manuscript |
|---|---|---|
| **Original controls** (selection) | One offset per signal, drawn in [1 µs, T − 1 µs], from `SeedSequence([7, set, crc32(replica), observable index])`, then `np.roll`; sets 1–200 | `kras_hmr_coherence_null.py`, scheme circshift-v2. Every signal is shifted by the same amount as in your coherence controls. |
| **Per-pair check** | k = the number of sets in which the unit reaches at least its real number of replicas; p = (1 + k)/(1 + sets). A unit in ≥ 2 replicas with k ≥ 1 is set aside (`set_aside`). | the per-pair check that set aside the four χ pairs |
| **Screened-set counts** | for the DMS units, the number in ≥ 1, ≥ 2 and 3 replicas, real against each control; also how many sets reach the real ≥ 2 count | the network counts of Section 2.1 and Fig. S1a |
| **≥ 1 µs relative shift** | from the same runs, counting only signal pairs whose two offsets are ≥ 1 µs apart (circular distance). Per pair: k as above. Counts are rescaled by the fraction of signal pairs passing. | `kras_hmr_stats.py offset` |
| **Seam masking** | from the same runs: ±722 ns around each signal's seam (where `np.roll` joins its end to its start) is set to 0 in the control series, and the same zones in the real series. A control counts when it reaches max(real masked replicas, 2). | `kras_hmr_controls_review.py seam` |
| **Per-residue χ shift** | separate controls with one offset per residue for all its χ angles, from `SeedSequence([7, set, crc32(replica), 99])`; sets 1–200; χ only | `kras_hmr_controls_review.py reswise` |

As in the manuscript, pairs are set aside on the original controls only. The other three test how robust the result is.

There is no smoothing variant, because WLCC has no smoothing parameter. Its counterparts would be the window length and the lag range, which can be changed with `--window-ns` and `--max-lag-ns` into another `--out` folder.

When the controls run:
1. `band` and `broad` run 20 original sets for the calibration pairs only, because the threshold needs them. This mirrors the 20 controls used for the band and threshold scans.
2. `confirm` runs only when WLCC finds a DMS unit in ≥ 2 replicas. It then runs all of the analyses above, with 200 sets each, for every DMS unit. All DMS units form the screened set, so that the counts can be compared as in Section 2.1.

Coherent pairs already passed the coherence controls. The background pairs are a reference, not results, so neither needs its own controls.

Two differences [flagged]:
- **Which H-bond signals are used.** This is taken from the real data. Your pipeline re-applied the occupancy/transition filter to each shifted copy, where the joint can change a transition count by one.
- **Mask edges.** Seam zones mask whole window centres on the 10 ns grid, rather than cutting episode intervals at exact times. The edges therefore differ by at most 10 ns.

Results for a control set are stored per signal pair in `controls/{original,reswise}/s{k}.npz`. Later runs add signal pairs to these files rather than overwriting them.

## 7. DMS couplings against a distance-matched background

DMS units are the couplings kept by `kras_overlap_null.py` (|ε| > RT and 95% width < 1 kcal/mol). They are tested in every observable where the pair is testable in at least two replicas:
- χ: both residues have χ angles;
- SASA: both residues are testable and ≥ 3 apart in sequence;
- H-bond: at least one bond pair, as defined in 3.5.

For each DMS unit, the script draws `--n-bg` (default 5) residue pairs that:
- are not DMS couplings;
- are testable in the same observable;
- lie in the same 2 Å bin of Cα distance.

These form the background.

`dms_vs_background.csv` gives, per observable:
- the number of DMS units found in ≥ 2 replicas;
- the same count excluding units set aside by the original controls (after `confirm`);
- the background fraction.

It also gives a matched test. 10,000 times, one background pair is drawn per DMS unit, and the number in ≥ 2 replicas is counted. The output is the mean, the 2.5–97.5% range, and the fraction of draws reaching the observed DMS count.

This is the set-level question; the per-pair answer is in `units_results.csv`.

## 8. Outputs

| File | Content |
|---|---|
| `params.json` | all settings and a hash; reruns with other settings stop unless `--force` or another `--out` |
| `units.csv` | one row per unit. Columns: set, observable, residues, coherence replicas (`coh_reps`), DMS trait, distance and bin, `bg_of` (the DMS unit a background pair belongs to), and the signal-pair ids per replica. |
| `registry.npz` | the signal-pair list per observable and replica, and readable signal names |
| `real_{obs}_{rep}.npz` | per signal pair: S, and the q, lag and sign series on the 10 ns grid |
| `real_{obs}_{rep}_Q.npy` | the real q series, uncompressed, read by the controls for seam masking |
| `controls/original/s{k}.npz`, `controls/reswise/s{k}.npz` | per signal pair and replica: S, the offset distance, and for the original scheme the seam-masked control and real values |
| `controls_summary.csv` | screened-set counts, real against controls, for each control analysis |
| `thresholds.csv` | derived and calibrated θ per observable, the one used, and the calibration rate reached |
| `units_results.csv` | per unit, described below |
| `units_reproducible.txt` | unit ids in ≥ 2 replicas |
| `units_dms.txt` | the DMS units (screened set) given controls by `confirm` |
| `coherent_crosstab.csv` | coherence replicas against WLCC replicas, per observable |
| `dms_vs_background.csv` | the matched test of Section 7 |
| `report.txt` | short summary |

`units_results.csv` contains, per unit:
- S per replica, θ, testable replicas and WLCC replicas;
- per replica: coverage, episode intervals, the best signal pair, its median lag and lag SD in episodes, and the fraction of episode points with positive r;
- `lag_sign_consistent`: whether the lag signs agree across replicas, treating |lag| < 5 ns as zero (`--lag-zero-ns`);
- the control columns of Section 6: `ctrl_sets`, `ctrl_k`, `ctrl_p`, `set_aside`, `k_rel_1us`, `k_seam_masked`, `k_per_residue`, `per_residue_sets`.

## 9. How to run

Run from the analysis folder that holds `wco_hmr/`. The script finds its inputs there:

| Input | Found as |
|---|---|
| signals | `wco_hmr/{namd_3, namd_3_2, namd_3_3}_{chi, sasa, hbond}.npz` |
| χ coherent pairs | newest `wco_hmr/P80-120_thr0.75_ep500_ts2_*/residue_pairs.csv` |
| SASA coherent pairs | newest `wco_hmr/SASA_P80-120_thr0.75_ep500_ts2_*sep3*/edges.csv` |
| H-bond coherent pairs | newest `wco_hmr/HB_P80-120_thr0.75_ep500_ts2_*pf0.05*/hyperedges.csv` |
| DMS couplings | `./kras_coupling_edges_full_canonical.csv`, else `wco_hmr/kras_coupling_edges_full_canonical.csv` |
| distances | `wco_hmr/namd_3_2_wrapped.pdb` |
| self-test examples | `wco_hmr/pairsignals/signals.npz` |

These are the same patterns `kras_overlap_null.py` uses. Every run prints the files it picked; check them once.

```
python kras_wlcc.py test            # must end with SELFTEST PASSED
python kras_wlcc.py band            # -> wlcc_band/report.txt
python kras_wlcc.py broad           # -> wlcc_broad/report.txt
python kras_wlcc.py confirm broad   # the manuscript's control analyses for the DMS units, if any is in >= 2 replicas
```

- **Processes.** The script uses (number of CPUs − 2) processes and one thread per process; change this with `--jobs`.
- **Overrides.** Any input or setting can be overridden; see `python kras_wlcc.py band -h`.
- **Advanced steps.** The individual steps remain available as `run` and `summarize`.
- **Resuming.** A rerun reuses finished results, and an interrupted run continues where it stopped.
- **Changed settings.** If you change settings or inputs, the script stops instead of mixing results; use `--out` for a new folder or `--force`.

Runtime, measured here, per signal pair, per replica, per data set:
- band mode: about 5 ms for SASA or H-bond and 15 ms for a χ-angle pair;
- broad mode: about 9 ms and 25 ms (twice the lags).

[Rough estimate] Each of `band` and `broad` computes the real data plus 20 sets for 3 × 2000 calibration pairs. That should take well under an hour each on your M4 Max.

## 10. Reading the results

- **Band mode.**
  - Compare WLCC replicas with coherence replicas in `coherent_crosstab.csv`. Agreement shows that the coherent pairs are not an artefact of the wavelet machinery.
  - It is not independent evidence. Band WLCC and coherence measure nearly the same thing on the same data, so they share the same coincidental detections.
  - The lag columns give each pair's timing in ns.
  - Episode lags and coherence phases (Fig. S2) should agree. Phase φ corresponds to a lag of φ/360 × 100 ns when |φ| ≤ 90°, and to an antiphase lag of (φ ∓ 180)/360 × 100 ns otherwise.
- **Broad mode.**
  - New pairs or non-zero lags here are coordination outside the 80–120 ns band or in a single window.
  - Judge them only by recurrence across replicas and by the controls, as in the manuscript.
- **Lags.**
  - Within one window, the lag is uncertain by a few ns. In the synthetic test with noise, the median lag over one stretch was 2–5 ns across noise realizations, against a true +3 ns.
  - Read only lags whose sign is consistent across replicas and larger than about 5 ns.
  - A lag shows which signal changes first. It does not show cause.

## 11. Limitations [flagged where they are my interpretation]

- **Boxcar window.** A short, strong shared oscillation spreads over every window that contains it. In the synthetic test, a 200 ns burst of double amplitude produced a 710 ns WLCC episode. Coherence, with its Gaussian smoothing, would count only a shorter crossing. Recurrence and controls guard against this, but single long episodes from short events are possible.
- **Few independent values per window.** This is why the threshold is calibrated. It also means a single window says little, and runs of windows carry the information.
- **Broad mode depends on detrending.** Linear detrending removes slow drifts only within each window. Steps inside a window can still dominate it.
- **Local geometry.** As with coherence, passing the controls does not separate coordination caused by local geometry from coordination within a larger motion. This covers exclusive bonds of one donor, bonds sharing a residue and residues that bury each other.
- **Absolute lags.** Lags inherit the HMR/Langevin caveat on absolute timescales.
- **No network-level comparison.** Controls are run on the selected units, not on all signal pairs. Use per-unit numbers and the matched DMS test, not network sizes.

## 12. Validation results obtained here

| Test | Result |
|---|---|
| window sums against explicit regression (scalar and χ, both modes) | max difference 3 × 10⁻⁸ |
| synthetic pair, shared 100 ns oscillation, second signal trailing by 3 ns, no noise | lag +3 ns, q = 1.0000 |
| same with noise and drift | episodes over both long stretches; median lag +3 ns |
| same pair with the second signal shifted as in control set 1 | sustained level 0.83 against 0.98 for the real pair |
| your example pairs (S17–Y32, D69–R102, Q99–R102) | see Section 5.1; per-series values in `wlcc_selftest_examples.csv` |
| end-to-end run on small synthetic files in your formats (both modes; 4 control sets, then 4 more for the selected units) | ran and resumed; injected lags (−4, −5 and 0 ns) recovered in every replica, with lag SD < 1 ns |
