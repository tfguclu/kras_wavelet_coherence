#!/usr/bin/env bash
# Regenerate every figure from the data in this repository (run from the repository root).
set -e
run() { echo "== $1"; python "$1" "${@:2}"; }
d=Figure1_workflow_and_networks; run $d/figure/fig1_workflow.py; run $d/figure/fig1_networks.py; run $d/figure/compose_figure1.py
d=Figure2_core_pairs_structure
if [ -f $d/data/view_front.png ] && [ -f $d/data/view_back.png ]; then
  run $d/figure/fig2_structure_panel.py; run $d/figure/fig2_sequence_map.py; run $d/figure/compose_figure2.py
else
  run $d/figure/fig2_sequence_map.py
  echo "   Figure 2 panel a skipped: render data/view_front.png and data/view_back.png with VMD first (see $d/README.md)"
fi
run Figure3_phase_and_joint_states/figure/fig3_phase_states.py
run Figure4_local_networks/figure/fig_network_episodes.py --part A
run Figure5_DMS_and_evolutionary_couplings/figure/fig5_dms_ev.py
run Figure6_timescales/figure/fig6_timescales.py
run Figure7_correlation_and_WLCC/figure/fig7_correlation_wlcc.py
run Figure8_wavelet_coherence_method/figure/fig8_wavelet_method.py
S=Supplementary_information
run $S/FigureS1_time_shifted_controls/figure/figS1_controls.py
run $S/FigureS2_controls_and_sensitivity/figure/figS2_controls_sensitivity.py
run $S/FigureS3_K88_barcodes/figure/figS3_k88_barcodes.py
run $S/FigureS4_SwitchI_and_nucleotide_site/figure/figS4_switch_traces.py
run $S/FigureS5_local_networks_other/figure/fig_network_episodes.py --part B
run $S/FigureS6_runs_without_HMR/figure/figS6_hmr_vs_2fs.py
run $S/FigureS7_threshold_scan/figure/figS7_threshold.py
run TOC_graphic/figure/toc_graphic.py
echo "all figures written (each into its folder's figure/)"
