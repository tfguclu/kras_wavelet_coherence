# PyMOL script: KRAS·GTP·Mg starting model with the 16 core coherent pairs
load kras_unbound_mgwaters.pdb, kras
bg_color white
set ray_opaque_background, 0
set cartoon_transparency, 0
set ray_trace_mode, 1
set ray_trace_color, black
set antialias, 2
set cartoon_fancy_helices, 1
set label_size, 18
set label_color, black
set label_font_id, 7
hide everything
show cartoon, polymer
set_color c_base, [0.85,0.85,0.85]
set_color c_ploop, [0.878,0.702,0.290]
set_color c_sw1, [0.498,0.690,0.412]
set_color c_sw2, [0.655,0.545,0.816]
set_color c_a3, [0.725,0.812,0.902]
set_color c_chi, [0.722,0.231,0.447]
set_color c_sasa, [0.165,0.616,0.561]
set_color c_hb, [0.816,0.549,0.122]
color c_base, polymer
color c_ploop, resi 10-17 and polymer
color c_sw1, resi 30-38 and polymer
color c_sw2, resi 59-76 and polymer
color c_a3, resi 87-104 and polymer
show sticks, resn GTP
color grey40, resn GTP and elem C
show spheres, resn MG
set sphere_scale, 0.4, resn MG
color forest, resn MG
select chi_core, polymer and resi 1+3+52+54+126+128+132
select sasa_core, polymer and resi 8+10+16+69+75+80+88+92+93+94+102+104+133
select hb_core, polymer and resi 69+73+88+91+92+99+102+103+105+116+144+150
# order: chi, SASA, then H-bond (shared residues end H-bond coloured)
show sticks, (chi_core or sasa_core or hb_core) and (sidechain or name CA)
show spheres, polymer and resi 10+75 and name CA
set sphere_scale, 0.35, polymer and resi 10+75 and name CA
color c_chi, chi_core and elem C
color c_sasa, sasa_core and elem C
color c_hb, hb_core and elem C
util.cnc("chi_core or sasa_core or hb_core")
# core H-bonds as dashes (from model geometry)
distance hb1, resi 102 and polymer, resi 69+99 and polymer, mode=2
distance hb2, resi 73 and polymer, resi 103+105 and polymer, mode=2
distance hb3, resi 88 and polymer, resi 91+92 and polymer, mode=2
distance hb4, resi 144 and polymer, resi 116+150 and polymer, mode=2
color c_hb, hb*
hide labels, hb*
set dash_gap, 0.25
set dash_radius, 0.06
label (chi_core or sasa_core or hb_core) and name CA, "%s%s" % (one_letter[resn], resi)
orient resi 59-76+87-105
# adjust the view, then: ray 3000,2400 ; png kras_core_structure.png, dpi=400
