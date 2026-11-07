# VMD script: KRAS·GTP·Mg starting model with the 16 core coherent pairs
# usage: vmd -e kras_core_vmd.tcl   (PDB in the same folder)
mol new kras_unbound_mgwaters.pdb
mol delrep 0 top
display projection Orthographic
display depthcue off
color Display Background white
axes location off
# custom colours (ids 23-31 are free in the default palette)
proc hex {id r g b} { color change rgb $id [expr $r/255.] [expr $g/255.] [expr $b/255.] }
hex 23 217 217 217 ; hex 24 224 179 74 ; hex 25 127 176 105 ; hex 26 167 139 208 ; hex 27 185 207 230
hex 28 184 59 114  ; hex 29 42 157 143  ; hex 30 208 140 31
proc rep {style col sel} { eval mol representation $style; mol color ColorID $col; mol selection $sel; mol material AOChalky; mol addrep top }
rep "NewCartoon 0.3 50 4.1 0" 23 "protein and not resid 10 to 17 30 to 38 59 to 76 87 to 104"
rep "NewCartoon 0.3 50 4.1 0" 24 "protein and resid 10 to 17"
rep "NewCartoon 0.3 50 4.1 0" 25 "protein and resid 30 to 38"
rep "NewCartoon 0.3 50 4.1 0" 26 "protein and resid 59 to 76"
rep "NewCartoon 0.3 50 4.1 0" 27 "protein and resid 87 to 104"
mol representation Licorice 0.25 50 50; mol color Element; mol selection "resname GTP"; mol material AOChalky; mol addrep top
mol representation VDW 0.6 50; mol color ColorID 19; mol selection "resname MG"; mol addrep top
# core residues: side chains + CA (glycines as CA sphere); H-bond colour last
rep "Licorice 0.3 50 50" 28 "protein and resid 1 3 52 54 126 128 132 and (sidechain or name CA)"
rep "Licorice 0.3 50 50" 29 "protein and resid 8 16 80 93 94 104 133 and (sidechain or name CA)"
rep "Licorice 0.3 50 50" 30 "protein and resid 69 73 88 91 92 99 102 103 105 116 144 150 and (sidechain or name CA)"
rep "VDW 0.8 50" 29 "protein and resid 10 75 and name CA"
# core H-bonds
rep "HBonds 3.5 30 4" 30 "protein and (resid 102 69 99 or resid 73 103 105 or resid 88 91 92 or resid 144 116 150)"
display resetview
# render: render TachyonInternal kras_core_structure.tga
