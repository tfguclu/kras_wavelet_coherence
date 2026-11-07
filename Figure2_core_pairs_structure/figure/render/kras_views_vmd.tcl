# VMD script: KRAS·GTP·Mg²⁺ front and back views with the core coherent residues (+ optional close-ups)
# usage:  vmd -e kras_views_vmd.tcl        (PDB in the same folder)
# then, in the VMD console:
#   overview              -> whole protein; interface residues as cartoon colour; H-bond residues, H94–L133 and K128–D132 as Cα balls
#   rotate with the mouse to the front view you want, then
#   views                 -> renders view_front.tga and view_back.tga (the back view is the front turned 180° about y)
#   back                  -> turns the molecule 180° about y (to check or adjust the back view by hand)
#   inset <name>; shot <name>  -> optional close-ups (see INSETS below)
mol new kras_unbound_mgwaters.pdb
mol delrep 0 top
display projection Orthographic
display depthcue off
display ambientocclusion on
display aoambient 0.8
display aodirect 0.4
color Display Background white
axes location off

# ---------- colours (ids 23-31 are free in the default palette)
proc hex {id r g b} { color change rgb $id [expr $r/255.] [expr $g/255.] [expr $b/255.] }
hex 23 217 217 217 ; hex 24 224 179 74 ; hex 25 127 176 105 ; hex 26 167 139 208 ; hex 27 185 207 230
hex 28 184 59 114  ; hex 29 42 157 143  ; hex 30 208 140 31 ; hex 31 59 91 146

# ---------- core residues by observable (as in Table 1)
set CHI  {1 3 52 54 126 128 132}
set SASA {8 16 80 93 94 104 133}
set HB   {69 73 88 91 92 99 102 103 105 116 144 150}
set GLY  {10 75}
# Switch II–α3 interface residues of core pairs: known contacts, shown only as a cartoon colour
set IFACE {69 73 75 99 102 103 104 105}
# balls in the overview: all H-bond residues outside the interface, one SASA pair, one χ pair
set BALL_HB   {88 91 92 116 144 150}
set BALL_SASA {94 133}
set BALL_CHI  {128 132}

# ---------- representations, each remembered by name
set ::R [dict create]
proc addrep {name style col sel {mat AOChalky}} {
    eval mol representation $style
    if {$col eq "Element"} { mol color Element } else { mol color ColorID $col }
    mol selection $sel; mol material $mat; mol addrep top
    dict set ::R $name [expr {[molinfo top get numreps] - 1}]
}
set CART "NewCartoon 0.3 50 4.1 0"
addrep cart_rest  $CART 23 "protein and not resid 10 to 17 30 to 38 59 to 76 87 to 104 $IFACE"
addrep cart_ploop $CART 24 "protein and resid 10 to 17"
addrep cart_sw1   $CART 25 "protein and resid 30 to 38"
addrep cart_sw2   $CART 26 "protein and resid 59 to 76 and not resid $IFACE"
addrep cart_a3    $CART 27 "protein and resid 87 to 104 and not resid $IFACE"
addrep cart_iface $CART 31 "protein and resid $IFACE"
addrep gtp  "Licorice 0.25 50 50" Element "resname GTP"
addrep mg   "VDW 0.6 50" 19 "resname MG"
# overview: Cα balls for the selected residues
addrep ca_chi  "VDW 0.7 50" 28 "protein and name CA and resid $BALL_CHI"
addrep ca_sasa "VDW 0.7 50" 29 "protein and name CA and resid $BALL_SASA"
addrep ca_hb   "VDW 0.7 50" 30 "protein and name CA and resid $BALL_HB"
# insets: side chains (selections are narrowed by 'inset')
addrep sc_chi  "Licorice 0.3 50 50" 28 "none"
addrep sc_sasa "Licorice 0.3 50 50" 29 "none"
addrep sc_hb   "Licorice 0.3 50 50" 30 "none"
addrep sc_gly  "VDW 0.8 50" 29 "none"
addrep hbonds  "HBonds 3.5 40 3" 30 "none"

proc only {names} {
    dict for {n i} $::R { mol showrep top $i [expr {[lsearch $names $n] >= 0}] }
}
set CARTS {cart_rest cart_ploop cart_sw1 cart_sw2 cart_a3 cart_iface}

# ---------- inset presets: residues shown, zoom
# (D69/K88/D92/R102 are in SASA and H-bond pairs; they are drawn in the H-bond colour)
set INSETS [dict create \
    interface  {res {69 73 75 99 102 103 104 105}        zoom 3.0} \
    k88        {res {88 91 92}                           zoom 3.5} \
    nucleotide {res {8 10 16 116 144 150}                zoom 2.8} \
    a3a4       {res {80 93 94 126 128 132 133}           zoom 2.6} \
    nterm      {res {1 3 52 54}                          zoom 3.2} ]

proc overview_reps {} {
    global CARTS
    only [concat $CARTS gtp mg ca_chi ca_sasa ca_hb]
}
proc overview {} {
    overview_reps
    display resetview
}

proc inset {name} {
    global CARTS INSETS CHI SASA HB GLY
    set res  [dict get $INSETS $name res]
    set zoom [dict get $INSETS $name zoom]
    set R $::R
    mol modselect [dict get $R sc_chi]  top "protein and resid $res and resid $CHI and (sidechain or name CA)"
    mol modselect [dict get $R sc_sasa] top "protein and resid $res and resid $SASA and (sidechain or name CA)"
    mol modselect [dict get $R sc_hb]   top "protein and resid $res and resid $HB and (sidechain or name CA)"
    mol modselect [dict get $R sc_gly]  top "protein and resid $res and resid $GLY and name CA"
    mol modselect [dict get $R hbonds]  top "protein and resid $res and resid $HB"
    only [concat $CARTS gtp mg sc_chi sc_sasa sc_hb sc_gly hbonds]
    # rotate about the inset residues and put them in the middle of the screen
    set s [atomselect top "protein and resid $res"]
    molinfo top set center [list [measure center $s]]
    $s delete
    translate to 0 0 0
    scale to [expr {0.025 * $zoom}]
    puts "inset $name: residues $res. Rotate with the mouse, then: shot $name"
}

proc shot {name {px 1500}} {
    display resize $px $px
    render TachyonInternal $name.tga
    puts "wrote $name.tga"
}

proc back {} { rotate y by 180 }

proc views {{px 1500}} {
    overview_reps
    shot view_front $px
    rotate y by 180
    shot view_back $px
    rotate y by 180
    puts "wrote view_front.tga and view_back.tga"
}

overview
