#!/usr/bin/env python
"""
kras_hmr_hbond.py -- residue-pair hydrogen-bond time series (0/1 per frame).

Input : wco_hmr/{rep}_wrapped.pdb + {rep}_wrapped.dcd   (protein + GTP + Mg)
Output: wco_hmr/{rep}_hbond.npz
          hb        (n_frames, n_pairs) uint8  1 = residues i and j share >= 1 H-bond
          res_i, res_j            residue ids of the pair (i < j); GTP = its resid
          name_i, name_j          residue names
          n_atom_pairs            number of distinct donor-acceptor atom pairs seen
          occupancy, time_ns, meta (JSON provenance)

H-bond criterion (MDAnalysis HydrogenBondAnalysis): donor-acceptor <= 3.2 A and
D-H...A angle > 150 deg (as in the DHFR H-bond network paper; settable). Donor-hydrogen pairs are found by
distance (<= 1.2 A) because the dry PDB carries no bonds. All residues are
included (protein + GTP); intra-residue H-bonds are ignored. Water is not
available in the dry trajectory, so water-mediated H-bonds are not counted.

Donors   : any N/O/S of protein or GTP that carries a hydrogen (H* within 1.2 A)
Acceptors: all O; His NE2 (HSD) / ND1 (HSE); GTP N3, N7

  python kras_hmr_hbond.py                       # 3 replicas in parallel
  python kras_hmr_hbond.py --d-a 3.2 --angle 150 --stride 1
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis

DONORS = "(protein or resname GTP) and (name N* or name O* or name SG)"
HYDROGENS = "(protein or resname GTP) and name H*"
ACCEPTORS = ("(protein or resname GTP) and (name O* or (resname HSD and name NE2) "
             "or (resname HSE and name ND1) or (resname GTP and name N3 N7))")


def run(args):
    rep, indir, stride, dt_ns, d_a, angle = args
    t0 = time.time()
    u = mda.Universe(os.path.join(indir, f"{rep}_wrapped.pdb"),
                     os.path.join(indir, f"{rep}_wrapped.dcd"))
    if u.dimensions is None:
        raise RuntimeError(f"{rep}: no box in DCD")
    hba = HydrogenBondAnalysis(universe=u, donors_sel=DONORS, hydrogens_sel=HYDROGENS,
                               acceptors_sel=ACCEPTORS, d_h_cutoff=1.2, d_a_cutoff=d_a,
                               d_h_a_angle_cutoff=angle, update_selections=False)
    hba.run(step=stride)
    hb = hba.results.hbonds                        # frame, donor, hydrogen, acceptor, dist, angle
    nF = len(range(0, u.trajectory.n_frames, stride))
    resid = u.atoms.resids
    resname = u.atoms.resnames
    rd = resid[hb[:, 1].astype(int)]
    ra = resid[hb[:, 3].astype(int)]
    keep = rd != ra                                 # drop intra-residue
    hb, rd, ra = hb[keep], rd[keep], ra[keep]
    lo, hi = np.minimum(rd, ra), np.maximum(rd, ra)
    pairs = sorted(set(zip(lo.tolist(), hi.tolist())))
    col = {p: k for k, p in enumerate(pairs)}
    frames_run = np.arange(0, u.trajectory.n_frames, stride)
    row = np.searchsorted(frames_run, hb[:, 0].astype(int))
    M = np.zeros((nF, len(pairs)), np.uint8)
    M[row, [col[(a, b)] for a, b in zip(lo.tolist(), hi.tolist())]] = 1
    name_of = {r: n for r, n in zip(u.residues.resids, u.residues.resnames)}
    atom_pairs = {}
    for a, b, d_, h_ in zip(lo.tolist(), hi.tolist(), hb[:, 1].astype(int), hb[:, 3].astype(int)):
        atom_pairs.setdefault((a, b), set()).add((int(d_), int(h_)))
    meta = dict(stride=stride, dt_ns_input=dt_ns, d_a_cutoff_A=d_a, dha_angle_deg=angle,
                d_h_cutoff_A=1.2, donors=DONORS, hydrogens=HYDROGENS, acceptors=ACCEPTORS,
                intra_residue="excluded", water="not present (dry trajectory)",
                mdanalysis=mda.__version__)
    np.savez_compressed(
        os.path.join(indir, f"{rep}_hbond.npz"), hb=M,
        res_i=np.array([p[0] for p in pairs], np.int32), res_j=np.array([p[1] for p in pairs], np.int32),
        name_i=np.array([name_of[p[0]] for p in pairs]).astype(str),
        name_j=np.array([name_of[p[1]] for p in pairs]).astype(str),
        n_atom_pairs=np.array([len(atom_pairs[p]) for p in pairs], np.int32),
        occupancy=M.mean(axis=0).astype(np.float32),
        time_ns=np.arange(nF) * dt_ns * stride, meta=np.array(json.dumps(meta)))
    return rep, (f"{nF} frames, {len(hb)} H-bonds, {len(pairs)} residue pairs, "
                 f"{time.time() - t0:.0f} s")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="wco_hmr")
    p.add_argument("--reps", nargs="+", default=["namd_3", "namd_3_2", "namd_3_3"])
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--dt-ns", type=float, default=1.0)
    p.add_argument("--d-a", type=float, default=3.2, help="donor-acceptor cutoff (A)")
    p.add_argument("--angle", type=float, default=150.0, help="min D-H...A angle (deg)")
    p.add_argument("--nproc", type=int, default=3)
    a = p.parse_args()
    jobs = [(r, a.indir, a.stride, a.dt_ns, a.d_a, a.angle) for r in a.reps]
    with ProcessPoolExecutor(max_workers=min(a.nproc, len(jobs))) as ex:
        for rep, msg in ex.map(run, jobs):
            print(f"[{rep}] {msg}")


if __name__ == "__main__":
    main()
