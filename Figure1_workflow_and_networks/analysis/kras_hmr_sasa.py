#!/usr/bin/env python
"""
kras_hmr_sasa.py -- per-residue SASA time series (FreeSASA + MDAnalysis) from the
wrapped dry trajectories written by kras_hmr_prep.py.

Input : wco_hmr/{rep}_wrapped.pdb + {rep}_wrapped.dcd  (protein + GTP + Mg)
Output: wco_hmr/{rep}_sasa.npz   (provenance stored in key "meta")
          sasa      (n_frames, n_res) float32, A^2, protein residues only
          rel_sasa  (n_frames, n_res) float32, SASA / Tien 2013 theoretical max ASA
                    (reference-normalised exposure; not guaranteed to lie in [0,1])
          resid, resname, time_ns

Method (FreeSASA conventions): heavy atoms only, ProtOr radii (implicit H),
Lee-Richards, probe 1.4 A. CHARMM names are mapped to PDB names for the
classifier (HSD/HSE/HSP->HIS, ILE CD->CD1, OT1/OT2->O/OXT). GTP and Mg are kept
as occluders with element radii (C 1.70, N 1.55, O 1.52, P 1.80, S 1.80,
Mg 1.73 A) but not reported. FreeSASA ignores PBC, so the script aborts if
GTP/Mg are not in the protein's image.

Install:  conda install -c conda-forge freesasa     (or: pip install freesasa)
Run from the working directory:
  python kras_hmr_sasa.py                         # all 3 replicas
  python kras_hmr_sasa.py --stride 2 --slices 20 --nproc 3
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import freesasa

MAXASA = dict(ALA=129, ARG=274, ASN=195, ASP=193, CYS=167, GLU=223, GLN=225, GLY=104,
              HIS=224, ILE=197, LEU=201, LYS=236, MET=224, PHE=240, PRO=159, SER=155,
              THR=172, TRP=285, TYR=263, VAL=174)
RES_ALIAS = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS"}
ATOM_ALIAS = {("ILE", "CD"): "CD1", ("*", "OT1"): "O", ("*", "OT2"): "OXT"}
ELEM_R = {"C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80, "S": 1.80, "MG": 1.73}


def radii_for(atoms):
    cls = freesasa.Classifier()
    r, fallback = np.empty(len(atoms)), set()
    for k, at in enumerate(atoms):
        rn = RES_ALIAS.get(at.resname, at.resname)
        an = ATOM_ALIAS.get((rn, at.name), ATOM_ALIAS.get(("*", at.name), at.name))
        x = cls.radius(rn, an)
        if x <= 0:
            el = "MG" if at.resname == "MG" else at.name[0]
            if el not in ELEM_R:
                raise RuntimeError(f"no radius for {at.resname} {at.name}")
            x = ELEM_R[el]
            fallback.add(at.resname if at.resname in ("GTP", "MG") else f"{at.resname}:{at.name}")
        r[k] = x
    return r, fallback


def run(args):
    rep, indir, stride, dt_ns, algo, slices, points = args
    freesasa.setVerbosity(freesasa.silent)      # also in spawned workers
    t0 = time.time()
    pdb = os.path.join(indir, f"{rep}_wrapped.pdb")
    dcd = os.path.join(indir, f"{rep}_wrapped.dcd")
    u = mda.Universe(pdb, dcd)
    heavy = u.select_atoms("(protein or resname GTP MG) and not name H*")
    prot = heavy.select_atoms("protein")
    if len(set(prot.residues.resids.tolist())) != prot.n_residues:
        raise RuntimeError(f"{rep}: duplicate residue IDs in protein selection")
    radii, fallback = radii_for(heavy)
    unexpected = {f for f in fallback if f not in ("GTP", "MG")}
    if unexpected:
        raise RuntimeError(f"{rep}: protein atoms without ProtOr radius: {sorted(unexpected)}")
    # map heavy atoms of the protein to residue columns
    resids = prot.residues.resids
    col = {rid: k for k, rid in enumerate(resids)}
    is_prot = np.isin(heavy.indices, prot.indices)
    atom_col = np.array([col[a.resid] for a in heavy if a.resname not in ("GTP", "MG")])
    resname = np.array(prot.residues.resnames).astype(str)
    maxasa = np.array([MAXASA[RES_ALIAS.get(n, n)] for n in resname], float)
    lig = u.select_atoms("resname GTP MG")
    ca = u.select_atoms("protein and name CA")

    if algo == "lr":
        par = freesasa.Parameters({"algorithm": freesasa.LeeRichards,
                                   "n-slices": slices, "probe-radius": 1.4})
    else:
        par = freesasa.Parameters({"algorithm": freesasa.ShrakeRupley,
                                   "n-points": points, "probe-radius": 1.4})
    rlist = radii.tolist()
    frames = u.trajectory[::stride]
    out = np.empty((len(frames), len(resids)), np.float32)
    for f, ts in enumerate(frames):
        x = heavy.positions
        if not np.isfinite(x).all():
            raise RuntimeError(f"{rep}: non-finite coordinates at frame {ts.frame}")
        if lig.n_atoms and distance_array(lig.positions, ca.positions).min() > 12.0:
            raise RuntimeError(f"{rep}: GTP/Mg > 12 A from any CA at frame {ts.frame}; "
                               "check wrapping")
        res = freesasa.calcCoord(x.ravel().tolist(), rlist, par)
        area = np.fromiter((res.atomArea(i) for i in range(heavy.n_atoms)), float,
                           count=heavy.n_atoms)
        out[f] = np.bincount(atom_col, weights=area[is_prot], minlength=len(resids))
        if f and f % 500 == 0:
            el = time.time() - t0
            print(f"  [{rep}] {f}/{len(frames)}  {f / el:.1f} fr/s  "
                  f"ETA {(len(frames) - f) / (f / el) / 60:.1f} min", flush=True)
    if not np.isfinite(out).all():
        raise RuntimeError(f"{rep}: non-finite SASA")
    meta = dict(stride=stride, dt_ns_input=dt_ns, probe_A=1.4, program="freesasa",
                algorithm="LeeRichards" if algo == "lr" else "ShrakeRupley",
                slices=slices if algo == "lr" else None,
                sphere_points=points if algo == "sr" else None,
                radii="ProtOr (heavy atoms, implicit H); GTP/Mg element radii",
                selection="SASA on protein+GTP+Mg heavy atoms; protein residues reported",
                rel_sasa="SASA / Tien 2013 theoretical max (reference-normalised)",
                mdanalysis=mda.__version__)
    np.savez_compressed(os.path.join(indir, f"{rep}_sasa.npz"),
                        sasa=out, rel_sasa=(out / maxasa).astype(np.float32),
                        resid=resids.astype(np.int32), resname=resname,
                        time_ns=np.arange(len(out)) * dt_ns * stride,
                        meta=np.array(json.dumps(meta)))
    return rep, f"{out.shape[0]} frames x {out.shape[1]} residues, {time.time() - t0:.0f} s"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="wco_hmr")
    p.add_argument("--reps", nargs="+", default=["namd_3", "namd_3_2", "namd_3_3"])
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--dt-ns", type=float, default=1.0, help="time between saved frames")
    p.add_argument("--algorithm", choices=["lr", "sr"], default="lr",
                   help="lr = Lee-Richards (FreeSASA default), sr = Shrake-Rupley")
    p.add_argument("--slices", type=int, default=20, help="Lee-Richards slices per atom")
    p.add_argument("--points", type=int, default=100, help="Shrake-Rupley test points")
    p.add_argument("--nproc", type=int, default=3)
    a = p.parse_args()
    freesasa.setVerbosity(freesasa.silent)
    jobs = [(r, a.indir, a.stride, a.dt_ns, a.algorithm, a.slices, a.points) for r in a.reps]
    with ProcessPoolExecutor(max_workers=min(a.nproc, len(jobs))) as ex:
        for rep, msg in ex.map(run, jobs):     # exceptions propagate -> nonzero exit
            print(f"[{rep}] {msg}")


if __name__ == "__main__":
    main()
