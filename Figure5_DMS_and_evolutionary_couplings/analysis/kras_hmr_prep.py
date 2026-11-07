#!/usr/bin/env python
"""
kras_hmr_prep.py  --  Stage 1 for the HMR wavelet analysis.

Per replica (namd_3, namd_3_2, namd_3_3):
  1. load PSF + step5_production{,_1.._N}.dcd in run order
  2. wrap: make protein/GTP/Mg whole, center protein in box,
     wrap non-protein selection by fragment
  3. write dry wrapped trajectory (DCD + PDB topology)
  4. compute side-chain chi1..chi4 per frame -> npz

Run from the working directory that holds KRAS_unbound_mg_gtp_1/{namd_3,namd_3_2,namd_3_3}/
(each with step3_input_hmr.psf and step5_production*.dcd; see the top-level README)

  python kras_hmr_prep.py                     # defaults
  python kras_hmr_prep.py --stride 1 --nproc 3
  python kras_hmr_prep.py --reps namd_3 --dry-run   # list DCDs/frames only
  python kras_hmr_prep.py --from-wrapped wco_hmr/namd_3 wco_hmr/namd_3_2 wco_hmr/namd_3_3
                                              # chi angles from the deposited wrapped PDB + DCD

Residue numbering: PSF resid is kept as-is (resid 0 = leading GLY;
PSF resid = canonical KRAS numbering for 1-167).
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import MDAnalysis as mda
from MDAnalysis import transformations as trans
from MDAnalysis.lib.distances import calc_dihedrals, distance_array

# ---------------------------------------------------------------- chi table
# CHARMM atom names; alternatives listed where naming differs (ILE CD/CD1).
CHI = {
    "ARG": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")],
    "ASN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "ASP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "CYS": [("N", "CA", "CB", "SG")],
    "GLN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "OE1")],
    "GLU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "OE1")],
    "HIS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")],
    "ILE": [("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", ("CD", "CD1"))],
    "LEU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "LYS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")],
    "MET": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"),
            ("CB", "CG", "SD", "CE")],
    "PHE": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "PRO": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")],
    "SER": [("N", "CA", "CB", "OG")],
    "THR": [("N", "CA", "CB", "OG1")],
    "TRP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "TYR": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "VAL": [("N", "CA", "CB", "CG1")],
}
RESNAME_ALIAS = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS",
                 "HIE": "HIS", "HIP": "HIS"}


def build_chi_index(protein):
    """Return (idx[n,4], labels[list of (resid, resname, chi_k)])."""
    idx, labels = [], []
    for res in protein.residues:
        rn = RESNAME_ALIAS.get(res.resname, res.resname)
        for k, quad in enumerate(CHI.get(rn, []), start=1):
            ids = []
            for name in quad:
                names = name if isinstance(name, tuple) else (name,)
                sel = [a for a in res.atoms if a.name in names]
                if len(sel) != 1:
                    ids = None
                    break
                ids.append(sel[0].index)
            if ids is None:
                print(f"  [warn] missing atoms for {res.resname}{res.resid} "
                      f"chi{k}; skipped", file=sys.stderr)
                continue
            idx.append(ids)
            labels.append((int(res.resid), res.resname, k))
    return np.asarray(idx, dtype=np.int64), labels


# ---------------------------------------------------------------- DCD order
def dcd_key(path):
    m = re.search(r"step5_production(?:_(\d+))?\.dcd$", os.path.basename(path))
    return -1 if m.group(1) is None else int(m.group(1))


def find_dcds(repdir):
    files = [f for f in glob.glob(os.path.join(repdir, "step5_production*.dcd"))
             if re.search(r"step5_production(_\d+)?\.dcd$", f)]
    return sorted(files, key=dcd_key)


# ---------------------------------------------------------------- per replica
def process_replica(args_tuple):
    (rep, repdir, psf, outdir, keep_sel, stride, dt_ns, dry_run,
     write_traj, allow_gaps) = args_tuple
    t0 = time.time()
    dcds = find_dcds(repdir)
    if not dcds:
        return {"rep": rep, "error": f"no step5_production*.dcd in {repdir}"}

    # per-segment frame counts (also sanity print of header dt)
    seg, edge_xyz = [], []
    for d in dcds:
        u1 = mda.Universe(psf, d)
        hdr = getattr(u1.trajectory, "_file", None)
        hdr = hdr.header if hdr is not None and hasattr(hdr, "header") else {}
        seg.append({"dcd": os.path.basename(d), "n_frames": u1.trajectory.n_frames,
                    "header_dt_ps": float(u1.trajectory.dt),
                    "istart": int(hdr.get("istart", -1)), "nsavc": int(hdr.get("nsavc", -1))})
        ca = u1.select_atoms("protein and name CA")
        u1.trajectory[0]
        first = ca.positions.copy()
        u1.trajectory[-1]
        last = ca.positions.copy()
        edge_xyz.append((first, last))
    n_total = sum(s["n_frames"] for s in seg)
    # --- segment continuity (DCD header step counters + duplicated restart frames)
    cont = []
    for k in range(1, len(seg)):
        p_, q_ = seg[k - 1], seg[k]
        step_ok = None
        if p_["istart"] > 0 and q_["istart"] > 0 and p_["nsavc"] > 0:
            step_ok = (q_["istart"] - p_["istart"]) == p_["n_frames"] * p_["nsavc"]
        jump = float(np.sqrt(((edge_xyz[k][0] - edge_xyz[k - 1][1]) ** 2).sum(1).mean()))
        cont.append(dict(boundary=f"{p_['dcd']}->{q_['dcd']}", istart_prev=p_["istart"],
                         istart_next=q_["istart"], step_continuous=step_ok,
                         ca_rmsd_across_A=round(jump, 3)))
    brk = [c for c in cont if c["step_continuous"] is False or c["ca_rmsd_across_A"] < 1e-3]
    if brk and not allow_gaps:
        return {"rep": rep, "error": "segment discontinuity or duplicated frame: "
                + "; ".join(f"{c['boundary']} (step_ok={c['step_continuous']}, "
                            f"CA-RMSD {c['ca_rmsd_across_A']} A)" for c in brk)
                + "  [use --allow-gaps to override]"}
    # frame spacing check against the DCD headers (MDAnalysis dt in ps)
    bad = [x for x in seg if abs(x["header_dt_ps"] - dt_ns * 1000.0) > 1e-3 * dt_ns * 1000.0]
    if bad:
        return {"rep": rep, "error": f"DCD header dt {[b['header_dt_ps'] for b in bad]} ps "
                                     f"!= --dt-ns {dt_ns} ns ({[b['dcd'] for b in bad]})"}
    info = {"rep": rep, "psf": psf, "segments": seg, "continuity": cont,
            "n_frames_raw": n_total, "dt_ns_assumed": dt_ns, "stride": stride}
    if dry_run:
        return info

    u = mda.Universe(psf, dcds)
    protein = u.select_atoms("protein")
    keep = u.select_atoms(keep_sel)
    if protein.n_atoms == 0:
        return {"rep": rep, "error": "empty protein selection"}

    # --- wrapping workflow
    to_whole = keep  # make protein + ligand whole (needs PSF bonds)
    nonprot = keep - protein
    if u.dimensions is None:
        print(f"  [{rep}] WARNING: no box in DCD; wrapping skipped", flush=True)
        info["wrapped"] = False
    else:
        wf = [trans.unwrap(to_whole), trans.center_in_box(protein, center="mass")]
        if nonprot.n_atoms:
            wf.append(trans.wrap(nonprot, compound="fragments"))
        u.trajectory.add_transformations(*wf)
        info["wrapped"] = True

    idx, labels = build_chi_index(protein)
    a, b, c, d = (idx[:, i] for i in range(4))

    frames = range(0, n_total, stride)
    nF = len(frames)
    ang = np.empty((nF, len(labels)), dtype=np.float32)

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"{rep}")
    writer = None
    if write_traj:
        keep.write(f"{stem}_wrapped.pdb", frames=[0])
        writer = mda.Writer(f"{stem}_wrapped.dcd", keep.n_atoms,
                            dt=1000.0 * dt_ns * stride)   # ps between written frames

    # ligand placement check (same periodic image as the binding site)
    gtp = u.select_atoms("resname GTP and not name H*")
    gtpO = u.select_atoms("resname GTP and name O*")
    mg = u.select_atoms("resname MG")
    pheavy = u.select_atoms("protein and not name H*")
    d_gtp = np.full(nF, np.nan, np.float32); d_mg = np.full(nF, np.nan, np.float32)

    t_loop = time.time()
    for i, ts in enumerate(u.trajectory[::stride]):
        x = u.atoms.positions
        ang[i] = np.degrees(calc_dihedrals(x[a], x[b], x[c], x[d]))
        if gtp.n_atoms:
            d_gtp[i] = distance_array(gtp.positions, pheavy.positions).min()
        if mg.n_atoms and gtpO.n_atoms:
            d_mg[i] = distance_array(mg.positions, gtpO.positions).min()
        if writer is not None:
            writer.write(keep)
        if i and i % 200 == 0:
            rate = i / (time.time() - t_loop)
            print(f"  [{rep}] {i}/{nF}  {rate:.1f} fr/s  "
                  f"ETA {(nF - i) / rate / 60:.1f} min", flush=True)
    if writer is not None:
        writer.close()
    if not np.isfinite(ang).all():
        return {"rep": rep, "error": f"non-finite chi angles: {int((~np.isfinite(ang)).sum())} values"}
    lig = dict(gtp_protein_min_A_max=float(np.nanmax(d_gtp)) if gtp.n_atoms else None,
               mg_gtpO_min_A_max=float(np.nanmax(d_mg)) if mg.n_atoms else None)
    info["ligand_check"] = lig
    if (lig["gtp_protein_min_A_max"] or 0) > 4.5 or (lig["mg_gtpO_min_A_max"] or 0) > 3.0:
        info["ligand_warning"] = ("GTP or Mg left its binding site / wrong image in some frames: "
                                  f"{lig}; check before SASA/contact analyses")
        print(f"  [{rep}] WARNING: {info['ligand_warning']}", flush=True)

    # segment start index in strided frame space
    starts, cum = [], 0
    for s in seg:
        starts.append(int(np.ceil(cum / stride)))
        cum += s["n_frames"]

    np.savez_compressed(
        f"{stem}_chi.npz",
        angles=ang,                                   # (n_frames, n_dih) deg
        resid=np.array([l[0] for l in labels], np.int32),
        resname=np.array([l[1] for l in labels]),
        chi=np.array([l[2] for l in labels], np.int8),
        time_ns=np.arange(nF, dtype=np.float64) * dt_ns * stride,
        segment_start=np.array(starts, np.int64),
        segment_name=np.array([s["dcd"] for s in seg]),
        gtp_protein_min_A=d_gtp, mg_gtpO_min_A=d_mg,
    )
    info.update(n_frames_out=nF, n_dihedrals=len(labels),
                wall_s=round(time.time() - t0, 1))
    return info


# ---------------------------------------------------------------- from deposited wrapped files
def chi_from_wrapped(stem, dt_ns=1.0):
    """chi angles from {stem}_wrapped.pdb + {stem}_wrapped.dcd (the dry, wrapped trajectories deposited
    on Zenodo). Same chi table, atom order and output fields as process_replica; the dihedrals do not
    depend on wrapping because the protein is whole in these files."""
    t0 = time.time()
    pdb, dcd = f"{stem}_wrapped.pdb", f"{stem}_wrapped.dcd"
    for f in (pdb, dcd):
        if not os.path.isfile(f):
            return {"rep": os.path.basename(stem), "error": f"not found: {f}"}
    u = mda.Universe(pdb, dcd)
    protein = u.select_atoms("protein")
    idx, labels = build_chi_index(protein)
    a, b, c, d = (idx[:, i] for i in range(4))
    nF = u.trajectory.n_frames
    ang = np.empty((nF, len(labels)), dtype=np.float32)
    gtp = u.select_atoms("resname GTP and not name H*"); gtpO = u.select_atoms("resname GTP and name O*")
    mg = u.select_atoms("resname MG"); pheavy = u.select_atoms("protein and not name H*")
    d_gtp = np.full(nF, np.nan, np.float32); d_mg = np.full(nF, np.nan, np.float32)
    for i, ts in enumerate(u.trajectory):
        x = u.atoms.positions
        ang[i] = np.degrees(calc_dihedrals(x[a], x[b], x[c], x[d]))
        if gtp.n_atoms:
            d_gtp[i] = distance_array(gtp.positions, pheavy.positions).min()
        if mg.n_atoms and gtpO.n_atoms:
            d_mg[i] = distance_array(mg.positions, gtpO.positions).min()
    np.savez_compressed(
        f"{stem}_chi.npz", angles=ang,
        resid=np.array([l[0] for l in labels], np.int32),
        resname=np.array([l[1] for l in labels]),
        chi=np.array([l[2] for l in labels], np.int8),
        time_ns=np.arange(nF, dtype=np.float64) * dt_ns,
        segment_start=np.array([0], np.int64),
        segment_name=np.array([os.path.basename(dcd)]),
        gtp_protein_min_A=d_gtp, mg_gtpO_min_A=d_mg)
    return {"rep": os.path.basename(stem), "n_frames_out": nF, "n_dihedrals": len(labels),
            "wall_s": round(time.time() - t0, 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".", help="working dir (benchmark_phys)")
    p.add_argument("--system", default="KRAS_unbound_mg_gtp_1")
    p.add_argument("--reps", nargs="+", default=["namd_3", "namd_3_2", "namd_3_3"])
    p.add_argument("--psf", default="step3_input_hmr.psf",
                   help="PSF name inside each replica dir")
    p.add_argument("--keep", default="protein or resname GTP MG",
                   help="atoms kept in wrapped output (wrapped as fragments)")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--dt-ns", type=float, default=1.0,
                   help="time between saved frames (HMR: 250000 x 4 fs = 1 ns)")
    p.add_argument("--outdir", default="wco_hmr")
    p.add_argument("--nproc", type=int, default=3)
    p.add_argument("--no-traj", action="store_true", help="skip wrapped DCD output")
    p.add_argument("--allow-gaps", action="store_true",
                   help="do not abort on segment step-counter gaps or duplicated frames")
    p.add_argument("--dry-run", action="store_true",
                   help="only list DCD order and frame counts")
    p.add_argument("--from-wrapped", nargs="+", metavar="STEM",
                   help="compute only the chi angles from STEM_wrapped.pdb/.dcd (the deposited dry "
                        "trajectories), e.g. --from-wrapped wco_hmr/namd_3 wco_hmr/namd_3_2 wco_hmr/namd_3_3")
    args = p.parse_args()
    if args.from_wrapped:
        with ProcessPoolExecutor(max_workers=min(args.nproc, len(args.from_wrapped))) as ex:
            for r in ex.map(chi_from_wrapped, args.from_wrapped, [args.dt_ns] * len(args.from_wrapped)):
                print(r)
        return

    root = os.path.abspath(args.root)
    outdir = os.path.join(root, args.outdir)
    jobs = []
    for rep in args.reps:
        repdir = os.path.join(root, args.system, rep)
        psf = os.path.join(repdir, args.psf)
        if not os.path.isfile(psf):
            sys.exit(f"PSF not found: {psf}")
        jobs.append((rep, repdir, psf, outdir, args.keep, args.stride,
                     args.dt_ns, args.dry_run, not args.no_traj, args.allow_gaps))

    with ProcessPoolExecutor(max_workers=min(args.nproc, len(jobs))) as ex:
        results = list(ex.map(process_replica, jobs))

    for r in results:
        if "error" in r:
            print(f"[{r['rep']}] ERROR: {r['error']}")
            continue
        print(f"\n[{r['rep']}]  raw frames {r['n_frames_raw']} "
              f"(= {r['n_frames_raw'] * args.dt_ns / 1000:.2f} us at "
              f"{args.dt_ns} ns/frame)")
        for s in r["segments"]:
            print(f"   {s['dcd']:28s} {s['n_frames']:7d} frames  "
                  f"header dt {s['header_dt_ps']:.1f} ps  istart {s['istart']}  nsavc {s['nsavc']}")
        for c in r.get("continuity", []):
            print(f"   {c['boundary']:52s} step_ok={c['step_continuous']}  "
                  f"CA-RMSD across {c['ca_rmsd_across_A']} A")
        if "ligand_check" in r:
            print(f"   ligand check (max over frames): {r['ligand_check']}")
        if not args.dry_run:
            print(f"   -> {r['n_frames_out']} frames, {r['n_dihedrals']} chi, "
                  f"{r['wall_s']} s")

    if not args.dry_run:
        with open(os.path.join(outdir, "prep_manifest.json"), "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
