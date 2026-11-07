#!/usr/bin/env python
"""
kras_hmr_runcheck.py -- shared run construction and cache validation for the wrappers
(kras_hmr_coherence_null.py, kras_hmr_sensitivity.py, kras_hmr_multiband.py).

A run is defined by: observable, band (pmin, pmax), threshold, minimum episode and
time decimation; smoothing (2 x scale), scale step (1/12 octave) and the
observable-specific filters are fixed and always passed explicitly.

An output folder is reused only if
  (i)   params.json exists and every recorded setting matches (replicas, band,
        threshold, episode, tsmooth, tdec, dj, dt_ns = 1 ns, filters, H-bond geometry),
  (ii)  the required result files and per-replica metadata (valid_ns) exist, and
  (iii) its provenance sidecar (multiband_provenance.json) matches the expected
        provenance: run settings, SHA-256 of the input .npz files and, for
        surrogates, seed, minimum shift and surrogate set.
Folders without a sidecar ("legacy", made before this check existed) are accepted
only for real signals and only when the caller allows it (--accept-legacy-real).
"""
import glob, hashlib, json, math, os, subprocess, sys, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TSMOOTH, DJ = 2.0, 1 / 12
DJ_STR = repr(DJ)
SCRIPT = {"chi": "kras_hmr_wco.py", "sasa": "kras_hmr_sasa_wco.py", "hb": "kras_hmr_hbond_wco.py"}
EXTRA = {"chi": [], "sasa": ["--min-seqsep", "3", "--weak-sd", "1"],
         "hb": ["--occ-min", "0.05", "--occ-max", "0.95", "--min-trans", "10", "--pow-floor", "0.05"]}
EXTRA_CHECK = {"chi": {}, "sasa": dict(min_seqsep=3, weak_sd=1.0, keep_weak=False),
               "hb": dict(occ_min=0.05, occ_max=0.95, min_trans=10, pow_floor=0.05)}
OPTIONAL_KEYS = {"keep_weak"}
HB_SETTINGS = dict(d_a_cutoff_A=3.2, dha_angle_deg=150.0)
INPUT = {"chi": "chi", "sasa": "sasa", "hb": "hbond"}
REQUIRED = {"chi": ["residue_pairs.csv", "params.json"], "sasa": ["edges.csv", "params.json"],
            "hb": ["hyperedges.csv", "params.json"]}
PREFIX = {"chi": "", "sasa": "SASA_", "hb": "HB_"}
SIDE = "multiband_provenance.json"
_HASH = {}


def script_path(obs):
    p = os.path.join(SCRIPT_DIR, SCRIPT[obs])
    if not os.path.exists(p):
        sys.exit(f"{SCRIPT[obs]} not found next to {__file__}")
    return p


def sha256(path):
    st = os.stat(path); key = (os.path.abspath(path), st.st_size, st.st_mtime)
    if key not in _HASH:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 22), b""):
                h.update(blk)
        _HASH[key] = h.hexdigest()
    return _HASH[key]


def input_hashes(indir, obs, reps):
    return {r: sha256(os.path.join(indir, f"{r}_{INPUT[obs]}.npz")) for r in reps}


def eff_ep(s, dt_ns=1.0):
    g = dt_ns * s["tdec"]
    return math.ceil(s["ep"] / g - 1e-9) * g


def settings_args(obs, s, reps):
    return ["--pmin", f"{s['pmin']:g}", "--pmax", f"{s['pmax']:g}", "--thr", f"{s['thr']:g}",
            "--min-ep", f"{s['ep']:g}", "--tsmooth", f"{TSMOOTH:g}", "--tdec", str(s["tdec"]),
            "--dj", DJ_STR, "--reps", *reps, *EXTRA[obs]]


def cmd_for(obs, s, indir, reps):
    return [sys.executable, script_path(obs), "--indir", indir, *settings_args(obs, s, reps)]


def _pattern(obs, s, indir):
    return os.path.join(indir, f"{PREFIX[obs]}P{s['pmin']:g}-{s['pmax']:g}_thr{s['thr']:g}"
                               f"_ep{s['ep']:g}_ts{TSMOOTH:g}_*")


def params_ok(obs, s, d, reps, why):
    for f in REQUIRED[obs]:
        if not os.path.exists(os.path.join(d, f)):
            why.append(f"missing {f}"); return False
    try:
        P = json.load(open(os.path.join(d, "params.json")))
    except Exception as e:
        why.append(f"params.json unreadable ({e})"); return False
    A = P.get("args", {})
    want = dict(pmin=s["pmin"], pmax=s["pmax"], thr=s["thr"], min_ep=s["ep"], tsmooth=TSMOOTH,
                tdec=s["tdec"], dj=DJ, **EXTRA_CHECK[obs])
    for k, v in want.items():
        if k not in A:
            if k in OPTIONAL_KEYS:
                continue
            why.append(f"{k} not recorded"); return False
        av = A[k]
        if isinstance(v, bool) or isinstance(av, bool):
            if bool(av) != v:
                why.append(f"{k}={av}"); return False
        elif av is None or not np.isclose(float(av), float(v), rtol=1e-6, atol=1e-9):
            why.append(f"{k}={av} (want {v})"); return False
    if list(A.get("reps", [])) != list(reps):
        why.append(f"reps={A.get('reps')}"); return False
    dt = A.get("dt_ns")
    if dt is None or not np.isclose(float(dt), 1.0):
        why.append(f"dt_ns={dt}"); return False
    if obs == "hb":
        hs = P.get("hbond_settings", {})
        for k, v in HB_SETTINGS.items():
            if k not in hs or not np.isclose(float(hs[k]), v):
                why.append(f"hbond {k}={hs.get(k)}"); return False
    M = P.get("meta", {})
    if not all(r in M and "valid_ns" in M[r] for r in reps):
        why.append("meta lacks per-replica valid_ns"); return False
    return True


def expectation(obs, s, reps, real_hashes, surrogate=None):
    e = dict(kind="real" if surrogate is None else "surrogate",
             settings=settings_args(obs, s, reps), inputs=real_hashes)
    if surrogate is not None:
        e.update(surrogate)          # dict(set=k, seed=..., min_shift_ns=..., scheme=...)
    return e


def _sidecar(d, expect, why):
    f = os.path.join(d, SIDE)
    if not os.path.exists(f):
        why.append("no provenance sidecar"); return None
    try:
        s = json.load(open(f))
    except Exception:
        why.append("sidecar unreadable"); return False
    for k, v in expect.items():
        if s.get(k) != v:
            why.append(f"sidecar '{k}' differs"); return False
    return True


def find_valid(obs, s, indir, reps, expect, accept_legacy=False, verbose=True):
    good, legacy = [], []
    for d in sorted(glob.glob(_pattern(obs, s, indir))):
        why = []
        if not params_ok(obs, s, d, reps, why):
            if verbose: print(f"    reject {d}: {'; '.join(why)}")
            continue
        sc = _sidecar(d, expect, why)
        if sc:
            good.append(d)
        elif sc is None and accept_legacy and expect.get("kind") == "real":
            legacy.append(d)
        elif verbose:
            print(f"    reject {d}: {'; '.join(why)}")
    pick = good or legacy
    if len(pick) > 1 and verbose:
        print(f"    WARNING: {len(pick)} valid outputs; using the newest")
    if not good and legacy and verbose:
        print(f"    accepting legacy real output (params validated, no sidecar): {max(legacy, key=os.path.getmtime)}")
    return max(pick, key=os.path.getmtime) if pick else None


def run_and_tag(obs, s, indir, reps, expect, log, dry=False):
    c = cmd_for(obs, s, indir, reps)
    log.append(" ".join(c)); print("   ", " ".join(c), flush=True)
    if dry:
        return None
    t0 = time.time()
    r = subprocess.run(c)
    if r.returncode:
        sys.exit(f"failed ({r.returncode}): {' '.join(c)}")
    new = [d for d in glob.glob(_pattern(obs, s, indir))
           if os.path.exists(os.path.join(d, "params.json"))
           and os.path.getmtime(os.path.join(d, "params.json")) >= t0 - 1]
    ok = [d for d in new if params_ok(obs, s, d, reps, [])]
    if len(ok) != 1:
        sys.exit(f"could not identify the completed output of {' '.join(c)} (candidates {new})")
    with open(os.path.join(ok[0], SIDE), "w") as fh:
        json.dump(dict(expect, written=time.strftime("%Y-%m-%d %H:%M:%S")), fh, indent=1)
    print(f"    {(time.time() - t0) / 60:.1f} min -> {ok[0]}", flush=True)
    return ok[0]


def per_rep_meta(d, reps, key):
    M = json.load(open(os.path.join(d, "params.json"))).get("meta", {})
    return [float(M[r][key]) for r in reps if r in M and key in M[r]]


def plot_items(M, path, title="replicas in which the association is detected", xlabels=None):
    """heat map of 0-3 replica counts; NaN cells (not analysed) are grey and marked NA."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#ffffff", "#d4d4d4", "#8f8f8f", "#2b2b2b"]); cmap.set_bad("#eeeeee")
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4)
    V = M.values.astype(float)
    fig, ax = plt.subplots(figsize=(1.6 + .6 * M.shape[1], .22 * M.shape[0] + .9))
    ax.imshow(np.ma.masked_invalid(V), cmap=cmap, norm=norm, aspect="auto")
    for (r, c), v in np.ndenumerate(V):
        ax.text(c, r, "NA" if np.isnan(v) else int(v), ha="center", va="center", fontsize=6.5,
                color="#999999" if np.isnan(v) else ("white" if v >= 2.5 else "#333"))
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(xlabels or M.columns, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index, fontsize=7)
    ax.set_xticks(np.arange(-.5, M.shape[1]), minor=True); ax.set_yticks(np.arange(-.5, M.shape[0]), minor=True)
    ax.grid(which="minor", color="white", lw=1); ax.tick_params(which="both", length=0)
    ax.set_title(title + " (NA: not analysed)", fontsize=7.5)
    plt.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
