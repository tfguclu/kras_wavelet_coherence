"""Validate the in-house scalar wavelet coherence against pycwt (port of Grinsted et al.)."""
import sys
import numpy as np, pycwt
from kras_hmr_wco import cwt_batch, Smoother, make_scales, FF
SIG = sys.argv[1] if len(sys.argv) > 1 else 'signals.npz'   # wco_hmr/pairsignals/signals.npz (kras_hmr_pairsignals.py)
z = np.load(SIG, allow_pickle=True)
dj = 1/12; scales = make_scales(80, 120, dj, 0.3); per = scales*FF; band = (per >= 80) & (per <= 120)
T = 10000; t = np.arange(T); edge = 1500; valid = (t >= edge) & (t <= T-1-edge)

def ours(x, y, tfac):
    W = cwt_batch(np.vstack([x, y]).astype(float), 1.0, scales)
    sm = Smoother(scales, T, 1.0, dj, tfac)
    P = sm(np.abs(W)**2).real; C = sm(W[0][None]*np.conj(W[1][None]))[0]
    return (np.abs(C)**2/(P[0]*P[1]+1e-30))[band].mean(0)

import pycwt.mothers as _pm
_f0 = _pm.Morlet._set_f0
def _set_f0(self, f0):                            # pycwt's scale boxcar is deltaj0/dj*2 wide (1.2 octave);
    _f0(self, f0); self.deltaj0 = 0.30            # match ours and Grinsted (0.6 octave). Must be set per instance.
_pm.Morlet._set_f0 = _set_f0
_pm.rect = lambda n, normalize=False: np.ones(n) / (n if normalize else 1)   # uniform boxcar (pycwt: trapezoid ends)

def theirs(x, y):
    WCT, aWCT, coi, freq, sig = pycwt.wct(x, y, 1.0, dj=dj, s0=scales[0], J=len(scales)-1, sig=False, wavelet='morlet', normalize=True)
    p = 1/freq
    assert np.allclose(p, per, rtol=1e-6), (p[:3], per[:3])
    return np.real(WCT)[band].mean(0)

pairs = []
for key, lab in [('17-32|sasa|sasa', 'SASA S17-Y32'), ('69-102|sasa|sasa', 'SASA D69-R102'), ('99-102|sasa|sasa', 'SASA Q99-R102'),
                 ('69-102|hb|69-102 | 99-102', 'HB R102 bridge')]:
    for r in ['namd_3', 'namd_3_2', 'namd_3_3']:
        raw = z[f'{key}|{r}|raw'].astype(float); pairs.append((f'{lab} {r}', raw[0], raw[1]))
rng = np.random.default_rng(0)
def ar1(n, tau=25):
    a = np.exp(-1/tau); e = rng.normal(0, np.sqrt(1-a*a), n); x = np.zeros(n)
    for i in range(1, n): x[i] = a*x[i-1] + e[i]
    return x
o = 2*((t > 3000) & (t < 6000))*np.sin(2*np.pi*t/100)
pairs.append(('synthetic shared 100 ns', ar1(T)+o, ar1(T)+np.roll(o, 3)))
print(f"{'pair':32s} r(ours,pycwt)  max|diff|  mean|diff|  frac(both>=0.75 agree)")
for name, x, y in pairs:
    a_, b_ = ours(x, y, 1.0)[valid], theirs(x, y)[valid]
    agree = np.mean((a_ >= .75) == (b_ >= .75))
    print(f"{name:32s} {np.corrcoef(a_, b_)[0,1]:.4f}        {np.abs(a_-b_).max():.4f}     {np.abs(a_-b_).mean():.4f}      {agree:.4f}", flush=True)
