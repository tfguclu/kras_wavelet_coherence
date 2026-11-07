#!/usr/bin/env python
"""
kras_hmr_steptest.py -- can one shared transition produce a coherent episode at 80-120 ns?

Synthetic signals (10 us at 1 ns; AR(1) noise, tau 25 ns, unit SD) analysed with the
unchanged pipeline functions (band_setup, pair_band_scalar, episodes):
  step     one shared sigmoid step (width 5 ns) of height A at 5 us, A = 10..50
  block    two independent H-bond-like telegraph signals (occupancy 0.3, mean on-time 20 ns)
           with one shared block of 50-500 ns in which both bonds are formed
  switch   both telegraph signals change occupancy (0.3 -> 0.7) at 5 us
  osc      a shared 100 ns oscillation lasting 0.5, 1 or 2 us (calibrates episode length)
Fast fluctuation of the noise = SD around a 50 ns running mean (0.55 for unit AR(1)),
so a step of height A is A / 0.55 times the fast fluctuation.
Run next to the kras_hmr_*.py scripts:  python kras_hmr_steptest.py   (about 10 min)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import kras_hmr_averages as av
from kras_hmr_wco import episodes
import kras_hmr_runcheck as rc

T, dt, tdec = 10000, 1.0, av.BASE['tdec']
t = np.arange(T) * dt
B = av.band_setup(T, dt, tdec); B['dt'] = dt
MINEP, THR, TE = rc.eff_ep(av.BASE), av.BASE['thr'], 5000.0


def ar1(rng, tau, n):
    a = np.exp(-dt / tau); e = rng.normal(0, np.sqrt(1 - a * a), n); x = np.zeros(n)
    for k in range(1, n): x[k] = a * x[k - 1] + e[k]
    return x


def telegraph(rng, p_on, tau_on, n):
    k_off = 1 / tau_on; k_on = k_off * p_on / (1 - p_on); x = np.zeros(n); s = rng.random() < p_on
    for i in range(n):
        x[i] = s
        if s and rng.random() < k_off: s = False
        elif (not s) and rng.random() < k_on: s = True
    return x


def ep_len(x, y, floor=None, near=None):
    Rb, _ = av.pair_band_scalar(x, y, B, tdec, floor=floor)
    ep = episodes((Rb >= THR) & B['valid'], B['dtd'], MINEP)
    if near is not None:
        ep = [e for e in ep if e[0] - 1000 <= near <= e[1] + 1000]
    return max([l for *_, l in ep], default=0.0)


rows = []
for seed in range(40):
    rng = np.random.default_rng(2000 + seed)
    for A in [10, 15, 20, 30, 50]:
        x, y = ar1(rng, 25, T), ar1(rng, 25, T); s = A / (1 + np.exp(-(t - TE) / 5.0))
        rows.append(dict(case='step', size=A, seed=seed, ep_ns=ep_len(x + s, y + s, near=TE)))
    for w in [50, 200, 500]:
        x, y = telegraph(rng, .3, 20, T), telegraph(rng, .3, 20, T); m = (t >= TE) & (t < TE + w); x[m] = y[m] = 1
        rows.append(dict(case='block', size=w, seed=seed, ep_ns=ep_len(x, y, floor=av.POW_FLOOR)))
    x, y = telegraph(rng, .3, 20, T), telegraph(rng, .3, 20, T)
    x2, y2 = telegraph(rng, .7, 47, T), telegraph(rng, .7, 47, T)
    rows.append(dict(case='switch', size=0, seed=seed,
                     ep_ns=ep_len(np.where(t < TE, x, x2), np.where(t < TE, y, y2), floor=av.POW_FLOOR)))
    if seed < 20:
        for D in [500, 1000, 2000]:
            x, y = ar1(rng, 25, T), ar1(rng, 25, T)
            o = 2 * ((t >= TE - D / 2) & (t < TE + D / 2)) * np.sin(2 * np.pi * t / 100)
            rows.append(dict(case='osc', size=D, seed=seed, ep_ns=ep_len(x + o, y + o)))
R = pd.DataFrame(rows); R.to_csv('steptest.csv', index=False)
print(R.groupby(['case', 'size']).ep_ns.agg(n='size', frac_with_episode=lambda v: (v > 0).mean(),
                                            median_ns=lambda v: v[v > 0].median() if (v > 0).any() else 0,
                                            max_ns='max').to_string())
