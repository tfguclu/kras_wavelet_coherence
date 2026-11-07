"""Methods figure: how band wavelet coherence and episodes are obtained (synthetic signals,
computed with the unchanged pipeline functions)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'analysis')]
os.chdir(HERE)
import sys, os
from pub_style import *
import numpy as np
from matplotlib.patches import Rectangle
from kras_hmr_wco import cwt_batch, episodes, W0, FF
from kras_hmr_averages import band_setup, BASE
import kras_hmr_runcheck as rc

rng = np.random.default_rng(7)
T, dt, tdec = 10000, 1.0, BASE['tdec']
t = np.arange(T) * dt


def ar1(tau, n):
    a = np.exp(-dt / tau); e = rng.normal(0, np.sqrt(1 - a * a), n); x = np.zeros(n)
    for k in range(1, n):
        x[k] = a * x[k - 1] + e[k]
    return x


def env(t, s, e, r=150):
    return 0.5 * (np.tanh((t - s) / r * 4) - np.tanh((t - e) / r * 4)) * 0.5 + 0.0


E = np.zeros(T)
for s, e, g in [(2400, 4300, 1), (6000, 7100, 1), (8000, 8200, 1.8)]:
    E += g * np.clip(0.5 * (np.tanh((t - s) / 60) - np.tanh((t - e) / 60)), 0, 1)
osc = np.sin(2 * np.pi * t / 100)
A = 2.0
x = A * E * osc + ar1(25, T) + 0.8 * np.sin(2 * np.pi * t / 2600)
y = A * E * np.sin(2 * np.pi * (t - 3) / 100) + ar1(25, T) + 0.8 * np.sin(2 * np.pi * t / 1900 + 1)
ysur = np.roll(y, 4600)                      # circular-shift surrogate

# ---------------- band (80-120 ns) coherence, exactly as in the pipeline
B = band_setup(T, dt, tdec); B['dt'] = dt
def band_R(a, b):
    W = cwt_batch(np.vstack([a, b]), dt, B['scales'])[:, :, ::tdec]
    P = np.maximum(B['sm'](np.abs(W) ** 2).real, 0)
    C = B['sm'](W[0][None] * np.conj(W[1][None]))[0]
    R = np.abs(C) ** 2 / (P[0] * P[1] + 1e-30)
    return R[B['band']].mean(0), C[B['band']]
Rb, Cb = band_R(x, y); Rs, _ = band_R(x, ysur)
td = np.arange(B['Td']) * B['dtd']
ep = episodes((Rb >= BASE['thr']) & B['valid'], B['dtd'], rc.eff_ep(BASE))
above = (Rb >= BASE['thr']) & B['valid']

# ---------------- broad maps (10-1000 ns) with the same smoothing
M = band_setup(T, dt, tdec, 10, 1000); M['dt'] = dt
M['band'] = (M['periods'] >= 10) & (M['periods'] <= 1000)
W = cwt_batch(np.vstack([x, y]), dt, M['scales'])[:, :, ::tdec]
P = np.maximum(M['sm'](np.abs(W) ** 2).real, 0)
C = M['sm'](W[0][None] * np.conj(W[1][None]))[0]
R = np.abs(C) ** 2 / (P[0] * P[1] + 1e-30)
per = M['periods']; keep = M['band']
edge_s = np.sqrt(2) * M['scales'] + 2 * 2.0 * M['scales']
coi = (td[None, :] >= edge_s[:, None]) & (td[None, :] <= (T - 1) * dt - edge_s[:, None])
print('episodes', [(round(s), round(e)) for s, e, _ in ep], 'max surrogate R', Rs[B['valid']].max().round(2))

# ================================================================ figure
fig = plt.figure(figsize=(7.1, 4.3))
def lab(ax, l, title, x=-0.02):
    ax.text(x, 1.10, l, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')
    ax.text(x + 0.07 if x > -0.1 else x + .1, 1.10, title, transform=ax.transAxes, fontsize=7.2,
            fontweight='semibold', va='bottom', color=INK)
cx, cy = INK, '#8c8c8c'
us = lambda v: v / 1000

# a signals
ax = fig.add_axes([.055, .64, .56, .26])
for v, c, o, l in [(x, cx, 1.6, 'signal 1'), (y, cy, -1.6, 'signal 2')]:
    vv = (v - v.mean()) / v.std() * .55
    ax.plot(us(t), vv + o, color=c, lw=.35)
    pass
ax.set_xlim(0, 10); ax.set_ylim(-3.2, 4.3); ax.set_yticks([]); ax.spines['left'].set_visible(False)
ax.text(0, 3.75, 'signal 1', color=cx, fontsize=6, ha='left', va='center'); ax.text(1.25, 3.75, 'signal 2', color=cy, fontsize=6, ha='left', va='center')
ax.set_xticks([0, 2, 4, 6, 8, 10]); ax.set_xticklabels(['0', '2', '4', '6', '8', '10 µs'])
for s, e in [(2.4, 4.3), (6.0, 7.1), (8.0, 8.2)]:
    ax.axvspan(s, e, color=BAND, zorder=0, lw=0)
lab(ax, 'a', 'Two signals of one residue pair (synthetic)')
# zoom
az = fig.add_axes([.66, .64, .11, .26])
sl = (t >= 3000) & (t <= 3300)
smo = lambda v: np.convolve(v, np.ones(15) / 15, 'same')
for v, c, o in [(x, cx, .9), (y, cy, -.9)]:
    vv = smo(v)[sl]; vv = (vv - vv.min()) / (vv.max() - vv.min()) * 1.5 - .75
    az.plot(t[sl] - 3000, vv + o, color=c, lw=.6)
az.set_yticks([]); az.spines['left'].set_visible(False); az.set_xticks([0, 100, 200, 300])
az.set_xticklabels(['0', '', '', '300 ns'], fontsize=5.6); az.set_ylim(-1.9, 1.9)
az.text(0.5, 1.06, 'zoom (300 ns,\n15 ns mean)', transform=az.transAxes, ha='center', va='bottom', fontsize=5.2, color=INK2, linespacing=1.0)

# b wavelets
ax = fig.add_axes([.83, .64, .16, .26])
for k, (pp, o) in enumerate([(25, 2), (100, 0), (400, -2)]):
    s = pp / FF; tt = np.linspace(-700, 700, 3000); u = tt / s
    psi = np.pi ** -.25 * np.exp(1j * W0 * u) * np.exp(-u * u / 2)
    ax.plot(tt, psi.real * .8 + o, color=INK, lw=.5); ax.plot(tt, psi.imag * .8 + o, color=MUTE, lw=.5)
    ax.text(720, o, f'{pp} ns', fontsize=5.4, color=INK2, va='center')
ax.set_xlim(-700, 1000); ax.set_ylim(-3, 3); ax.axis('off')
lab(ax, 'b', 'Morlet wavelets (ω₀ = 6)', x=-0.12)
ax.text(0, -3.1, 're  ', color=INK, fontsize=5.2, ha='right', va='top'); ax.text(0, -3.1, '  im', color=MUTE, fontsize=5.2, ha='left', va='top')

def mapax(rect):
    a = fig.add_axes(rect); a.set_yscale('log'); a.set_ylim(10, 1000)
    a.set_yticks([10, 100, 1000]); a.set_yticklabels(['10', '100', '1000'])
    a.set_xlim(0, 10); a.set_xticks([0, 5, 10]); a.set_xticklabels(['0', '5', '10 µs'])
    return a
def shade_coi(a):
    m = np.where(coi[keep], np.nan, 1.0)
    a.pcolormesh(us(td), per[keep], m, cmap=matplotlib.colors.ListedColormap(['white']), alpha=.65,
                 shading='auto', rasterized=True)
import matplotlib

# c power
ax = mapax([.055, .1, .25, .36])
Pn = P[0][keep] / np.median(P[0][keep], axis=1, keepdims=True)
ax.pcolormesh(us(td), per[keep], np.log10(Pn), cmap='Greys', vmin=-1, vmax=1.2, shading='auto', rasterized=True)
shade_coi(ax)
ax.set_ylabel('period (ns)', fontsize=6.4, labelpad=1)
lab(ax, 'c', 'Wavelet power, signal 1')
ax.text(.02, .93, 'edge zone', transform=ax.transAxes, fontsize=5, color=MUTE)

# d coherence + phase
ax = mapax([.375, .1, .25, .36])
ax.pcolormesh(us(td), per[keep], R[keep], cmap='Greys', vmin=0, vmax=1, shading='auto', rasterized=True)
import matplotlib.patheffects as pe
cs = ax.contour(us(td), per[keep], np.where(coi[keep], R[keep], 0), levels=[.75], colors=[INK], linewidths=.6)
plt.setp(cs.collections if hasattr(cs,'collections') else [cs], path_effects=[pe.Stroke(linewidth=1.6, foreground='white'), pe.Normal()])
shade_coi(ax)
ax.axhline(80, color=INK, lw=.6, ls=':'); ax.axhline(120, color=INK, lw=.6, ls=':')
ph = np.angle(C[keep])
pk = per[keep]
for ti in np.arange(25, B['Td'] - 25, 40):
    for pi in [25, 100, 300]:
        si = np.argmin(abs(pk - pi))
        if R[keep][si, ti] > .75 and coi[keep][si, ti]:
            a_ = ph[si, ti]; L = .35
            ax.annotate('', xy=(us(td[ti]) + L * np.cos(a_), pi * 10 ** (.08 * np.sin(a_))),
                        xytext=(us(td[ti]), pi), arrowprops=dict(arrowstyle='-|>', lw=.6, color='white', mutation_scale=4))
lab(ax, 'd', 'Coherence R(s, t) and phase')
ax.text(4.3, 125, '80–120 ns', fontsize=5.2, color=INK, va='bottom')

# e band R(t)
ax = fig.add_axes([.72, .1, .27, .36])
v = B['valid']
excl(ax, 0, us(B['edge']), zorder=0, ymax=1 / 1.32); excl(ax, us(T - B['edge']), 10, zorder=0, ymax=1 / 1.32)
ax.plot(us(td), np.where(v, Rs, np.nan), color=MUTE, lw=.5)
ax.plot(us(td), np.where(v, Rb, np.nan), color=INK, lw=.7)
ax.axhline(BASE['thr'], color=MUTE, lw=.5, ls='--')
for s, e, _ in ep:
    ax.add_patch(Rectangle((us(s), 1.02), us(e - s), .05, color=INK2, lw=0, clip_on=False))
# short crossings that fail the 500 ns rule
lab_ = np.diff(np.concatenate([[0], above.astype(int), [0]]))
for s, e in zip(np.where(lab_ == 1)[0], np.where(lab_ == -1)[0]):
    if (e - s) * B['dtd'] < rc.eff_ep(BASE):
        ax.add_patch(Rectangle((us(td[s]), 1.02), us((e - s) * B['dtd']), .05, fill=False, ec=MUTE, lw=.4, clip_on=False))
ax.set_xlim(0, 10); ax.set_ylim(0, 1.32); ax.spines['left'].set_bounds(0, 1); ax.set_xticks([0, 5, 10]); ax.set_xticklabels(['0', '5', '10 µs'])
ax.set_yticks([0, .5, .75, 1]); ax.set_yticklabels(['0', '0.5', '0.75', '1'])
ax.set_ylabel('band coherence', fontsize=6.4, labelpad=1)
lab(ax, 'e', 'Band average, episodes ≥ 500 ns')
ax.plot([0.9, 1.6], [1.24, 1.24], color=INK, lw=.7); ax.text(1.75, 1.24, 'pair', va='center', fontsize=5.2, color=INK)
ax.plot([3.4, 4.1], [1.24, 1.24], color=MUTE, lw=.5); ax.text(4.25, 1.24, 'time-shifted control', va='center', fontsize=5.2, color=MUTE)
save(fig, 'Figure8.png')
