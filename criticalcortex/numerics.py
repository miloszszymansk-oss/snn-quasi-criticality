"""
criticalcortex.numerics — golden Python reference for the M0 numeric core.

This module defines the EXACT numerical semantics the Rust routing kernel (M1)
must reproduce bit-for-bit under the T0 golden-master tests. Two objects live
here, and nothing else — no network, no plasticity, no RNG:

  1. The Izhikevich (2003) neuron step, integrated per SPEC §4.2:
       - v : TWO half-dt forward-Euler substeps (stability near the 0.04 v^2 term)
       - u : ONE full-dt forward-Euler step, evaluated at the post-substep v
       - reset: if v >= V_PEAK  ->  v <- c,  u <- u + d   (applied after the update)

  2. The current-based exponential synapse: g_{t+dt} = g_t * exp(-dt/tau_syn),
     the EXACT one-step propagator of g' = -g/tau_syn (exponential Euler).

Everything is float64 and vectorized over neurons (Structure-of-Arrays): there is
no Python-level per-neuron loop in the hot paths, only per-timestep iteration.
"""

from __future__ import annotations

import numpy as np

V_PEAK = 30.0  # mV — spike cutoff / detection threshold (Izhikevich 2003)


# ---------------------------------------------------------------------------
# Smooth right-hand side (no reset). Exposed so an INDEPENDENT high-order
# integrator (RK4, in the T1.3 test) can verify the production scheme's order.
# ---------------------------------------------------------------------------
def izh_rhs(v, u, I, a, b):
    """dv/dt, du/dt of the Izhikevich ODE (v in mV, t in ms, I in mV/ms)."""
    dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
    du = a * (b * v - u)
    return dv, du


# ---------------------------------------------------------------------------
# The production step: two half-dt substeps on v, one full-dt on u, then reset.
# ---------------------------------------------------------------------------
def izh_step(v, u, I, a, b, c, d, dt):
    """
    Advance (v, u) by one step of size `dt` (ms) under input current `I`.

    SPEC §4.2 'half-step Euler'. The exact ordering below is the contract the
    Rust port must match bit-for-bit — in particular, the v-substeps hold u
    frozen at u0, and the u-step uses the fully-updated v:

        dv1    = f(v0,   u0)
        v_half = v0     + (dt/2)*dv1
        dv2    = f(v_half, u0)                 # u still frozen at u0
        v_full = v_half + (dt/2)*dv2
        u_new  = u0 + dt * a*(b*v_full - u0)   # u uses the NEW v
        if v_full >= V_PEAK:  v <- c ;  u <- u_new + d   # reset applied this step

    Parameters may be scalars or ndarrays; the body is fully vectorized.
    Returns (v_out, u_out, spiked), `spiked` a boolean of the input's shape.
    """
    v = np.asarray(v, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    half = 0.5 * dt

    dv1 = 0.04 * v * v + 5.0 * v + 140.0 - u + I
    v_half = v + half * dv1
    dv2 = 0.04 * v_half * v_half + 5.0 * v_half + 140.0 - u + I
    v_full = v_half + half * dv2

    u_new = u + dt * (a * (b * v_full - u))

    spiked = v_full >= V_PEAK
    v_out = np.where(spiked, c, v_full)
    u_out = np.where(spiked, u_new + d, u_new)
    return v_out, u_out, spiked


# ---------------------------------------------------------------------------
# Exponential (current-based) synapse.
# ---------------------------------------------------------------------------
def exp_decay_factor(dt, tau_syn):
    """Per-step propagator r = exp(-dt/tau_syn) of g' = -g/tau_syn. The kernel
    simply does `g *= r` each step; because r is the exact solution over one
    step, the discrete trajectory equals the continuous one to fp precision."""
    return float(np.exp(-dt / tau_syn))


def simulate_decay(g0, tau_syn, dt, n):
    """
    Iterate g <- g * exp(-dt/tau_syn) for `n` steps (the actual kernel update).

    Returns g_hist[k] = g after (k+1) steps, i.e. the value at t = (k+1)*dt.
    Since exp(-dt/tau)**(k+1) == exp(-(k+1)*dt/tau) analytically, this array must
    match g0*exp(-t/tau) to floating-point precision — the invariant of T1.2.
    """
    r = exp_decay_factor(dt, tau_syn)
    g_hist = np.empty(int(n), dtype=np.float64)
    g = float(g0)
    for k in range(int(n)):
        g *= r
        g_hist[k] = g
    return g_hist


# ---------------------------------------------------------------------------
# Steady-state f–I curve of an isolated neuron (the calibration primitive).
# ---------------------------------------------------------------------------
def measure_fI(a, b, c, d, I, dt, T_ms, warmup_ms=200.0, v0=-65.0):
    """
    Firing rate (Hz) of an isolated Izhikevich neuron vs constant input current.

    Vectorized across the current array `I`: one neuron per current, all advanced
    in lockstep (SoA), so the entire sweep is O(n_steps) vectorized updates with
    NO per-neuron Python loop. Spikes during the first `warmup_ms` are discarded
    to strip the initial transient; f = (spikes in the counting window)/window[s].

    Deterministic: constant current, no RNG. Returns f as float64, shape == I.
    """
    I = np.atleast_1d(np.asarray(I, dtype=np.float64))
    n_steps = int(round(T_ms / dt))
    n_warm = int(round(warmup_ms / dt))

    v = np.full(I.shape, float(v0), dtype=np.float64)
    u = b * v
    counts = np.zeros(I.shape, dtype=np.int64)

    for s in range(n_steps):
        v, u, spiked = izh_step(v, u, I, a, b, c, d, dt)
        if s >= n_warm:
            counts += spiked.astype(np.int64)

    window_s = (n_steps - n_warm) * dt / 1000.0
    return (counts / window_s).astype(np.float64)
