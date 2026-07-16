"""
criticalcortex.rust_driver — thin Python driver for the compiled Rust kernel.

Encapsulates the `criticalcortex._kernel.step_block` call: allocates the transpose-free
STDP state, flattens the delay ring to the 1-D layout the kernel expects, builds the
`KernelParams` (with decay factors precomputed to f32), and streams the run in blocks
while tracking the maximum `alloc_bytes` reported (which must stay 0). Used by the M4
long-horizon verification and the benchmark. Imports the extension lazily so the rest of
the package works whether or not the kernel is built.
"""

from __future__ import annotations

import numpy as np

from .simulation import build_network

HIST_CAP = 64   # per-neuron post-spike history capacity for deferred potentiation


def kernel_available() -> bool:
    """True iff the compiled Rust extension `criticalcortex._kernel` can be imported."""
    try:
        from criticalcortex import _kernel  # noqa: F401
        return True
    except Exception:
        return False


def make_kernel_params(km, p):
    """Build KernelParams with decay factors rounded to f32 exactly as the reference does."""
    a_syn = float(np.float32(np.exp(-p.dt / p.tau_syn)))
    a_stdp = float(np.float32(np.exp(-p.dt / p.tau_stdp)))
    return km.KernelParams(
        dt=p.dt, a=p.a, b=p.b, c=p.c, d=p.d, v_peak=p.v_peak,
        a_plus=p.a_plus, a_minus=p.a_minus, a_syn=a_syn, a_stdp=a_stdp,
        mu_ext=p.mu_ext, sigma_ext=p.sigma_ext, refractory_steps=p.refractory_steps, seed=p.seed,
        homeo_enabled=1 if p.homeo_enabled else 0, tau_homeo=p.tau_homeo,
    )


def allocate_kernel_state(conn, params, hist_cap: int = HIST_CAP, aer_capacity: int = 4_000_000):
    """Preallocate everything the kernel borrows: neuron state (via build_network, incl.
    the deterministic v0-spread), the transpose-free STDP state, and the AER sink."""
    from criticalcortex import _kernel as km
    net = build_network(conn, params, aer_capacity)
    n = conn.n
    state = dict(
        net=net,
        ring_flat=net.ring.reshape(-1),                       # zero-copy 1-D view
        last_pre=np.full(n, -1, np.int64),
        xhat=np.zeros(n, np.float32),
        post_hist=np.full(n * hist_cap, -1, np.int64),
        post_cnt=np.zeros(n, np.int64),
        kp=make_kernel_params(km, params),
    )
    return state


def step_kernel(state, start_step: int, n_steps: int):
    """One `step_block` call over the preallocated state; returns StepStats."""
    from criticalcortex import _kernel as km
    net = state["net"]
    return km.step_block(
        net.v, net.u, net.g_exc, net.g_inh, net.refrac, net.t_last,
        net.conn.indptr, net.conn.indices, net.weight, net.conn.delay,
        state["ring_flat"], net.x_pre, net.x_post,
        state["last_pre"], state["xhat"], state["post_hist"], state["post_cnt"],
        net.x_avail, net.homeo_keep,
        state["kp"], int(start_step), int(n_steps),
        net.aer_step, net.aer_neuron, net.aer_count,
    )


def run_rust(conn, params, steps: int, block: int = 1000, hist_cap: int = HIST_CAP,
             aer_capacity: int = 4_000_000):
    """Run `steps` through the Rust kernel in blocks of `block`. Returns (net, max_alloc)."""
    state = allocate_kernel_state(conn, params, hist_cap, aer_capacity)
    max_alloc, done = 0, 0
    while done < steps:
        nb = min(block, steps - done)
        stats = step_kernel(state, done, nb)
        max_alloc = max(max_alloc, int(stats.alloc_bytes))
        done += nb
    return state["net"], max_alloc
