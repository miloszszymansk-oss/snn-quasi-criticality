"""
m5_viz.sim_engine — backend-agnostic streaming driver for the live SOqC visualization.

Wraps the SAME simulation the M3 validation uses (E/I-balanced Izhikevich + depressing-synapse
homeostasis) and advances it in small batches so a server can broadcast render state at a capped
frame rate, DECOUPLED from the simulation tick. The Rust kernel is used when it is importable
(your Mac); otherwise the pure-NumPy reference runs (so this module is testable anywhere).

Per `step(n_steps)` batch it returns:
  * the neuron ids that spiked in the batch   (drives the "glow" in the WebGL cloud)
  * the instantaneous firing rate (Hz)         (drives the rate plot)
  * mean available synaptic resource ⟨x⟩       (the SOqC adaptation variable)
and it maintains a rolling per-step activity buffer from which a live avalanche-size histogram
is (re)computed on demand — the "watch SOqC organize itself" analytics.

MEMORY: the AER sink is RESET every batch (aer_count[0] ← 0 before stepping, spikes read after),
so an unbounded live run reuses a fixed buffer — nothing grows without bound. Zero per-step
Python allocation in the kernel path is preserved; this module only touches the batch boundary.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# make the sibling `criticalcortex` package importable when run from m5_viz/ without installing
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from criticalcortex.connectome import build_connectome
from criticalcortex.spatial_connectome import (build_spatial_connectome, visualization_edges,
                                               fibonacci_sphere)
from criticalcortex.simulation import SimParams, build_network, step_block_reference

try:
    from criticalcortex.rust_driver import kernel_available, allocate_kernel_state, step_kernel
except Exception:                                       # pragma: no cover
    def kernel_available():
        return False

DT_MS = 0.1
MAX_SPIKES_PER_FRAME = 20_000                           # payload guard for fast playback / bursts


def _avalanche_sizes(activity: np.ndarray) -> np.ndarray:
    """⟨ISI⟩-binned avalanche sizes from a per-step population-activity array (Beggs–Plenz;
    the frozen §7.4 protocol). Maximal runs of non-empty bins at bin width Δ=⟨ISI⟩."""
    total = activity.sum()
    if total < 10:
        return np.empty(0, dtype=np.int64)
    bw = max(1, int(round(activity.size / total)))
    nb = activity.size // bw
    if nb < 2:
        return np.empty(0, dtype=np.int64)
    B = activity[: nb * bw].reshape(nb, bw).sum(axis=1)
    occ = (B > 0).astype(np.int8)
    edges = np.diff(np.concatenate(([0], occ, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return np.array([B[s:e].sum() for s, e in zip(starts, ends)], dtype=np.int64)


class SimEngine:
    def __init__(self, N=1000, g=3.5, mu_ext=3.3, ei_ratio=0.8, u_release=0.2,
                 tau_homeo=50_000.0, sigma_ext=2.0, seed=1, k=100,
                 hist_window_steps=200_000, aer_capacity=4_000_000,
                 spatial=False, locality=0.35, distance_delays=True, viz_max_edges=6000):
        self.N = int(N)
        self.dt_ms = DT_MS
        self.spatial = bool(spatial)
        if self.spatial:
            self.conn = build_spatial_connectome(n=self.N, k=k, g=g, seed=seed, ei_ratio=ei_ratio,
                                                 locality=locality, distance_delays=distance_delays)
            self.positions = self.conn._positions
            self.viz_edges = visualization_edges(self.conn, max_edges=viz_max_edges)
        else:
            self.conn = build_connectome(n=self.N, k=k, g=g, seed=seed, ei_ratio=ei_ratio)
            self.positions = fibonacci_sphere(self.N)    # layout only (no local edges to draw)
            self.viz_edges = np.empty((0, 2), dtype=np.int32)
        self.params = SimParams(seed=seed, mu_ext=mu_ext, sigma_ext=sigma_ext, v0_spread=45.0,
                                homeo_enabled=True, tau_homeo=float(tau_homeo), u_release=u_release)
        self.cfg = dict(N=self.N, g=g, mu_ext=mu_ext, ei_ratio=ei_ratio, u_release=u_release,
                        tau_homeo=tau_homeo, seed=seed, k=k, spatial=self.spatial,
                        locality=locality if self.spatial else None)
        self.is_inhib = self.conn.is_inhib.astype(np.uint8)

        self.use_rust = kernel_available()
        if self.use_rust:
            self.state = allocate_kernel_state(self.conn, self.params, aer_capacity=aer_capacity)
            self.net = self.state["net"]
        else:
            self.net = build_network(self.conn, self.params, aer_capacity=aer_capacity)
        self.abs_step = 0

        # rolling per-step activity ring (for the live avalanche histogram)
        self.W = int(hist_window_steps)
        self._act = np.zeros(self.W, dtype=np.int32)
        self._apos = 0
        self._afilled = 0

        # fixed log-spaced size-histogram edges (shared with the frontend via meta())
        self.hist_edges = np.unique(
            np.floor(np.logspace(0, np.log10(max(8 * self.N, 64)), 40)).astype(np.int64))

    # -- one simulation batch -----------------------------------------------------------
    def step(self, n_steps: int) -> dict:
        net = self.net
        net.aer_count[0] = 0                             # reuse the AER buffer each batch
        if self.use_rust:
            step_kernel(self.state, self.abs_step, n_steps)
        else:
            step_block_reference(
                net.v, net.u, net.g_exc, net.g_inh, net.refrac, net.t_last,
                net.conn.indptr, net.conn.indices, net.weight, net.conn.delay,
                net.in_indptr, net.in_edge_ids, net.in_src,
                net.ring, net.x_pre, net.x_post,
                self.params, self.abs_step, n_steps,
                net.aer_step, net.aer_neuron, net.aer_count,
                net.x_avail, net.homeo_keep, net.soc_state,
            )
        c = int(net.aer_count[0])
        spike_steps = net.aer_step[:c]
        spike_neurons = net.aer_neuron[:c]

        # per-step population activity for this batch -> rolling ring
        batch_act = np.bincount(spike_steps - self.abs_step, minlength=n_steps)[:n_steps]
        self._push_activity(batch_act.astype(np.int32))

        self.abs_step += n_steps
        rate = c / (self.N * n_steps * self.dt_ms / 1000.0)     # Hz over this batch

        # payload guard: subsample the glow set if a burst overflows the frame
        neurons = spike_neurons
        if neurons.size > MAX_SPIKES_PER_FRAME:
            idx = np.linspace(0, neurons.size - 1, MAX_SPIKES_PER_FRAME).astype(np.int64)
            neurons = neurons[idx]
        # per-neuron resource ⟨x⟩ ∈ (0,1] -> uint8 for the heatmap (1 byte/neuron/frame)
        x_u8 = np.clip(net.x_avail * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return dict(
            neurons=neurons.astype(np.uint16),
            n_spikes=int(c),
            rate=float(rate),
            xbar=float(net.x_avail.mean()),
            abs_step=int(self.abs_step),
            x_u8=x_u8,
        )

    def _push_activity(self, batch: np.ndarray):
        L = batch.size
        W = self.W
        if L >= W:
            self._act[:] = batch[-W:]
            self._apos, self._afilled = 0, W
            return
        p = self._apos
        end = p + L
        if end <= W:
            self._act[p:end] = batch
        else:
            first = W - p
            self._act[p:] = batch[:first]
            self._act[: end - W] = batch[first:]
        self._apos = end % W
        self._afilled = min(W, self._afilled + L)

    def _chrono_activity(self) -> np.ndarray:
        if self._afilled < self.W:
            return self._act[: self._afilled]
        return np.concatenate([self._act[self._apos:], self._act[: self._apos]])

    # -- live avalanche-size histogram over the rolling window --------------------------
    def avalanche_histogram(self) -> dict:
        sizes = _avalanche_sizes(self._chrono_activity())
        counts, _ = np.histogram(sizes, bins=np.append(self.hist_edges, self.hist_edges[-1] + 1))
        return dict(edges=self.hist_edges.tolist(),
                    counts=counts[: self.hist_edges.size].astype(np.int64).tolist(),
                    n_aval=int(sizes.size),
                    max_size=int(sizes.max()) if sizes.size else 0)

    # -- one-time metadata for a newly connected client --------------------------------
    def meta(self) -> dict:
        return dict(
            type="hello", N=self.N, dt_ms=self.dt_ms,
            backend="rust" if self.use_rust else "reference",
            is_inhib=self.is_inhib.tolist(),
            hist_edges=self.hist_edges.tolist(),
            spatial=self.spatial,
            positions=np.round(self.positions, 4).ravel().tolist(),   # flat 3N (frontend cloud)
            edges=self.viz_edges.ravel().tolist(),                    # flat 2M local-edge mesh
            cfg=self.cfg,
        )
