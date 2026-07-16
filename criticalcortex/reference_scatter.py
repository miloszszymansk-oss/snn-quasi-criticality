"""
criticalcortex.reference_scatter — pure-NumPy golden master for M1 (SPEC §3.3, §3.5).

Three obviously-correct reference objects the Rust kernel must reproduce under T2:

  1. csr_scatter            — immediate out-edge scatter (one timestep, no delay).
  2. DelayRing              — the delayed-delivery ring buffer (drain-before-schedule).
  3. pair-STDP references   — a DENSE eager scheme (brute-force ground truth) and the
                              TRANSPOSE-FREE online scheme (out-edge only) that must
                              match it to floating-point precision.

STDP model (discrete, all-to-all pair rule; the exact per-step convention IS the
contract, so dense and CSR agree bit-for-fp-bit):

    per step t:
      (1) decay traces:      x *= exp(-dt/tau_plus);   y *= exp(-dt/tau_minus)
      (2) DEPRESSION  (pre):  for pre spike i, every out-edge i->j:  w += -A_minus * y[j]
      (3) POTENTIATION (post): for post spike j, every in-edge i->j: w += +A_plus  * x[i]
      (4) increment spikers:  x[F] += 1;  y[F] += 1
    Traces are read BEFORE the same-step increment, so a coincident (Δt=0) pair
    contributes nothing.

The transpose-free scheme keeps ONLY out-edges. Depression is naturally out-edge
indexed (step 2). Potentiation (step 3) would need in-edges; instead each post spike
of j is realized LATER, when its presynaptic partner i next fires (and finally at an
end-of-run flush), reconstructing x_i at the post-spike step from i's stored trace —
so potentiation is folded into the presynaptic out-edge traversal with no transpose.
This is exact for the all-to-all pair model up to axonal delay (SPEC §3.5); the STDP
pairing here uses somatic spike steps (delay affects current delivery, not pairing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .connectome import Connectome


# ===========================================================================
# 1. Immediate CSR scatter (one timestep, no delay)
# ===========================================================================
def csr_scatter(indptr, indices, weight, spikes, n) -> np.ndarray:
    """
    Accumulate, per postsynaptic neuron, the summed weight of all edges emanating
    from the currently spiking presynaptic neurons `spikes`.

    Equivalent to the dense  W[:, spikes].sum(axis=1)  where W[post, pre] (SPEC §3.3
    'deliver' step, with delay factored out). Returns float64 [n].
    """
    out = np.zeros(n, dtype=np.float64)
    spikes = np.asarray(spikes, dtype=np.int64)
    if spikes.size == 0:
        return out
    idx = np.concatenate([indices[indptr[i]:indptr[i + 1]] for i in spikes])
    w = np.concatenate([weight[indptr[i]:indptr[i + 1]] for i in spikes])
    np.add.at(out, idx, w)                       # unbuffered: correct on duplicate targets
    return out


# ===========================================================================
# 2. Delayed-delivery ring buffer (SPEC §3.3)
# ===========================================================================
class DelayRing:
    """
    Dense conductance ring: `d_max` slots x `n` neurons. A delivery scheduled for
    absolute step `at_step` lands in slot `at_step % d_max`. The network loop must
    DRAIN the current step's slot before SCHEDULING new deliveries, which lets a
    ring of exactly `d_max` slots carry delays in [1, d_max] with no collision.
    """

    def __init__(self, d_max: int, n: int):
        self.d_max = int(d_max)
        self.n = int(n)
        self.buf = np.zeros((self.d_max, self.n), dtype=np.float64)

    def schedule(self, target, weight, at_step):
        """Add `weight` to `target`, to be delivered exactly at absolute `at_step`."""
        self.buf[int(at_step) % self.d_max, target] += weight

    def schedule_many(self, targets, weights, at_steps):
        """Vectorized schedule of many deliveries (targets/weights/at_steps arrays)."""
        slots = np.mod(np.asarray(at_steps, dtype=np.int64), self.d_max)
        np.add.at(self.buf, (slots, np.asarray(targets, dtype=np.int64)),
                  np.asarray(weights, dtype=np.float64))

    def drain(self, step) -> np.ndarray:
        """Return (a copy of) the currents due at absolute `step`, then clear the slot."""
        slot = int(step) % self.d_max
        out = self.buf[slot].copy()
        self.buf[slot] = 0.0
        return out


# ===========================================================================
# STDP configuration + result
# ===========================================================================
@dataclass
class STDPParams:
    a_plus: float = 0.01
    a_minus: float = 0.0105
    tau_plus: float = 20.0      # ms
    tau_minus: float = 20.0     # ms
    dt: float = 0.1             # ms


@dataclass
class STDPResult:
    weights: np.ndarray          # edge-aligned final weights (parallel to conn.indices)
    W: np.ndarray | None = None  # dense final matrix W[post,pre] (dense reference only)


def generate_spike_train(n, steps, rate_hz, dt, seed) -> np.ndarray:
    """Deterministic Bernoulli/Poisson spike train, bool[steps, n]. p = rate*dt/1000."""
    rng = np.random.default_rng(seed)
    p = rate_hz * dt / 1000.0
    return rng.random((steps, n)) < p


# ===========================================================================
# 3a. DENSE eager pair-STDP — brute-force ground truth
# ===========================================================================
def stdp_dense_reference(conn: Connectome, spikes: np.ndarray, p: STDPParams) -> STDPResult:
    """
    All-to-all pair STDP on the full dense matrix. Potentiation is applied at post
    spikes over IN-edges (matrix rows), depression at pre spikes over OUT-edges
    (matrix columns). Obviously correct; O(N^2) — reference/testing only.
    """
    n = conn.n
    src = conn.edge_sources()
    dst = conn.indices.astype(np.int64)
    W = conn.to_dense()                                  # [post, pre], float64
    A = conn.adjacency()                                 # bool mask of existing synapses

    ap = np.exp(-p.dt / p.tau_plus)
    am = np.exp(-p.dt / p.tau_minus)
    x = np.zeros(n)                                       # presynaptic trace
    y = np.zeros(n)                                       # postsynaptic trace

    steps = spikes.shape[0]
    for t in range(steps):
        x *= ap
        y *= am
        F = np.nonzero(spikes[t])[0]
        if F.size:
            # depression: for pre i in F, all posts -> columns F, masked
            W[:, F] += (-p.a_minus) * (y[:, None] * A[:, F])
            # potentiation: for post j in F, all pres -> rows F, masked
            W[F, :] += (p.a_plus) * (x[None, :] * A[F, :])
            x[F] += 1.0
            y[F] += 1.0

    w_edges = W[dst, src].astype(np.float64)
    return STDPResult(weights=w_edges, W=W)


# ===========================================================================
# 3b. TRANSPOSE-FREE online pair-STDP — out-edges only (the SPEC §3.5 scheme)
# ===========================================================================
def stdp_online_csr(conn: Connectome, spikes: np.ndarray, p: STDPParams) -> STDPResult:
    """
    Same pair rule as `stdp_dense_reference`, but using ONLY out-edge CSR + per-neuron
    traces (no transpose / no in-edge lists).

    * Depression is applied when the presynaptic neuron fires, walking its out-edges
      and reading the postsynaptic trace y[j]  (step 2 of the contract).
    * Potentiation (which the eager scheme applies at the POST spike over in-edges) is
      DEFERRED: each post spike of j is realized when its presynaptic partner i next
      fires — walking i's out-edges and, for every target j, summing over j's post
      spikes strictly between i's previous and current spike, with x_i reconstructed at
      each post-spike step from i's stored trace. Post spikes after i's last spike are
      realized in a final flush. This reproduces the eager result exactly (final state).
    """
    n = conn.n
    indptr = conn.indptr
    dst = conn.indices.astype(np.int64)
    w = conn.weight.astype(np.float64).copy()

    ap = np.exp(-p.dt / p.tau_plus)
    am = np.exp(-p.dt / p.tau_minus)
    x = np.zeros(n)                       # presynaptic trace (drives potentiation amplitude)
    y = np.zeros(n)                       # postsynaptic trace (drives depression amplitude)

    post_arch = [[] for _ in range(n)]    # per-neuron list of steps it fired (as post)
    last_pre = np.full(n, -1, dtype=np.int64)   # step of each neuron's previous PRE spike
    xhat = np.zeros(n)                    # x[i] captured right AFTER i's last pre-spike increment

    def _potentiation_catchup(i, t_now):
        """Realize potentiation for out-edges of i over post spikes in (last_pre[i], t_now].

        The interval is half-open on the LEFT (exclude i's previous spike step: that
        pairing is Δt=0) and CLOSED on the right (include t_now). A post spike landing
        on i's current spike step still pairs with i's EARLIER spikes via x_i pre-
        increment = xhat_i*ap**(t_now-tp); the coincident (Δt=0) pair is naturally
        excluded because xhat_i predates i's current-step increment.
        """
        tp = last_pre[i]
        if tp < 0 or xhat[i] == 0.0:
            return
        xh = xhat[i]
        for e in range(indptr[i], indptr[i + 1]):
            j = dst[e]
            acc = 0.0
            for s in post_arch[j]:
                if s <= tp:                # strictly after i's previous pre spike
                    continue
                if s > t_now:              # up to and including the current step
                    break                  # archive is chronological
                acc += ap ** (s - tp)      # x_i(s) = xhat_i * ap**(s-tp)  (i silent in-between)
            if acc:
                w[e] += p.a_plus * xh * acc

    steps = spikes.shape[0]
    for t in range(steps):
        x *= ap
        y *= am
        F = np.nonzero(spikes[t])[0]
        if F.size:
            # record this step's spikes FIRST so coincident post spikes (s == t) are
            # visible to the potentiation catch-up below.
            for j in F:
                post_arch[j].append(t)
            for i in F:
                lo, hi = indptr[i], indptr[i + 1]
                js = dst[lo:hi]
                # (2) depression along out-edges, using current post trace
                w[lo:hi] += (-p.a_minus) * y[js]
                # (3, deferred) potentiation catch-up for post spikes since i's last spike
                _potentiation_catchup(i, t)
            # (4) increments + bookkeeping
            x[F] += 1.0
            y[F] += 1.0
            for i in F:
                last_pre[i] = t
                xhat[i] = x[i]

    # end-of-run flush: post spikes after each neuron's last pre spike (interval (last_pre, steps))
    for i in range(n):
        tp = last_pre[i]
        if tp < 0 or xhat[i] == 0.0:
            continue
        xh = xhat[i]
        for e in range(indptr[i], indptr[i + 1]):
            j = dst[e]
            acc = 0.0
            for s in post_arch[j]:
                if s > tp:
                    acc += ap ** (s - tp)
            if acc:
                w[e] += p.a_plus * xh * acc

    return STDPResult(weights=w, W=None)
