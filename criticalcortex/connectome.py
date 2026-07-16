"""
criticalcortex.connectome — deterministic connectome builder (M1 reference).

Builds a random sparse network with FIXED IN-DEGREE K (each postsynaptic neuron
receives exactly K distinct presynaptic inputs, no autapses, no multi-edges) and
materializes it as the out-edge CSR layout the SPEC §3.2 kernel consumes:

    indptr  : int64 [N+1]   row i spans indices[indptr[i]:indptr[i+1]]  (edges FROM i)
    indices : int32 [M]     postsynaptic target j of each edge          (M = N*K)
    weight  : float32 [M]   synaptic weight (signed by Dale's law)
    delay   : uint16 [M]    axonal delay in timesteps, in [d_min, d_max]

Weights are scaled with the M0-calibrated single-neuron gain kappa so that the
linearized branching ratio equals the control parameter g (SPEC §4.4):

    w0 = g / (K * kappa)         (excitatory magnitude)
    w_exc = +w0 ,  w_inh = -inh_ratio * w0

Everything is a pure deterministic function of `seed`. The METIS cache-locality
permutation (SPEC §3.4) is intentionally SKIPPED here — this is the correctness
baseline; the reordering is a performance-phase concern that does not change any
edge, weight, or delay, only their storage order.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

# dtypes are part of the SPEC §3.2 memory contract.
DT_INDPTR = np.int64
DT_INDEX = np.int32
DT_WEIGHT = np.float32
DT_DELAY = np.uint16


def load_kappa(path: str | None = None) -> float:
    """Read the M0-calibrated gain kappa from reference/izh_rs_fI.npz (SPEC §4.4)."""
    if path is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "reference", "izh_rs_fI.npz")
    with np.load(path) as ref:
        return float(ref["kappa"])


@dataclass
class Connectome:
    """Immutable out-edge CSR connectome plus the metadata needed to reproduce it."""

    n: int
    k: int
    indptr: np.ndarray      # int64  [N+1]
    indices: np.ndarray     # int32  [M]  postsynaptic targets
    weight: np.ndarray      # f32    [M]
    delay: np.ndarray       # u16    [M]
    is_inhib: np.ndarray    # bool   [N]  presynaptic sign (Dale's law)
    meta: dict = field(default_factory=dict)

    # ---- derived views -----------------------------------------------------
    @property
    def m(self) -> int:
        """Total synapse count M = N*K."""
        return int(self.indices.size)

    def edge_sources(self) -> np.ndarray:
        """Presynaptic neuron id of every edge (the implicit CSR row index)."""
        return np.repeat(np.arange(self.n, dtype=np.int64), np.diff(self.indptr))

    def out_degrees(self) -> np.ndarray:
        """Out-degree per neuron (variable under fixed in-degree)."""
        return np.diff(self.indptr).astype(np.int64)

    def in_degrees(self) -> np.ndarray:
        """In-degree per neuron (exactly K by construction)."""
        return np.bincount(self.indices, minlength=self.n).astype(np.int64)

    # ---- dense materialisations (reference / testing only) -----------------
    def to_dense(self) -> np.ndarray:
        """Dense weight matrix W[post, pre] (float64). Non-edges are 0."""
        W = np.zeros((self.n, self.n), dtype=np.float64)
        W[self.indices, self.edge_sources()] = self.weight
        return W

    def adjacency(self) -> np.ndarray:
        """Boolean adjacency A[post, pre] (True where a synapse exists)."""
        A = np.zeros((self.n, self.n), dtype=bool)
        A[self.indices, self.edge_sources()] = True
        return A

    # ---- explicit-edge factory (for unit tests) ----------------------------
    @classmethod
    def from_edges(cls, n, src, dst, weight, delay, is_inhib=None):
        """Build an out-edge CSR connectome from explicit (src -> dst) edge lists."""
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        weight = np.asarray(weight, dtype=DT_WEIGHT)
        delay = np.asarray(delay, dtype=DT_DELAY)
        order = np.argsort(src, kind="stable")           # group edges by source
        src_s = src[order]
        indptr = np.zeros(n + 1, dtype=DT_INDPTR)
        indptr[1:] = np.cumsum(np.bincount(src_s, minlength=n))
        if is_inhib is None:
            is_inhib = np.zeros(n, dtype=bool)
        return cls(
            n=int(n), k=-1,
            indptr=indptr,
            indices=dst[order].astype(DT_INDEX),
            weight=weight[order].astype(DT_WEIGHT),
            delay=delay[order].astype(DT_DELAY),
            is_inhib=np.asarray(is_inhib, dtype=bool),
            meta={"origin": "from_edges"},
        )


def build_connectome(
    n: int,
    k: int,
    *,
    g: float = 1.0,
    kappa: float | None = None,
    seed: int = 0,
    ei_ratio: float = 0.8,
    inh_ratio: float = 4.0,
    d_min: int = 1,
    d_max: int = 20,
) -> Connectome:
    """
    Deterministic fixed-in-degree random connectome in out-edge CSR form.

    Parameters
    ----------
    n, k        : neuron count and (exact) in-degree per neuron.
    g           : control parameter == target branching ratio (SPEC §4.4).
    kappa       : single-neuron gain; if None, loaded from the M0 reference npz.
    seed        : RNG seed — the entire graph is a pure function of this.
    ei_ratio    : fraction excitatory (first n_exc neurons are excitatory).
    inh_ratio   : inhibitory weight magnitude factor (E/I balance; distinct from g).
    d_min,d_max : axonal delay bounds (timesteps); delays sampled uniform in [d_min,d_max].

    Notes
    -----
    * Sampling uses the "skip-self" bijection: draw K distinct sources from {0..N-2}
      and map any value >= j to value+1, giving K distinct sources in {0..N-1}\{j}.
      This is the O(N*K) reference builder; the production builder (M1 Rust) will
      vectorize it, but the resulting distribution and seeding contract are fixed here.
    * Requires 1 <= K <= N-1 and 1 <= d_min <= d_max < 2**16.
    """
    if not (1 <= k <= n - 1):
        raise ValueError(f"require 1 <= k <= n-1, got k={k}, n={n}")
    if not (1 <= d_min <= d_max < 2 ** 16):
        raise ValueError(f"require 1 <= d_min <= d_max < 65536, got [{d_min},{d_max}]")

    if kappa is None:
        kappa = load_kappa()
    rng = np.random.default_rng(seed)

    n_exc = int(round(ei_ratio * n))
    is_inhib = np.zeros(n, dtype=bool)
    is_inhib[n_exc:] = True                    # contiguous: [0,n_exc) excitatory, rest inhibitory

    w0 = g / (k * kappa)                        # excitatory weight magnitude (SPEC §4.4)

    # --- sample fixed in-degree sources per postsynaptic neuron j ---
    src = np.empty(n * k, dtype=np.int64)
    dst = np.empty(n * k, dtype=np.int64)
    for j in range(n):
        s = rng.choice(n - 1, size=k, replace=False)   # K distinct in {0..N-2}
        s = s + (s >= j)                               # skip-self bijection -> {0..N-1}\{j}
        blk = slice(j * k, (j + 1) * k)
        src[blk] = s
        dst[blk] = j

    # --- signed weights (Dale's law: sign set by PRESYNAPTIC neuron) ---
    w = np.where(is_inhib[src], -inh_ratio * w0, w0).astype(DT_WEIGHT)

    # --- axonal delays (>= 1 step: a spike never lands in the step it is emitted) ---
    delay = rng.integers(d_min, d_max + 1, size=n * k).astype(DT_DELAY)

    # --- materialize out-edge CSR: stable-sort edges by source ---
    order = np.argsort(src, kind="stable")
    src_s = src[order]
    indptr = np.zeros(n + 1, dtype=DT_INDPTR)
    indptr[1:] = np.cumsum(np.bincount(src_s, minlength=n))

    return Connectome(
        n=n, k=k,
        indptr=indptr,
        indices=dst[order].astype(DT_INDEX),
        weight=w[order],
        delay=delay[order],
        is_inhib=is_inhib,
        meta=dict(g=g, kappa=float(kappa), w0=float(w0), ei_ratio=ei_ratio,
                  inh_ratio=inh_ratio, seed=seed, d_min=d_min, d_max=d_max),
    )
