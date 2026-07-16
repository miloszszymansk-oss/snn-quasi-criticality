"""
criticalcortex.spatial_connectome — distance-embedded connectome variant (M5 spatial money-shot).

Same OUTPUT contract as `connectome.build_connectome` (out-edge CSR: indptr/indices/weight/delay
+ is_inhib, identical dtypes) so it feeds the zero-allocation Rust kernel with NO kernel change —
the kernel only ever reads CSR and is topology-agnostic. What differs is HOW edges are chosen:

  * neurons are embedded on a Fibonacci sphere (the exact layout the frontend renders);
  * each postsynaptic neuron still draws EXACTLY K inputs (fixed in-degree) and the exact E/I
    split (k_exc excitatory + k_inh inhibitory), but partners are sampled with probability
    ∝ exp(-‖xᵢ−xⱼ‖ / locality) — a distance kernel — via the Gumbel-top-k trick;
  * inhibitory identity is assigned to a RANDOM interspersed subset (NOT the last indices —
    Fibonacci index maps monotonically to latitude, which would clump all inhibition at one pole
    and destroy local E/I balance);
  * axonal delays optionally scale with distance (finite conduction velocity) so cascades
    propagate as visible spatial waves instead of instantaneously.

Weight scaling (w0 = g/(K·κ)), Dale signs, and fixed in-degree are preserved, so the linearized
control parameter g is unchanged. Spatial CORRELATIONS may still shift the realized branching
ratio, so re-run validate_m3_criticality.py on this variant to confirm/retune SOqC.
"""

from __future__ import annotations

import numpy as np

from .connectome import Connectome, DT_INDPTR, DT_INDEX, DT_WEIGHT, DT_DELAY, load_kappa


def fibonacci_sphere(n: int) -> np.ndarray:
    """N points on the unit sphere — IDENTICAL formula to static/app.js so backend connectivity
    and frontend rendering share one geometry. Returns (N,3) float64."""
    i = np.arange(n, dtype=np.float64)
    z = 1.0 - 2.0 * (i + 0.5) / n                       # cos(phi); index -> latitude
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    th = np.pi * (1.0 + np.sqrt(5.0)) * (i + 0.5)       # golden-angle spiral
    return np.stack([r * np.cos(th), r * np.sin(th), z], axis=1)


def _gumbel_top_k(log_w: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Weighted sampling of k items WITHOUT replacement, prob ∝ exp(log_w): add i.i.d. Gumbel
    noise to log-weights and take the k largest (Vieira 2014). Excluded items = log_w −inf."""
    g = -np.log(-np.log(rng.random(log_w.shape)))
    keys = log_w + g
    if k >= keys.size:
        return np.argsort(-keys)
    idx = np.argpartition(-keys, k)[:k]
    return idx


def build_spatial_connectome(
    n: int, k: int, *, g: float = 1.0, kappa: float | None = None, seed: int = 0,
    ei_ratio: float = 0.8, inh_ratio: float = 4.0, d_min: int = 1, d_max: int = 20,
    locality: float = 0.35, distance_delays: bool = True,
) -> Connectome:
    """Distance-embedded fixed-in-degree connectome. `locality` is the exponential length scale
    (in unit-sphere chord units, ~[0,2]); large locality → back toward a random graph."""
    if not (1 <= k <= n - 1):
        raise ValueError(f"require 1 <= k <= n-1, got k={k}, n={n}")
    if kappa is None:
        kappa = load_kappa()
    rng = np.random.default_rng(seed)
    pos = fibonacci_sphere(n)

    # interspersed inhibitory subset (NOT contiguous indices — keeps local E/I balance)
    n_exc = int(round(ei_ratio * n))
    is_inhib = np.zeros(n, dtype=bool)
    inh_sel = rng.choice(n, size=n - n_exc, replace=False)
    is_inhib[inh_sel] = True
    exc_ids = np.flatnonzero(~is_inhib)
    inh_ids = np.flatnonzero(is_inhib)
    k_inh = int(round((1.0 - ei_ratio) * k))
    k_exc = k - k_inh
    w0 = g / (k * kappa)

    src = np.empty(n * k, dtype=np.int64)
    dst = np.empty(n * k, dtype=np.int64)
    dist = np.empty(n * k, dtype=np.float64)             # per-edge distance (for delays / viz)
    inv_loc = 1.0 / max(locality, 1e-6)

    for j in range(n):
        d = np.linalg.norm(pos - pos[j], axis=1)         # chord distance to every neuron
        logw = -d * inv_loc
        logw[j] = -np.inf                                # no autapse
        # distance-weighted draw of the exact E/I split
        se = exc_ids[_gumbel_top_k(logw[exc_ids], k_exc, rng)]
        si = inh_ids[_gumbel_top_k(logw[inh_ids], k_inh, rng)]
        s = np.concatenate([se, si])
        blk = slice(j * k, (j + 1) * k)
        src[blk] = s
        dst[blk] = j
        dist[blk] = d[s]

    # signed weights (Dale's law: sign set by PRESYNAPTIC neuron)
    w = np.where(is_inhib[src], -inh_ratio * w0, w0).astype(DT_WEIGHT)

    # delays: distance-proportional (finite conduction velocity) or uniform
    if distance_delays:
        span = max(float(dist.max()), 1e-9)
        delay = np.clip(np.round(d_min + (dist / span) * (d_max - d_min)),
                        d_min, d_max).astype(DT_DELAY)
    else:
        delay = rng.integers(d_min, d_max + 1, size=n * k).astype(DT_DELAY)

    # materialize out-edge CSR (stable sort by source) — identical layout to build_connectome
    order = np.argsort(src, kind="stable")
    src_s = src[order]
    indptr = np.zeros(n + 1, dtype=DT_INDPTR)
    indptr[1:] = np.cumsum(np.bincount(src_s, minlength=n))
    conn = Connectome(
        n=n, k=k, indptr=indptr,
        indices=dst[order].astype(DT_INDEX),
        weight=w[order], delay=delay[order],
        is_inhib=is_inhib,
        meta=dict(g=g, kappa=float(kappa), w0=float(w0), ei_ratio=ei_ratio, inh_ratio=inh_ratio,
                  seed=seed, d_min=d_min, d_max=d_max, spatial=True, locality=locality,
                  distance_delays=bool(distance_delays)),
    )
    conn._positions = pos                                 # attach for the engine/frontend
    conn._edge_dist = dist[order]
    return conn


def visualization_edges(conn: Connectome, max_edges: int = 6000, exc_only: bool = True):
    """Return the SHORTEST `max_edges` synapses as (i,j) index pairs — the local mesh that makes
    spatial structure legible. Prefers excitatory edges (the ones that propagate avalanches)."""
    src = conn.edge_sources()
    dst = conn.indices.astype(np.int64)
    dist = getattr(conn, "_edge_dist", None)
    if dist is None:
        pos = getattr(conn, "_positions")
        dist = np.linalg.norm(pos[src] - pos[dst], axis=1)
    mask = (~conn.is_inhib[src]) if exc_only else np.ones(src.size, bool)
    s, dd, di = src[mask], dst[mask], dist[mask]
    keep = np.argsort(di)[:max_edges]
    pairs = np.stack([s[keep], dd[keep]], axis=1).astype(np.int32)
    return pairs
