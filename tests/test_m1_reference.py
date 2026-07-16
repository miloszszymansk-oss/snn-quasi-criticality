"""
tests/test_m1_reference.py — Milestone M1 acceptance suite (SPEC §7.3, T2 subset).

Pins the pure-NumPy references that the M1 Rust kernel must later reproduce:
  * connectome determinism + fixed-in-degree structure,
  * CSR scatter == dense brute force,
  * delay ring delivers EXACTLY at t+d (unit + through-loop, incl. d = d_max),
  * pair-STDP window == analytic exponential,
  * transpose-free online STDP == dense eager STDP (to fp precision).

Runs under pytest AND standalone (`python3 tests/test_m1_reference.py`); the
standalone harness exists only because this environment has no pytest.
Network sizes are kept at N=200-300 so the whole suite runs in ~1 s.
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from criticalcortex.connectome import Connectome, build_connectome, load_kappa
from criticalcortex.reference_scatter import (
    csr_scatter, DelayRing, STDPParams, generate_spike_train,
    stdp_dense_reference, stdp_online_csr,
)

KAPPA = load_kappa()


# ===========================================================================
# T2.0 — connectome determinism + fixed-in-degree structure
# ===========================================================================
def test_connectome_is_deterministic_and_well_formed():
    n, k = 250, 25
    a = build_connectome(n, k, g=1.0, kappa=KAPPA, seed=42)
    b = build_connectome(n, k, g=1.0, kappa=KAPPA, seed=42)
    # same seed => byte-identical CSR
    assert np.array_equal(a.indptr, b.indptr)
    assert np.array_equal(a.indices, b.indices)
    assert np.array_equal(a.weight, b.weight)
    assert np.array_equal(a.delay, b.delay)
    # a different seed changes the graph
    c = build_connectome(n, k, g=1.0, kappa=KAPPA, seed=43)
    assert not np.array_equal(a.indices, c.indices)
    # exact fixed in-degree, M = N*K, no autapses, delays in range, dtype contract
    assert a.m == n * k
    assert np.all(a.in_degrees() == k)
    assert np.all(a.edge_sources() != a.indices)                  # no self-connections
    assert a.delay.min() >= 1 and a.delay.max() <= a.meta["d_max"]
    assert (a.indptr.dtype, a.indices.dtype, a.weight.dtype, a.delay.dtype) == \
           (np.int64, np.int32, np.float32, np.uint16)
    # weight magnitude/sign contract: w0 = g/(K*kappa); inhibitory rows negative
    w0 = 1.0 / (k * KAPPA)
    src = a.edge_sources()
    assert np.allclose(a.weight[~a.is_inhib[src]], np.float32(w0), rtol=1e-5)
    assert np.all(a.weight[a.is_inhib[src]] < 0)


# ===========================================================================
# T2.1 — CSR scatter == dense  W[:, spikes].sum(axis=1)
# ===========================================================================
def test_csr_scatter_matches_dense():
    conn = build_connectome(n=200, k=20, g=1.0, kappa=KAPPA, seed=3)
    W = conn.to_dense()                                   # [post, pre]
    for spikes in ([5, 17, 88], [0], [], list(range(0, 200, 7))):
        spikes = np.array(spikes, dtype=np.int64)
        got = csr_scatter(conn.indptr, conn.indices, conn.weight, spikes, conn.n)
        exp = W[:, spikes].sum(axis=1) if spikes.size else np.zeros(conn.n)
        assert np.allclose(got, exp, atol=1e-6)


# ===========================================================================
# T2.2 — delay ring delivers EXACTLY at t + d
# ===========================================================================
def test_delay_ring_delivers_at_exact_step():
    ring = DelayRing(d_max=32, n=10)
    ring.schedule(target=7, weight=0.5, at_step=12)
    for s in range(12):
        assert ring.drain(s)[7] == 0.0                    # nothing before t+d
    assert ring.drain(12)[7] == 0.5                        # delivered precisely at t+d


def test_delay_ring_delivery_timing_in_loop():
    # A spike scheduled at t0 with delay d must arrive at exactly t0+d — not one
    # step early or late — under the drain-before-schedule loop, for delays up to d_max.
    d_max, n, target, w, t0 = 8, 4, 2, 0.75, 3
    for d in (1, 3, d_max):
        ring = DelayRing(d_max, n)
        received = []
        for t in range(25):
            cur = ring.drain(t)                            # drain current step first
            received.append(cur[target])
            if t == t0:
                ring.schedule(target, w, at_step=t0 + d)   # then schedule the future delivery
        received = np.array(received)
        assert received[t0 + d] == w
        assert np.count_nonzero(received) == 1             # exactly once, at t0+d


# ===========================================================================
# T2.3 — pair-STDP window matches the analytic exponential (both schemes)
# ===========================================================================
def _stdp_window(dt_ms, p: STDPParams):
    """Analytic pair-STDP weight change for lag dt = t_post - t_pre (ms)."""
    if dt_ms > 0:
        return p.a_plus * np.exp(-dt_ms / p.tau_plus)      # pre-before-post: potentiation
    if dt_ms < 0:
        return -p.a_minus * np.exp(dt_ms / p.tau_minus)    # post-before-pre: depression
    return 0.0


def test_stdp_pair_window_matches_analytic():
    p = STDPParams()
    w_init = 1.0
    conn = Connectome.from_edges(n=2, src=[0], dst=[1], weight=[w_init], delay=[1])
    steps, r = 1000, 500
    for lag_ms in (-40, -20, -5, 5, 20, 40):
        lag = int(round(lag_ms / p.dt))
        s = r + lag
        spikes = np.zeros((steps, 2), dtype=bool)
        spikes[r, 0] = True                                # presynaptic neuron 0 at step r
        spikes[s, 1] = True                                # postsynaptic neuron 1 at step s
        dw_dense = stdp_dense_reference(conn, spikes, p).weights[0] - w_init
        dw_csr = stdp_online_csr(conn, spikes, p).weights[0] - w_init
        expected = _stdp_window(lag_ms, p)
        assert np.isclose(dw_dense, expected, rtol=1e-6, atol=1e-12)
        assert np.isclose(dw_csr, expected, rtol=1e-6, atol=1e-12)


# ===========================================================================
# T2.4 — transpose-free online STDP == dense eager STDP (fp precision)
# ===========================================================================
def test_stdp_transpose_free_equals_dense_reference():
    p = STDPParams()
    conn = build_connectome(n=300, k=30, g=1.0, kappa=KAPPA, seed=11)
    spikes = generate_spike_train(conn.n, steps=3000, rate_hz=20.0, dt=p.dt, seed=7)

    w0 = conn.weight.astype(np.float64)
    w_dense = stdp_dense_reference(conn, spikes, p).weights
    w_csr = stdp_online_csr(conn, spikes, p).weights

    # the suite must be non-trivial: STDP actually moved the weights
    assert np.max(np.abs(w_dense - w0)) > 1e-6
    # transpose-free scheme reproduces the eager scheme to fp precision
    assert np.allclose(w_csr, w_dense, rtol=1e-4, atol=1e-9)
    assert np.allclose(w_csr - w0, w_dense - w0, atol=1e-9)   # the STDP *deltas* match


# ===========================================================================
# Standalone harness (pytest ignores __main__).
# ===========================================================================
if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:
            fails += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()

    # diagnostics (not assertions)
    p = STDPParams()
    conn = build_connectome(n=300, k=30, g=1.0, kappa=KAPPA, seed=11)
    spikes = generate_spike_train(conn.n, 3000, 20.0, p.dt, seed=7)
    wd = stdp_dense_reference(conn, spikes, p).weights
    wc = stdp_online_csr(conn, spikes, p).weights
    w0 = conn.weight.astype(np.float64)
    print(f"\nSTDP diagnostic: spikes/neuron≈{spikes.sum() / conn.n:.1f}  "
          f"max|Δw|={np.max(np.abs(wd - w0)):.3e}  "
          f"max|dense-csr|={np.max(np.abs(wd - wc)):.2e}")
    print(f"{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
