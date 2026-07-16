"""
tests/test_m3_exact_raster.py — Milestone M3 short-horizon exact raster match.

Initializes the Rust kernel with the IDENTICAL config as the golden master
(N=1000, K=100, g=1.0, seed=42, deterministic v0-spread + hash drive), runs the Rust
`step_block` for exactly 50 steps, and asserts the emitted (step, neuron) raster is a
bit-for-bit match of the first 50 steps of reference/golden_master.aer.

Why 50 steps: this is a chaotic f32 system, so numpy-f32 and Rust-f32 rasters can only
agree on a SHORT horizon (before any 1-ULP difference amplifies); long horizons are
validated statistically by T3. STDP does not affect the first-50 raster (verified in
Python: STDP-on == STDP-off), so this test isolates the core loop: hash drive, Izhikevich
integration, synaptic decay, delay-ring drain, threshold/reset, and the recurrent scatter.

BUILD FIRST:
    maturin develop && python3 tests/test_m3_exact_raster.py
Regenerate the golden master (`python3 generate_golden_master.py`) if the reference
simulator changed. Skips cleanly if the extension is not built.
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class _Skip(Exception):
    pass


try:
    from criticalcortex import _kernel
    _HAVE = True
    _ERR = None
except Exception as _e:
    _HAVE = False
    _ERR = _e

from criticalcortex.connectome import build_connectome
from criticalcortex.simulation import SimParams, build_network

REF = os.path.join(_ROOT, "reference", "golden_master.aer")
N, K, G, SEED, STEPS = 1000, 100, 1.0, 42, 50
HIST_CAP = 64


def _golden_first(nsteps):
    with open(REF, "rb") as f:
        data = f.read()
    rec = np.frombuffer(data[64:], dtype=[("s", "<u4"), ("n", "<u4")])
    m = rec["s"] < nsteps
    return list(zip(rec["s"][m].tolist(), rec["n"][m].tolist()))


def test_m3_short_horizon_exact_raster():
    if not _HAVE:
        raise _Skip(f"extension not built ({_ERR})")

    conn = build_connectome(n=N, k=K, g=G, seed=SEED)
    p = SimParams(seed=SEED)
    net = build_network(conn, p)                         # applies the deterministic v0-spread

    # transpose-free potentiation state (Rust-only)
    last_pre = np.full(N, -1, np.int64)
    xhat = np.zeros(N, np.float32)
    post_hist = np.full(N * HIST_CAP, -1, np.int64)
    post_cnt = np.zeros(N, np.int64)

    # decay factors precomputed to f32 exactly as the reference does
    a_syn = float(np.float32(np.exp(-p.dt / p.tau_syn)))
    a_stdp = float(np.float32(np.exp(-p.dt / p.tau_stdp)))
    kp = _kernel.KernelParams(
        dt=p.dt, a=p.a, b=p.b, c=p.c, d=p.d, v_peak=p.v_peak,
        a_plus=p.a_plus, a_minus=p.a_minus, a_syn=a_syn, a_stdp=a_stdp,
        mu_ext=p.mu_ext, sigma_ext=p.sigma_ext, refractory_steps=p.refractory_steps, seed=SEED,
    )

    aer_step = np.zeros(1_000_000, np.uint32)
    aer_neuron = np.zeros(1_000_000, np.uint32)
    aer_count = np.zeros(1, np.int64)

    stats = _kernel.step_block(
        net.v, net.u, net.g_exc, net.g_inh, net.refrac, net.t_last,
        net.conn.indptr, net.conn.indices, net.weight, net.conn.delay,
        net.ring.reshape(-1), net.x_pre, net.x_post,       # ring flattened (zero-copy view)
        last_pre, xhat, post_hist, post_cnt,
        net.x_avail, net.homeo_keep,                       # SOC state (homeo off => no effect)
        kp, 0, STEPS,
        aer_step, aer_neuron, aer_count,
    )

    c = int(aer_count[0])
    rust = list(zip(aer_step[:c].tolist(), aer_neuron[:c].tolist()))
    gold = _golden_first(STEPS)

    assert stats.alloc_bytes == 0, "hot loop must not allocate"
    assert len(rust) == len(gold), f"spike count differs: Rust={len(rust)} golden={len(gold)}"
    if rust != gold:
        for k, (r, g) in enumerate(zip(rust, gold)):
            if r != g:
                raise AssertionError(f"first divergence at spike #{k}: Rust={r} golden={g}")
    assert rust == gold


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    fails = skips = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except _Skip as s:
            skips += 1
            print(f"SKIP  {t.__name__}: {s}")
        except Exception:
            fails += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    if skips and not fails:
        print("\nBuild the kernel first:  maturin develop  (then re-run).")
    print(f"{len(tests) - fails - skips} passed, {skips} skipped, {fails} failed")
    sys.exit(1 if fails else 0)
