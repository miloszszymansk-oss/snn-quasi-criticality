"""
tests/test_m4_long_horizon.py — Milestone M4 long-horizon verification.

Runs the compiled Rust kernel for a long run (10_000 steps) and asserts the network is
physically stable and the hot loop is allocation-free:

  * StepStats.alloc_bytes == 0 on EVERY block (the zero-allocation contract, SPEC §2.2).
  * v and weights stay finite (no blow-up).
  * activity is neither silent nor saturated, and is SUSTAINED (no dying out) — i.e. the
    pair-STDP updates keep the network in a stable operating regime rather than running
    away to LTP-saturation or LTD-silence.

Thresholds are loose stability bands (calibrated against the Python reference: ~7.5 Hz,
weights in ~[-1.5, 0.5]); they are intentionally scheme-robust, since the Rust transpose-
free STDP diverges from the eager reference at long horizons. RIGOROUS criticality
(branching ratio -> 1, power-law avalanches) is the separate T3 suite — here we print a
lightweight characterization and only assert stability.

BUILD FIRST:  maturin develop && python3 tests/test_m4_long_horizon.py
Skips cleanly if the extension is not built.
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class _Skip(Exception):
    pass


from criticalcortex.connectome import build_connectome
from criticalcortex.simulation import SimParams
from criticalcortex.rust_driver import kernel_available, allocate_kernel_state, step_kernel

N, K, G, SEED, STEPS, BLOCK = 1000, 100, 1.0, 42, 10_000, 1000


def test_long_horizon_stable_and_zero_alloc():
    if not kernel_available():
        raise _Skip("extension not built (run `maturin develop`)")

    conn = build_connectome(n=N, k=K, g=G, seed=SEED)
    p = SimParams(seed=SEED)
    state = allocate_kernel_state(conn, p)
    net = state["net"]

    # stream the run in blocks; the zero-allocation contract must hold on EVERY block
    max_alloc, done = 0, 0
    while done < STEPS:
        nb = min(BLOCK, STEPS - done)
        stats = step_kernel(state, done, nb)
        assert stats.alloc_bytes == 0, f"heap allocation in hot loop at step {done}"
        max_alloc = max(max_alloc, int(stats.alloc_bytes))
        done += nb

    c = int(net.aer_count[0])
    st = net.aer_neuron[:c], net.aer_step[:c]
    steps_arr = net.aer_step[:c]
    w = net.weight
    rate = c / (N * STEPS * p.dt / 1000.0)

    # ---- characterization (informational) ----
    def band(lo, hi):
        return int(((steps_arr >= lo) & (steps_arr < hi)).sum()) / (N * (hi - lo) * p.dt / 1000.0)

    early, late = band(1000, 3000), band(STEPS - 2000, STEPS)
    print("== M4 long-horizon (Rust kernel, 10_000 steps) ==")
    print(f"max alloc_bytes    : {max_alloc}   (zero-allocation hot loop)")
    print(f"spikes             : {c}   mean rate {rate:.2f} Hz")
    print(f"rate early/late    : {early:.1f} Hz -> {late:.1f} Hz")
    print(f"weight w0={conn.meta['w0']:.4f}: mean {w.mean():.4f} std {w.std():.4f} "
          f"min {w.min():.3f} max {w.max():.3f}")
    print(f"active neurons     : {int(np.unique(net.aer_neuron[:c]).size)}/{N}")

    # ---- stability assertions (scheme-robust bands) ----
    assert max_alloc == 0
    assert np.all(np.isfinite(net.v)) and np.all(np.isfinite(w)), "state went non-finite"
    assert 0.5 < rate < 100.0, f"rate {rate:.2f} Hz outside stable band"
    assert late > 0.2 * early, "activity collapsing toward silence"
    assert np.abs(w).max() < 20.0, "weights running away (STDP unstable)"
    assert int(np.unique(net.aer_neuron[:c]).size) > 100, "too few participating neurons"


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
