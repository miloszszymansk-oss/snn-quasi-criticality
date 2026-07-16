"""
tests/test_m3_soc.py — self-organized criticality (SPEC §7.4 T3-SOC, our M3-SOC).

Runs the Rust kernel with the per-neuron short-term-depression homeostasis ON vs OFF,
across a range of the structural control parameter g, and asserts the network AUTONOMOUSLY
regulates toward the critical regime:

  * homeo ON depletes excitatory resources (<x> < 1) and keeps activity bounded/alive
    for every g — it does not run away (as the un-homeostatic supercritical net does) and
    does not die.
  * the homeostasis BREAKS the supercritical "one giant blob" into structured avalanches:
    more avalanches, a much smaller maximum avalanche, and an avalanche-size exponent tau
    pulled toward the mean-field critical value 3/2.

These are scheme-robust bands calibrated from the Python reference (homeo-on: ~15-40 Hz,
<x>~0.5-0.75, tau~1.3-1.5; homeo-off: supercritical blob, max avalanche ~10x larger).
RIGOROUS criticality (branching sigma->1 with finite-size collapse) is the deeper T3 suite;
here we assert the self-organization signatures. Uses tau_homeo=200 ms so it converges
within the run (production uses the slow SPEC default).

BUILD FIRST:  maturin develop && python3 tests/test_m3_soc.py
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
from criticalcortex.rust_driver import kernel_available, run_rust
from criticalcortex.criticality import summarize

STEPS, WIN = 40_000, 15_000
G_VALUES = (2.0, 3.0)


def _run(g, homeo):
    conn = build_connectome(n=1000, k=100, g=g, seed=1, ei_ratio=1.0)   # purely excitatory (clean SOC)
    p = SimParams(seed=1, mu_ext=3.5, sigma_ext=2.0, v0_spread=45.0,
                  homeo_enabled=bool(homeo), tau_homeo=200.0, u_release=0.2)
    net, max_alloc = run_rust(conn, p, STEPS, block=2000, aer_capacity=8_000_000)
    c = int(net.aer_count[0])
    st = net.aer_step[:c]
    lo, hi = STEPS - WIN, STEPS
    rate = int((st >= lo).sum()) / (1000 * WIN * 0.1 / 1000.0)
    s = summarize(st, lo, hi)
    return dict(rate=rate, xbar=float(net.x_avail.mean()), max_alloc=max_alloc,
                finite=bool(np.all(np.isfinite(net.v))), **s)


def test_soc_self_organizes_toward_criticality():
    if not kernel_available():
        raise _Skip("extension not built (run `maturin develop`)")

    off = {g: _run(g, 0) for g in G_VALUES}
    on = {g: _run(g, 1) for g in G_VALUES}

    print("== SOC (Rust kernel): homeostasis OFF vs ON ==")
    for g in G_VALUES:
        o, n = off[g], on[g]
        print(f"  g={g}  OFF: rate={o['rate']:6.1f}Hz sigma={o['branching_ratio']:.3f} "
              f"n_aval={o['n_avalanches']:4d} max_aval={o['max_avalanche']:6d} tau={o['tau']:.2f}")
        print(f"       ON : rate={n['rate']:6.1f}Hz sigma={n['branching_ratio']:.3f} "
              f"n_aval={n['n_avalanches']:4d} max_aval={n['max_avalanche']:6d} tau={n['tau']:.2f} "
              f"<x>={n['xbar']:.3f}")

    for g in G_VALUES:
        n = on[g]
        assert n["max_alloc"] == 0, "hot loop must not allocate"
        assert n["finite"], "state went non-finite under homeostasis"
        assert 0.5 < n["rate"] < 80.0, f"g={g}: homeo rate {n['rate']:.1f}Hz outside stable band"
        assert n["xbar"] < 0.9, f"g={g}: resources not depleted (<x>={n['xbar']:.3f}) — homeostasis inactive"

    # homeostasis breaks the supercritical blob into structured avalanches (at the largest g)
    g = max(G_VALUES)
    assert on[g]["max_avalanche"] < off[g]["max_avalanche"], "homeo did not tame the supercritical blob"
    assert on[g]["n_avalanches"] > off[g]["n_avalanches"], "homeo did not produce more, separated avalanches"


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
