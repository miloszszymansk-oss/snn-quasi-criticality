"""
tests/test_m0_numerics.py — Milestone M0 acceptance suite (SPEC §7.2, T1).

The three T1 invariants of the numeric core. Self-contained: imports only the M0
reference module (no network, no Rust, no analysis pipeline — those arrive in M1+).

Runs under pytest (functions are auto-discovered) AND standalone:
    python3 tests/test_m0_numerics.py
The standalone harness exists only because this environment has no pytest; it does
not alter the tests. Regenerate the golden master first with `python generate_fI_ref.py`.
"""

import os
import sys

import numpy as np

# Make the repo importable whether launched via pytest or as a standalone script.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from criticalcortex.numerics import measure_fI, izh_step, izh_rhs, simulate_decay

REF = os.path.join(_ROOT, "reference", "izh_rs_fI.npz")


# ===========================================================================
# T1.1 — f–I golden master. Locks the EXACT numerics the Rust kernel must match.
# (Regression/golden-master: the same npz will later gate the M1 Rust port.)
# ===========================================================================
def test_izhikevich_fI_matches_reference():
    ref = np.load(REF)
    f = measure_fI(a=.02, b=.2, c=-65, d=8, I=ref["I"], dt=0.1, T_ms=1000)
    assert np.allclose(f, ref["f"], atol=0.5)   # Hz


# ===========================================================================
# T1.2 — the exponential synapse is the EXACT one-step propagator of g'=-g/tau.
# ===========================================================================
def test_exponential_synapse_is_exact():
    tau, dt, g0 = 5.0, 0.1, 1.0
    g = simulate_decay(g0, tau, dt, n=1000)
    t = np.arange(1, 1001) * dt
    assert np.max(np.abs(g - g0 * np.exp(-t / tau))) < 1e-6


# ===========================================================================
# T1.3 — global order 1 via Richardson extrapolation, checked against an
#        INDEPENDENT RK4 reference on the SMOOTH subthreshold flow.
#
# The order analysis requires a smooth trajectory: the spike reset is a C^0
# discontinuity that would collapse the observed order to 1 for the wrong reason.
# We therefore drive with I below rheobase (measured onset I≈3.8) and integrate for
# a window over which the neuron provably never spikes. I=2.0 stays subthreshold to
# t≈60 ms (v_max≈-65.6), so the 6 ms measurement window has ~10x margin.
# ===========================================================================
_A, _B, _C, _D = 0.02, 0.20, -65.0, 8.0
_I_SUB = 2.0            # < rheobase => smooth, spike-free trajectory (10x time margin)
_T_END = 6.0           # ms — measured mid-transient (fixed point not yet reached)
_V0, _U0 = -70.0, 0.20 * -70.0


def _rk4_ref(dt, t_end, I):
    """Classical RK4 on the smooth Izhikevich ODE (order 4). At dt=1e-4 this is
    ~machine-exact and, being a different method, an INDEPENDENT reference (not a
    finer copy of the scheme under test)."""
    n = int(round(t_end / dt))
    v, u = _V0, _U0
    for _ in range(n):
        k1v, k1u = izh_rhs(v, u, I, _A, _B)
        k2v, k2u = izh_rhs(v + 0.5 * dt * k1v, u + 0.5 * dt * k1u, I, _A, _B)
        k3v, k3u = izh_rhs(v + 0.5 * dt * k2v, u + 0.5 * dt * k2u, I, _A, _B)
        k4v, k4u = izh_rhs(v + dt * k3v, u + dt * k3u, I, _A, _B)
        v += (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        u += (dt / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
    return v, u


_VREF, _UREF = _rk4_ref(1e-4, _T_END, _I_SUB)   # essentially exact reference state


def _traj_error(dt):
    """L2 state error |(v,u)(T) - reference| for the half-step Izhikevich scheme.
    Asserts the trajectory stays on the smooth (spike-free) branch."""
    n = int(round(_T_END / dt))
    v = np.float64(_V0)
    u = np.float64(_U0)
    for _ in range(n):
        v, u, spiked = izh_step(v, u, _I_SUB, _A, _B, _C, _D, dt)
        assert not bool(spiked)      # guarantee we measure the smooth-flow order
    return float(np.hypot(float(v) - _VREF, float(u) - _UREF))


def test_integrator_convergence_order():
    e1, e2 = _traj_error(dt=0.1), _traj_error(dt=0.05)
    assert 1.7 < e1 / e2 < 2.3       # halving dt halves the error => global order 1


# ===========================================================================
# Standalone harness (no pytest in this environment). Pytest ignores __main__.
# ===========================================================================
if __name__ == "__main__":
    import sys, traceback

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
    # extra diagnostics (not assertions)
    print(f"\nT1.3 diagnostic: err(0.1)={_traj_error(0.1):.4e}  err(0.05)={_traj_error(0.05):.4e}  "
          f"ratio={_traj_error(0.1) / _traj_error(0.05):.4f}  (order-1 target 2.0)")
    print(f"{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
