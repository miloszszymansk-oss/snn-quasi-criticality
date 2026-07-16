"""
tests/test_soqc_gate.py — prove the SOqC verdict gate is a real CLASSIFIER, not a
rubber stamp calibrated to one run. A gate that passes everything is worthless; this
asserts it PASSES genuine SOqC and FAILS every non-SOqC regime we have seen.

Run:  python3 tests/test_soqc_gate.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validate_m3_criticality import evaluate_soqc          # noqa: E402
from criticalcortex import avalanche_stats as astat        # noqa: E402


def _passed(sigma, rng, tau, R, p, ksp):
    return evaluate_soqc(sigma, rng, tau, R, p, ks_pvalue=ksp)[2]


def test_negative_controls():
    print("SOqC gate — synthetic classifier table (args: σ_MR, range(N), τ, trunc_R, trunc_p, ks_p)")
    # (label, args, expected_pass)
    cases = [
        ("genuine SOqC (σ=0.98, τ=2.7, truncated)",   (0.980, 0.002, 2.70, -11.0, 0.001, 0.02), True),
        ("mean-field CRITICAL (σ=1.0, τ=1.5, pure PL)", (1.000, 0.004, 1.50,  0.00, 0.500, 0.30), False),
        ("near-critical pure-PL (σ=0.99, τ=1.52)",     (0.990, 0.002, 1.52,  0.00, 0.600, 0.40), False),
        ("DEAD / exponential (τ rails to 6)",          (0.500, 0.010, 6.00, 50.00, 0.000, 0.00), False),
        ("RUNAWAY / supercritical (σ=1.3)",            (1.300, 0.010, 2.50, -5.00, 0.010, 0.00), False),
        ("σ drifts with N (range 0.06, not invariant)",(0.980, 0.060, 2.70,-10.00, 0.001, 0.02), False),
        ("pure-PL strictly beats truncated (untrunc.)",(0.980, 0.002, 2.50,  8.00, 0.010, 0.20), False),
        ("too subcritical (σ=0.90)",                   (0.900, 0.005, 2.80, -9.00, 0.001, 0.00), False),
    ]
    ok = True
    for label, args, expect in cases:
        got = _passed(*args)
        flag = "PASS" if got else "FAIL"
        good = (got == expect)
        ok &= good
        print(f"  [{'ok ' if good else 'XX '}] gate={flag:4s} (want {'PASS' if expect else 'FAIL'})  {label}")
    assert ok, "SOqC gate misclassified a control — it is NOT discriminating"
    return ok


def test_real_soqc_data():
    """Feed the ACTUAL saved balanced-E/I avalanche sizes + measured σ_MR(N) through the gate."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "m3_avalanches", "balanced_g3.0_sizes.npy")
    if not os.path.exists(path):
        print("real-data check: skipped (m3_avalanches/balanced_g3.0_sizes.npy not present)")
        return True
    s = np.load(path)
    f = astat.powerlaw_mle(s, xmin="ks", gof=False, ci=False, compare=True, seed=0)
    tr = (f.compare or {}).get("truncated_pl", {})
    sigmas = [0.979, 0.981, 0.979]          # measured N-sweep σ_MR at N=500,1000,2000
    rng = float(np.ptp(sigmas))
    mf, so, passed = evaluate_soqc(0.980, rng, f.tau, tr.get("R"), tr.get("p"), ks_pvalue=0.077)
    print(f"real SOqC data: τ={f.tau:.3f}  trunc R={tr.get('R'):+.2f} p={tr.get('p'):.3f}  "
          f"σ_range={rng:.4f}  -> SOqC gate {'PASS' if passed else 'FAIL'}")
    for name, ok in so.items():
        print(f"    [{'PASS' if ok else 'FAIL'}]  {name}")
    assert passed, "gate rejected real SOqC data"
    # and confirm the mean-field reference correctly FAILS on the same data (honesty check)
    assert not all(mf.values()), "mean-field SOC should FAIL on SOqC data (τ≠1.5, σ<1)"
    return True


if __name__ == "__main__":
    a = test_negative_controls()
    print()
    b = test_real_soqc_data()
    print("\n" + ("ALL PASS — gate discriminates SOqC from non-SOqC" if (a and b) else "FAILURES ABOVE"))
    sys.exit(0 if (a and b) else 1)
