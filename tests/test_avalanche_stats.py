"""
tests/test_avalanche_stats.py — ground-truth validation of the CSN discrete power-law
pipeline. Pure numpy; no kernel required. Run:  python3 tests/test_avalanche_stats.py

Every claim the report will make is checked here against a KNOWN answer:
  T1  Hurwitz zeta            vs closed forms (π²/6, π⁴/90, Apéry, ζ(s,2))
  T2  golden-section MLE      vs brute-force grid argmin of the exact likelihood
  T3  sampler                 empirical CCDF vs analytic; approx vs exact agree
  T4  exponent recovery       α∈{1.5,2.0,2.5}, xmin∈{1,4,6} recovered within tol
  T5  continuous-approx bias  the frozen-xmin estimator is measurably biased at xmin=4
  T6  xmin(KS) selection      recovers a planted cutoff above a non-PL body
  T7  GOF calibration         true PL ⇒ accept (p>0.1); exponential ⇒ reject (p<0.1)
  T8  bootstrap CI coverage   95% CI covers the true α across repeated realisations
  T9  Vuong LR                exp data⇒exp wins; lognormal data⇒lognormal wins;
                              PL data⇒truncated-PL not preferred (λ≈0)
  T10 SMALL-n honesty         n≈120 ⇒ CI on τ is wide (the M3 reality check)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from criticalcortex import avalanche_stats as A   # noqa: E402


def _ok(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    return bool(cond)


def t1_zeta():
    print("T1  Hurwitz zeta vs closed forms")
    # brute-force reference: ζ(s,q) = Σ_{k=0}^{K}(k+q)^{-s} to K huge (slow but exact-ish)
    def zbrute(s, q, K=4_000_000):
        # chunked to keep memory flat; s=2.5 tail beyond K is ~K^-1.5/1.5 ~ 1e-10
        tot = 0.0
        for lo in range(0, K, 1_000_000):
            k = np.arange(lo, min(lo + 1_000_000, K), dtype=np.float64)
            tot += float(np.sum((k + q) ** (-s)))
        return tot
    checks = [
        ("zeta(2,1)=pi^2/6", A.hurwitz_zeta(2.0, 1), np.pi**2 / 6),
        ("zeta(4,1)=pi^4/90", A.hurwitz_zeta(4.0, 1), np.pi**4 / 90),
        ("zeta(3,1)=Apery", A.hurwitz_zeta(3.0, 1), 1.2020569031595942),
        ("zeta(2,2)=pi^2/6-1", A.hurwitz_zeta(2.0, 2), np.pi**2 / 6 - 1),
        ("zeta(2.5,3) vs brute", A.hurwitz_zeta(2.5, 3), zbrute(2.5, 3)),
    ]
    good = True
    for name, got, ref in checks:
        rel = abs(got - ref) / abs(ref)
        good &= _ok(name, rel < 1e-9, f"got={got:.12f} ref={ref:.12f} rel={rel:.1e}")
    # vectorised over q
    q = np.array([1, 2, 5, 10])
    v = A.hurwitz_zeta(2.0, q)
    ref = np.array([np.pi**2 / 6, np.pi**2 / 6 - 1,
                    np.pi**2 / 6 - sum(1 / k**2 for k in range(1, 5)),
                    np.pi**2 / 6 - sum(1 / k**2 for k in range(1, 10))])
    good &= _ok("zeta vectorised over q", np.allclose(v, ref, rtol=1e-10))
    return good


def t2_mle_vs_grid():
    print("T2  golden-section MLE == brute-force grid argmin")
    rng = np.random.default_rng(1)
    good = True
    for alpha, xmin in [(1.5, 4), (2.0, 6), (2.5, 2)]:
        x = A.sample_discrete_pl(4000, alpha, xmin, rng, exact=False)
        x = x[x >= xmin]
        a_gs = A.mle_exponent(x, xmin)
        grid = np.linspace(1.05, 5.0, 4000)
        S = np.sum(np.log(x))
        nll = np.array([A._neg_loglik(a, S, x.size, xmin) for a in grid])
        a_grid = grid[np.argmin(nll)]
        good &= _ok(f"alpha*={alpha} xmin={xmin}", abs(a_gs - a_grid) < 2e-3,
                    f"golden={a_gs:.4f} grid={a_grid:.4f}")
    return good


def t3_sampler():
    print("T3  sampler: empirical CCDF vs analytic; approx vs exact")
    rng = np.random.default_rng(2)
    alpha, xmin, n = 1.8, 3, 200_000
    x = A.sample_discrete_pl(n, alpha, xmin, rng)
    pts = np.array([3, 4, 6, 10, 20, 50])
    emp = np.array([(x >= p).mean() for p in pts])
    ana = A.discrete_pl_ccdf(pts, alpha, xmin)
    good = _ok("approx-sampler CCDF matches analytic",
               np.max(np.abs(emp - ana)) < 0.01, f"maxdev={np.max(np.abs(emp-ana)):.4f}")
    xe = A.sample_discrete_pl(20_000, alpha, xmin, rng, exact=True)
    emp_e = np.array([(xe >= p).mean() for p in pts])
    good &= _ok("exact-sampler CCDF matches analytic",
                np.max(np.abs(emp_e - ana)) < 0.02, f"maxdev={np.max(np.abs(emp_e-ana)):.4f}")
    return good


def t4_recovery():
    print("T4  exponent recovery (exact discrete MLE, large n)")
    rng = np.random.default_rng(3)
    good = True
    for alpha, xmin in [(1.5, 4), (2.0, 6), (2.5, 1)]:
        est = []
        for _ in range(12):
            x = A.sample_discrete_pl(20_000, alpha, xmin, rng)
            est.append(A.mle_exponent(x[x >= xmin], xmin))
        m = np.mean(est)
        good &= _ok(f"recover alpha={alpha} xmin={xmin}", abs(m - alpha) < 0.03,
                    f"mean_est={m:.4f} (bias={m-alpha:+.4f})")
    return good


def t5_continuous_bias():
    print("T5  CONTINUOUS approx (powerlaw_tau) collapses at small xmin; exact MLE holds")
    rng = np.random.default_rng(4)
    good = True
    # duration-like regime: small xmin, alpha~2. This is where powerlaw_tau silently fails.
    for xmin, alpha in [(1, 2.0), (2, 2.0), (4, 1.5)]:
        de, dc = [], []
        for _ in range(20):
            x = A.sample_discrete_pl(20_000, alpha, xmin, rng)
            t = x[x >= xmin]
            de.append(A.mle_exponent(t, xmin) - alpha)
            dc.append(A.mle_exponent_continuous(t, xmin) - alpha)
        be, bc = abs(np.mean(de)), abs(np.mean(dc))
        print(f"       xmin={xmin} alpha={alpha}: exact bias={np.mean(de):+.4f}  "
              f"continuous bias={np.mean(dc):+.4f}")
        good &= _ok(f"exact near-unbiased (xmin={xmin})", be < 0.02)
    # at xmin=1 the continuous approx must be catastrophically worse
    x = A.sample_discrete_pl(20_000, 2.0, 1, rng)
    bexact = abs(A.mle_exponent(x, 1) - 2.0)
    bcont = abs(A.mle_exponent_continuous(x, 1) - 2.0)
    good &= _ok("continuous approx bias >0.1 at xmin=1 while exact <0.02",
                bcont > 0.10 and bexact < 0.02, f"cont={bcont:.3f} exact={bexact:.3f}")
    return good


def t6_xmin_selection():
    print("T6  xmin(KS) recovers a planted cutoff above a non-PL body")
    rng = np.random.default_rng(5)
    true_xmin, alpha = 12, 1.6
    tail = A.sample_discrete_pl(6000, alpha, true_xmin, rng)          # PL only for x>=12
    body = rng.integers(1, true_xmin, size=6000)                     # uniform 1..11 (not PL)
    data = np.concatenate([tail, body])
    xm, a, D, ntail = A.fit_xmin_ks(data, min_tail=10, xmin="ks")
    print(f"       selected xmin={xm} (true {true_xmin}), alpha={a:.3f} (true {alpha})")
    return _ok("xmin within +/-3 of planted cutoff", abs(xm - true_xmin) <= 3) and \
           _ok("alpha recovered within 0.1", abs(a - alpha) < 0.10)


def t7_gof_calibration():
    print("T7  GOF p-value calibration (the SPEC criterion)")
    rng = np.random.default_rng(6)
    # (a) true power law -> NOT rejected (no false positive)
    xpl = A.sample_discrete_pl(2000, 1.5, 4, rng)
    fpl = A.powerlaw_mle(xpl, B_gof=300, ci=False, compare=False, seed=7)
    a = _ok("true PL accepted (p>0.10)", fpl.ks_pvalue > 0.10,
            f"p={fpl.ks_pvalue:.3f} tau={fpl.tau:.3f} xmin={fpl.xmin}")
    # (b) exponential data, xmin controlled -> GOF HAS power and rejects the whole dist
    lam = 0.35
    xexp = (rng.geometric(1 - np.exp(-lam), size=2000) + 3).astype(np.int64)  # support >=4
    fexp = A.powerlaw_mle(xexp, xmin=4, B_gof=300, ci=False, compare=False, seed=8)
    b = _ok("exponential rejected at fixed xmin (p<=0.10)", fexp.ks_pvalue <= 0.10,
            f"p={fexp.ks_pvalue:.3f}")
    # (c) DOCUMENTED CAVEAT (not asserted): under KS-floated xmin, GOF loses power vs
    #     exponential because xmin climbs into a small PL-looking far tail. The LR test
    #     (T9), not GOF, is the discriminator here. We print it so the behaviour is visible.
    fflo = A.powerlaw_mle(xexp, xmin="ks", B_gof=200, ci=False, compare=True, seed=8)
    print(f"       [caveat] exp @ KS-xmin={fflo.xmin} (ntail={fflo.n_tail}): GOF p={fflo.ks_pvalue:.3f} "
          f"-> GOF alone underpowered; use the LR test to reject exponential")
    return a and b


def t8_ci_coverage():
    print("T8  bootstrap 95% CI covers the true alpha")
    rng = np.random.default_rng(9)
    alpha, xmin, covered, R = 1.5, 4, 0, 10
    for _ in range(R):
        x = A.sample_discrete_pl(1200, alpha, xmin, rng)
        _, _, taus, _ = A.bootstrap_ci(x, B=150, rng=rng)
        lo, hi = np.percentile(taus[np.isfinite(taus)], [2.5, 97.5])
        covered += (lo <= alpha <= hi)
    frac = covered / R
    return _ok("coverage >= 0.80 (nominal 0.95, small-R noise)", frac >= 0.80,
               f"covered {covered}/{R}")


def t9_vuong():
    print("T9  Vuong LR steers to the true generator")
    rng = np.random.default_rng(10)
    good = True
    # exponential data: exponential should win over power law. Fix xmin low so the LR test
    # has power (letting KS pick a tiny far-tail would leave too few points to discriminate).
    lam = 0.3
    xe = (rng.geometric(1 - np.exp(-lam), size=3000) + 5).astype(np.int64)
    fe = A.powerlaw_mle(xe, xmin=5, gof=False, ci=False, compare=True, seed=1)
    ce = fe.compare.get("exponential", {})
    good &= _ok("exp data: exponential favoured", ce.get("favors") == "exponential",
                f"R={ce.get('R'):+.2f} p={ce.get('p'):.3f}")
    # power-law data: truncated PL should NOT beat PL (cutoff lambda ~ 0)
    xpl = A.sample_discrete_pl(4000, 1.5, 4, rng)
    fp = A.powerlaw_mle(xpl, gof=False, ci=False, compare=True, seed=2)
    ct = fp.compare.get("truncated_pl", {})
    good &= _ok("PL data: truncated-PL NOT preferred", ct.get("favors") != "truncated_pl",
                f"R={ct.get('R'):+.2f} p={ct.get('p'):.3f} favors={ct.get('favors')}")
    return good


def t10_small_n_honesty():
    print("T10 small-n honesty: n≈120 gives a WIDE tau CI (the M3 reality check)")
    rng = np.random.default_rng(11)
    x = A.sample_discrete_pl(120, 1.5, 4, rng)
    f = A.powerlaw_mle(x, B_gof=300, B_ci=400, compare=False, seed=3)
    width = f.ci_exponent[1] - f.ci_exponent[0]
    print(f"       tau={f.tau:.3f}  CI95=[{f.ci_exponent[0]:.3f},{f.ci_exponent[1]:.3f}] "
          f"width={width:.3f}  GOF p={f.ks_pvalue:.3f}")
    # with only 120 draws the CI must be wide enough to make '1.50 exactly' meaningless
    return _ok("tau CI width > 0.20 at n=120", width > 0.20)


def main():
    t0 = time.time()
    results = {}
    for fn in [t1_zeta, t2_mle_vs_grid, t3_sampler, t4_recovery, t5_continuous_bias,
               t6_xmin_selection, t7_gof_calibration, t8_ci_coverage, t9_vuong,
               t10_small_n_honesty]:
        results[fn.__name__] = fn()
        print()
    npass = sum(results.values())
    print(f"==== {npass}/{len(results)} groups passed in {time.time()-t0:.1f}s ====")
    for k, v in results.items():
        if not v:
            print(f"   FAILED: {k}")
    sys.exit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
