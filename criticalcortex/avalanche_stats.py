"""
criticalcortex.avalanche_stats — defensible discrete power-law inference for neuronal
avalanches (Clauset, Shalizi & Newman 2009, *SIAM Rev.* 51(4):661-703).

WHY THIS MODULE EXISTS
----------------------
`criticality.powerlaw_tau` reports a *point* exponent with `xmin` frozen at 4 using the
CONTINUOUS approximation (CSN Eq. 3.7). That is fine as a cheap online diagnostic but it
CANNOT support the SPEC §7.4 contract, which demands:

    fit = powerlaw_mle(sizes, xmin="ks")
    assert 1.45 <= fit.tau <= 1.55         # exponent
    assert fit.ks_pvalue > 0.10            # power law NOT rejected (bootstrap GOF)

Meeting that contract requires four things the cheap estimator omits:
  1. EXACT discrete MLE — maximise the true discrete log-likelihood, whose normaliser is
     the Hurwitz zeta  ζ(α, x_min) = Σ_{k≥0}(k+x_min)^{-α}.  The continuous formula is
     biased by O(1/x_min); at x_min=4 that bias is ~1-3% of τ, comparable to the whole
     [1.45,1.55] acceptance band.
  2. x_min SELECTION by KS — τ is enormously sensitive to where the tail is declared to
     start. Freezing x_min hides that sensitivity; KS selection exposes it and is the CSN
     standard.
  3. A GOODNESS-OF-FIT p-value — a *semi-parametric bootstrap* KS test. Without it, an
     exponent is just a number the formula always returns, even for data that is not
     remotely power-law. This p-value is the criticality criterion.
  4. UNCERTAINTY + ALTERNATIVES — a bootstrap CI on (τ, x_min), and likelihood-ratio tests
     vs exponential / lognormal / truncated-power-law, because "supercritical blob broken
     into avalanches" is exactly the situation that mimics a truncated power law.

DEPENDENCIES: numpy only (host venv is numpy-only). We implement Hurwitz zeta
(Euler-Maclaurin), a 1-D golden-section optimiser, the KS statistic, the discrete PL
sampler and a compact Nelder-Mead ourselves. Every primitive is checked against a closed
form in tests/test_avalanche_stats.py.

COMPLEXITY (n = #avalanches, U = #candidate x_min, B = bootstrap reps, G ≈ 40 golden
iters, Z ≈ 25 zeta terms):
  * single fit at fixed x_min : O(G·Z)                      (~1e3 flops)
  * KS x_min scan             : O(U·G·Z)                    U≲200
  * bootstrap CI              : O(B·U·G·Z)
  * GOF p-value               : O(B·U·G·Z)                  dominant term
For n~1e2..1e4, B=1000 this is ~1-10 s of vectorised numpy — offline, not the hot loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erfc, sqrt
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "hurwitz_zeta",
    "discrete_pl_logpmf",
    "discrete_pl_ccdf",
    "sample_discrete_pl",
    "mle_exponent",
    "mle_exponent_continuous",
    "ks_statistic",
    "fit_xmin_ks",
    "gof_bootstrap_pvalue",
    "bootstrap_ci",
    "compare_distributions",
    "powerlaw_mle",
    "PLFit",
    "format_report",
]

# --------------------------------------------------------------------------------------
# 1. Hurwitz zeta  ζ(s, q) = Σ_{k=0}^∞ (k+q)^{-s}   via Euler-Maclaurin.
# --------------------------------------------------------------------------------------
# EM:  ζ(s,q) = Σ_{k=0}^{N-1}(k+q)^{-s} + (N+q)^{1-s}/(s-1) + ½(N+q)^{-s}
#             + Σ_{j=1}^{M} [B_{2j}/(2j)!] · (s)_{2j-1} · (N+q)^{-(s+2j-1)}
# with (s)_{2j-1} the rising factorial s(s+1)…(s+2j-2).  For s∈[1.01,6], integer q≥1,
# N=20, M=7 gives ~1e-12 relative error (verified against ζ(2,1)=π²/6 etc.).
_EM_N = 20
# Bernoulli-number coefficients c_j = B_{2j}/(2j)!  for j=1..7
_EM_C = np.array([
    1.0 / 12.0,            # B2/2!
    -1.0 / 720.0,          # B4/4!
    1.0 / 30240.0,         # B6/6!
    -1.0 / 1209600.0,      # B8/8!
    1.0 / 47900160.0,      # B10/10!
    -691.0 / 1307674368000.0,   # B12/12!
    1.0 / 74724249600.0,   # B14/14!
], dtype=np.float64)


_EM_C_PY = tuple(float(c) for c in _EM_C)   # pure-Python copy for the scalar hot path


def _zeta_scalar(s: float, q: float) -> float:
    """Scalar Hurwitz ζ(s, q) in pure Python (no numpy dispatch). ~3 µs vs ~68 µs for the
    array path — this is the MLE inner loop, called ~10⁵× per bootstrap fit."""
    total = 0.0
    for k in range(_EM_N):                    # direct head Σ_{k<N}(k+q)^{-s}
        total += (k + q) ** (-s)
    Nq = _EM_N + q
    total += Nq ** (1.0 - s) / (s - 1.0) + 0.5 * Nq ** (-s)
    P = s                                     # rising factorial (s)_1
    total += _EM_C_PY[0] * P * Nq ** (-(s + 1.0))
    for j in range(2, len(_EM_C_PY) + 1):     # correction terms j=2..M
        P *= (s + (2 * j - 3)) * (s + (2 * j - 2))
        total += _EM_C_PY[j - 1] * P * Nq ** (-(s + 2 * j - 1.0))
    return total


def hurwitz_zeta(s, q) -> np.ndarray:
    """ζ(s, q) = Σ_{k≥0}(k+q)^{-s}. Vectorised over `s`; `q` scalar or broadcastable.

    Requires s > 1 (the regime for a normalisable discrete power law) and q > 0.
    """
    s = np.asarray(s, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    scalar = s.ndim == 0 and q.ndim == 0
    s = np.atleast_1d(s)
    # direct head: Σ_{k=0}^{N-1}(k+q)^{-s}; broadcast (N, ·) for scalar or 1-D q
    k = np.arange(_EM_N, dtype=np.float64)
    kq = k[:, None] + np.atleast_1d(q)[None, :] if q.ndim else k[:, None] + float(q)
    head = np.power(kq, -s[None, :]).sum(axis=0)
    Nq = _EM_N + q                                    # (N+q)
    tail = np.power(Nq, 1.0 - s) / (s - 1.0)          # integral term
    tail += 0.5 * np.power(Nq, -s)                    # ½ endpoint
    # Euler-Maclaurin correction terms
    rising = s.copy()                                 # (s)_1 = s
    for j in range(1, len(_EM_C) + 1):
        # (s)_{2j-1} = s(s+1)…(s+2j-2): extend rising factorial to 2j-1 factors
        if j == 1:
            pj = s.copy()
        else:
            pj = np.ones_like(s)
            for m in range(2 * j - 1):
                pj = pj * (s + m)
        tail = tail + _EM_C[j - 1] * pj * np.power(Nq, -(s + 2 * j - 1))
    out = head + tail
    if scalar:
        return float(out[0])
    return out


# --------------------------------------------------------------------------------------
# 2. Discrete power law  p(x) = x^{-α} / ζ(α, x_min),  x ∈ {x_min, x_min+1, …}
# --------------------------------------------------------------------------------------
def discrete_pl_logpmf(x, alpha: float, xmin: int) -> np.ndarray:
    """log p(x) for the discrete (zeta) power law with lower cutoff x_min."""
    x = np.asarray(x, dtype=np.float64)
    logZ = np.log(hurwitz_zeta(alpha, xmin))
    return -alpha * np.log(x) - logZ


def discrete_pl_ccdf(x, alpha: float, xmin: int) -> np.ndarray:
    """Complementary CDF  P(X ≥ x) = ζ(α, x) / ζ(α, x_min)  for x ≥ x_min."""
    x = np.asarray(x, dtype=np.float64)
    return hurwitz_zeta(alpha, x) / hurwitz_zeta(alpha, xmin)


def sample_discrete_pl(n: int, alpha: float, xmin: int, rng: np.random.Generator,
                       exact: bool = False) -> np.ndarray:
    """Draw n samples from the discrete power law ≥ x_min.

    Default is a HYBRID exact-quality sampler:
      * BULK  x ∈ [x_min, X_cap] — inverted from the *true* discrete CCDF by a tabulated
        `searchsorted` (fully vectorised, exact). X_cap is chosen so P(X > X_cap) ≈ 1e-6.
      * FAR TAIL  x > X_cap — the closed-form continuous inverse
        x = [u·(α−1)·ζ(α,x_min)]^{1/(1−α)}, whose O(1/x) discretisation error is negligible
        because x is already ≥ X_cap.
    This is exact where the pure transformation sampler is worst (small x_min / steep α):
    at x_min=1, α=2.5 the transformation method is biased by ~+0.25 in the recovered
    exponent; the tabulated bulk removes that entirely (see tests T3/T4).

    `exact=True` selects the slow bracket+bisection reference used only to certify the
    fast path.
    """
    u = rng.random(n)
    Z = hurwitz_zeta(alpha, xmin)
    if exact:
        out = np.empty(n, dtype=np.int64)
        for i in range(n):
            ui = u[i]
            x1, x2 = xmin, xmin
            while hurwitz_zeta(alpha, x2) / Z >= ui:
                x1, x2 = x2, x2 * 2
            while x2 - x1 > 1:
                mid = (x1 + x2) // 2
                if hurwitz_zeta(alpha, mid) / Z >= ui:
                    x1 = mid
                else:
                    x2 = mid
            out[i] = x1
        return out
    # analytic X_cap where CCDF ≈ eps; the table only needs to be exact at small-to-moderate
    # x (that's where the transformation sampler is biased) — beyond 20k the analytic far
    # tail is accurate to O(1/x) < 5e-5, so cap the table there to keep it cheap.
    eps = 1e-6
    Xcap = (eps * (alpha - 1.0) * Z) ** (1.0 / (1.0 - alpha))
    Xcap = int(min(max(Xcap, xmin + 64), xmin + 20_000))
    xs = np.arange(xmin, Xcap + 1, dtype=np.float64)
    ccdf = hurwitz_zeta(alpha, xs) / Z                       # P(X ≥ x), strictly decreasing
    tail_mass = float(hurwitz_zeta(alpha, Xcap + 1) / Z)     # P(X > X_cap)
    out = np.empty(n, dtype=np.int64)
    in_table = u >= tail_mass
    if in_table.any():
        # largest index i with ccdf[i] ≥ u  ==  #{ccdf ≥ u} − 1  ==  searchsorted(−ccdf, −u) − 1
        pos = np.searchsorted(-ccdf, -u[in_table], side="right") - 1
        out[in_table] = xmin + np.clip(pos, 0, xs.size - 1)
    far = ~in_table
    if far.any():
        xf = (u[far] * (alpha - 1.0) * Z) ** (1.0 / (1.0 - alpha))
        out[far] = np.maximum(np.round(xf).astype(np.int64), Xcap + 1)
    return out


# --------------------------------------------------------------------------------------
# 3. Maximum-likelihood exponent
# --------------------------------------------------------------------------------------
def mle_exponent_continuous(x_tail, xmin: int) -> float:
    """CSN Eq. 3.7 continuous approximation (what `powerlaw_tau` uses). Kept for
    side-by-side bias demonstration only."""
    x = np.asarray(x_tail, dtype=np.float64)
    return 1.0 + x.size / np.sum(np.log(x / (xmin - 0.5)))


def _neg_loglik(alpha: float, sum_log_x: float, n: int, xmin: int) -> float:
    """−log L(α) per CSN Eq. 3.5:  L = −α·Σln x − n·ln ζ(α, x_min). Uses the scalar ζ."""
    from math import log as _log
    return alpha * sum_log_x + n * _log(_zeta_scalar(alpha, float(xmin)))


def _mle_from_stats(sum_log_x: float, n: int, xmin: int,
                    bounds=(1.01, 15.0), tol: float = 1e-6) -> float:
    """Golden-section minimiser of −log L given the sufficient statistics (Σln x, n).
    Separated from array handling so the x_min scan can reuse precomputed suffix sums."""
    a, b = bounds
    invphi = (sqrt(5.0) - 1.0) / 2.0          # 1/φ  ≈ 0.618
    invphi2 = (3.0 - sqrt(5.0)) / 2.0         # 1/φ² ≈ 0.382
    c = a + invphi2 * (b - a)
    d = a + invphi * (b - a)
    fc = _neg_loglik(c, sum_log_x, n, xmin)
    fd = _neg_loglik(d, sum_log_x, n, xmin)
    while (b - a) > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = a + invphi2 * (b - a)
            fc = _neg_loglik(c, sum_log_x, n, xmin)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = _neg_loglik(d, sum_log_x, n, xmin)
    return 0.5 * (a + b)


def mle_exponent(x_tail, xmin: int, bounds=(1.01, 15.0), tol: float = 1e-6) -> float:
    """EXACT discrete MLE: minimise −log L over α by golden-section search.

    The discrete log-likelihood is strictly convex in α on (1, ∞) (ζ is log-convex in the
    exponent), so golden-section converges to the global optimum without derivatives.
    """
    x = np.asarray(x_tail, dtype=np.float64)
    return _mle_from_stats(float(np.sum(np.log(x))), x.size, xmin, bounds, tol)


# --------------------------------------------------------------------------------------
# 4. KS statistic and x_min selection
# --------------------------------------------------------------------------------------
def _ks_sorted(tail_sorted: np.ndarray, alpha: float, xmin: int) -> float:
    """KS distance for an ALREADY-SORTED integer tail (no re-sort). Hot-path helper."""
    n = tail_sorted.size
    if n < 2:
        return np.inf
    xs = np.unique(tail_sorted)
    emp = np.searchsorted(tail_sorted, xs, side="right") / n
    model = 1.0 - discrete_pl_ccdf(xs + 1, alpha, xmin)   # P(X ≤ xs) = 1 − P(X ≥ xs+1)
    return float(np.max(np.abs(emp - model)))


def ks_statistic(x_tail, alpha: float, xmin: int) -> float:
    """One-sample KS distance between the empirical CDF of x ≥ x_min and the fitted
    discrete-PL CDF:  D = max_x |S(x) − P(x)|  (CSN Eq. 3.9)."""
    return _ks_sorted(np.sort(np.asarray(x_tail, dtype=np.int64)), alpha, xmin)


def _candidates_from_sorted(ds: np.ndarray, min_tail: int, max_candidates: int):
    """Given SORTED data, return (vals, first_idx) for eligible x_min candidates, where
    first_idx[j] is the index of the first element ≥ vals[j] (so tail = ds[first_idx[j]:]).
    Counts are computed by a single vectorised searchsorted, not a Python loop."""
    vals = np.unique(ds)
    vals = vals[vals >= 1]
    first_idx = np.searchsorted(ds, vals, side="left")     # #{ds < v}; tail size = n − idx
    counts = ds.size - first_idx
    keep = counts >= min_tail
    vals, first_idx = vals[keep], first_idx[keep]
    if vals.size > max_candidates:                          # log-spaced thinning (τ smooth in log xmin)
        sel = np.unique(np.round(np.geomspace(1, vals.size, max_candidates)).astype(int) - 1)
        vals, first_idx = vals[sel], first_idx[sel]
    return vals, first_idx


def _candidate_xmins(data, min_tail: int, max_candidates: int = 200) -> np.ndarray:
    """Public helper (used in tests): eligible x_min values leaving ≥ min_tail tail points."""
    vals, _ = _candidates_from_sorted(np.sort(np.asarray(data, dtype=np.int64)),
                                      min_tail, max_candidates)
    return vals


@dataclass
class PLFit:
    """Result of a discrete power-law fit. `.tau` and `.alpha` both alias the exponent so
    the SPEC call sites `powerlaw_mle(sizes).tau` and `powerlaw_mle(durations).alpha` work."""
    exponent: float
    xmin: int
    ks_stat: float
    n_tail: int
    n_total: int
    ks_pvalue: float = float("nan")
    ci_exponent: Optional[tuple] = None
    ci_xmin: Optional[tuple] = None
    exponent_continuous: float = float("nan")
    loglik: float = float("nan")
    compare: dict = field(default_factory=dict)
    n_gof: int = 0

    @property
    def tau(self) -> float:
        return self.exponent

    @property
    def alpha(self) -> float:
        return self.exponent


def fit_xmin_ks(data, min_tail: int = 10, xmin: str | int = "ks",
                max_candidates: int = 200, tol: float = 1e-6):
    """Fit (x_min, α). If xmin is an int, fit α there; if "ks", pick x_min minimising the
    KS distance (CSN §3.3). Returns (xmin, alpha, ks_stat, n_tail).

    Hot-path design: sort once, precompute the suffix sum Σ_{i≥k} ln ds[i] so each candidate
    x_min supplies (Σln x, n_tail) in O(1); only the golden-section MLE and one vectorised
    KS evaluation remain per candidate. This is what makes B≈1000-2500 bootstraps tractable.
    """
    ds = np.sort(np.asarray(data, dtype=np.int64))
    ds = ds[ds >= 1]
    n = ds.size
    if isinstance(xmin, (int, np.integer)):
        i = int(np.searchsorted(ds, xmin, side="left"))
        tail = ds[i:]
        if tail.size < 2:
            return int(xmin), float("nan"), float("inf"), int(tail.size)
        a = _mle_from_stats(float(np.sum(np.log(tail))), tail.size, int(xmin), tol=tol)
        return int(xmin), a, _ks_sorted(tail, a, int(xmin)), int(tail.size)

    vals, first_idx = _candidates_from_sorted(ds, min_tail, max_candidates)
    if vals.size == 0:
        return int(ds.min()), float("nan"), float("inf"), int(n)
    logds = np.log(ds.astype(np.float64))
    suffix_logsum = np.cumsum(logds[::-1])[::-1]        # suffix_logsum[i] = Σ_{k≥i} ln ds[k]
    best = None
    for v, i in zip(vals.tolist(), first_idx.tolist()):
        ntail = n - i
        a = _mle_from_stats(float(suffix_logsum[i]), ntail, v, tol=tol)
        D = _ks_sorted(ds[i:], a, v)
        if best is None or D < best[2]:
            best = (v, a, D, ntail)
    return best


# --------------------------------------------------------------------------------------
# 5. Goodness-of-fit: semi-parametric bootstrap p-value (CSN §4.1)
# --------------------------------------------------------------------------------------
def gof_bootstrap_pvalue(data, fit: PLFit, B: int = 1000,
                         rng: Optional[np.random.Generator] = None,
                         min_tail: int = 10, max_candidates: int = 100,
                         tol: float = 1e-4) -> float:
    """p = fraction of synthetic KS distances ≥ the observed KS distance.

    Each synthetic dataset preserves the below-x_min empirical mass and replaces the tail
    with true power-law draws, then is RE-FIT from scratch (x_min re-selected) so the
    p-value accounts for x_min-selection uncertainty. p > 0.10 ⇒ the power law is a
    plausible generator; p ≤ 0.10 ⇒ reject (CSN's conservative threshold).
    """
    rng = rng or np.random.default_rng()
    data = np.asarray(data, dtype=np.int64)
    data = data[data >= 1]
    n = data.size
    below = data[data < fit.xmin]
    n_below = below.size
    p_tail = fit.n_tail / n
    D_obs = fit.ks_stat
    ge = 0
    for _ in range(B):
        n_tail_b = int(rng.binomial(n, p_tail))
        parts = []
        if n_tail_b > 0:
            parts.append(sample_discrete_pl(n_tail_b, fit.exponent, fit.xmin, rng))
        n_bel_b = n - n_tail_b
        if n_bel_b > 0:
            if n_below > 0:
                parts.append(rng.choice(below, size=n_bel_b, replace=True))
            else:                                       # no below-mass observed
                parts.append(np.full(n_bel_b, fit.xmin, dtype=np.int64))
        synth = np.concatenate(parts) if parts else np.array([fit.xmin])
        _, a_b, D_b, _ = fit_xmin_ks(synth, min_tail=min_tail, xmin="ks",
                                     max_candidates=max_candidates, tol=tol)
        if D_b >= D_obs:
            ge += 1
    return ge / B


# --------------------------------------------------------------------------------------
# 6. Bootstrap CI on (τ, x_min)  (CSN §3.5, nonparametric)
# --------------------------------------------------------------------------------------
def bootstrap_ci(data, B: int = 1000, ci: float = 0.95,
                 rng: Optional[np.random.Generator] = None, min_tail: int = 10,
                 max_candidates: int = 100, tol: float = 1e-4):
    """Percentile bootstrap: resample the WHOLE dataset with replacement, re-fit (x_min, τ)
    each time. Returns ((τ_lo, τ_hi), (xmin_lo, xmin_hi), taus, xmins)."""
    rng = rng or np.random.default_rng()
    data = np.asarray(data, dtype=np.int64)
    n = data.size
    taus = np.empty(B)
    xmins = np.empty(B)
    for b in range(B):
        samp = rng.choice(data, size=n, replace=True)
        xm, a, _, _ = fit_xmin_ks(samp, min_tail=min_tail, xmin="ks",
                                  max_candidates=max_candidates, tol=tol)
        taus[b] = a
        xmins[b] = xm
    lo = (1 - ci) / 2 * 100
    hi = (1 + ci) / 2 * 100
    good = np.isfinite(taus)
    tau_ci = (float(np.percentile(taus[good], lo)), float(np.percentile(taus[good], hi)))
    xmin_ci = (float(np.percentile(xmins[good], lo)), float(np.percentile(xmins[good], hi)))
    return tau_ci, xmin_ci, taus, xmins


# --------------------------------------------------------------------------------------
# 7. Alternative distributions + Vuong likelihood-ratio tests (CSN §5)
# --------------------------------------------------------------------------------------
def _nelder_mead(f, x0, step=0.5, tol=1e-6, max_iter=400):
    """Compact 2-D Nelder-Mead (no scipy). Minimises f: R^k → R."""
    x0 = np.asarray(x0, dtype=np.float64)
    k = x0.size
    sim = np.vstack([x0] + [x0 + step * np.eye(k)[i] for i in range(k)])
    fv = np.array([f(p) for p in sim])
    for _ in range(max_iter):
        order = np.argsort(fv)
        sim, fv = sim[order], fv[order]
        if abs(fv[-1] - fv[0]) < tol:
            break
        cen = sim[:-1].mean(axis=0)
        xr = cen + (cen - sim[-1])                      # reflect
        fr = f(xr)
        if fr < fv[0]:
            xe = cen + 2.0 * (cen - sim[-1])            # expand
            fe = f(xe)
            sim[-1], fv[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fv[-2]:
            sim[-1], fv[-1] = xr, fr
        else:
            xc = cen + 0.5 * (sim[-1] - cen)            # contract
            fc = f(xc)
            if fc < fv[-1]:
                sim[-1], fv[-1] = xc, fc
            else:                                        # shrink
                sim[1:] = sim[0] + 0.5 * (sim[1:] - sim[0])
                fv[1:] = np.array([f(p) for p in sim[1:]])
    return sim[np.argmin(fv)]


_NORM_CAP = 500_000   # hard bound on any normaliser grid (keeps memory/time flat)


def _logsumexp(a):
    m = float(np.max(a))
    return m + float(np.log(np.sum(np.exp(a - m))))


def _ll_exponential(x_tail, xmin):
    """Discrete exponential p(x) = (1−e^{−λ})e^{−λ(x−x_min)}. Closed-form MLE:
    λ = ln(1 + 1/⟨x−x_min⟩)."""
    x = np.asarray(x_tail, dtype=np.float64)
    m = np.mean(x - xmin)
    if m <= 0:
        return None
    lam = np.log1p(1.0 / m)
    logp = np.log1p(-np.exp(-lam)) - lam * (x - xmin)
    return logp, dict(lam=lam)


def _lognormal_logZ_and_logp(x, xmin, mu, s):
    """log-normaliser and per-point logp for the discrete lognormal. Grid is capped at
    exp(μ+12σ) (beyond which lognormal mass is ~0), never exceeding _NORM_CAP."""
    xup = min(int(np.exp(mu + 12.0 * s)) + 1, xmin + _NORM_CAP)
    xup = max(xup, xmin + 1000)
    grid = np.arange(xmin, xup + 1, dtype=np.float64)
    logf = -np.log(grid) - 0.5 * ((np.log(grid) - mu) / s) ** 2
    logZ = _logsumexp(logf)
    lx = np.log(x)
    logp = (-lx - 0.5 * ((lx - mu) / s) ** 2) - logZ
    return logZ, logp


def _ll_lognormal(x_tail, xmin):
    """Discrete lognormal p(x) ∝ (1/x)·exp(−(ln x − μ)²/2σ²) on x ≥ x_min. MLE over
    (μ, ln σ) by Nelder-Mead; normaliser by bounded summation."""
    x = np.asarray(x_tail, dtype=np.float64)
    lx = np.log(x)

    def negll(theta):
        mu, ls = theta
        s = min(max(np.exp(ls), 1e-3), 50.0)
        logZ, logp = _lognormal_logZ_and_logp(x, xmin, mu, s)
        return -float(logp.sum())

    mu, ls = _nelder_mead(negll, np.array([np.mean(lx), np.log(np.std(lx) + 1e-3)]))
    s = min(max(np.exp(ls), 1e-3), 50.0)
    _, logp = _lognormal_logZ_and_logp(x, xmin, mu, s)
    return logp, dict(mu=mu, sigma=s)


def _trunc_logZ(xmin, a, lam):
    """log Σ_{x≥xmin} x^{−a} e^{−λx}, computed on a bounded grid plus an analytic
    power-law tail  e^{−λ(xup+1)}·ζ(a, xup+1)  for the mass beyond the grid (exact when
    λ→0, negligible when λ is large enough that the cutoff sits inside the grid)."""
    if lam > 1e-9:
        xup = xmin + min(int(40.0 / lam) + 1, _NORM_CAP)
    else:
        xup = xmin + _NORM_CAP
    grid = np.arange(xmin, xup + 1, dtype=np.float64)
    logf = -a * np.log(grid) - lam * grid
    logZ_grid = _logsumexp(logf)
    Z = np.exp(logZ_grid)
    if a > 1.001:
        Z += np.exp(-lam * (xup + 1)) * hurwitz_zeta(a, xup + 1)
    return np.log(Z)


def _ll_truncated_pl(x_tail, xmin, alpha0):
    """Power law with exponential cutoff p(x) ∝ x^{−α}e^{−λx} on x ≥ x_min. MLE over
    (α, λ) by Nelder-Mead. This is the key alternative when a supercritical blob has been
    broken into finite avalanches — a finite-size cutoff mimics scale-freeness."""
    x = np.asarray(x_tail, dtype=np.float64)
    lx = np.log(x)

    def negll(theta):
        a, loglam = theta
        lam = min(max(np.exp(loglam), 0.0), 10.0)
        logZ = _trunc_logZ(xmin, a, lam)
        return -float(np.sum((-a * lx - lam * x) - logZ))

    a, loglam = _nelder_mead(negll, np.array([alpha0, np.log(1.0 / max(float(x.max()), 2.0))]))
    lam = min(max(np.exp(loglam), 0.0), 10.0)
    logZ = _trunc_logZ(xmin, a, lam)
    logp = (-a * lx - lam * x) - logZ
    return logp, dict(alpha=a, lam=lam)


def _vuong(ll1, ll2, nested: bool = False):
    """Vuong normalised LR test between two per-point log-likelihood vectors.
    R>0 favours model 1 (power law). p is the two-sided significance that the sign is real;
    for the nested truncated-PL case we report the one-sided p (PL ⊂ truncated-PL)."""
    d = np.asarray(ll1) - np.asarray(ll2)
    n = d.size
    R = float(d.sum())
    var = float(np.mean(d ** 2) - np.mean(d) ** 2)
    if var <= 0:
        return R, float("nan"), float("nan")
    z = R / (sqrt(n) * sqrt(var))
    p_two = erfc(abs(z) / sqrt(2.0))
    p = p_two / 2 if nested else p_two
    return R, float(z), float(p)


def compare_distributions(data, fit: PLFit) -> dict:
    """Vuong LR of the power law vs exponential, lognormal, truncated power law, all fit on
    the same tail x ≥ x_min. Each entry: dict(R, z, p, favors, params).
    R>0 & p<0.1 ⇒ power law wins; R<0 & p<0.1 ⇒ alternative wins; p≥0.1 ⇒ indistinguishable.
    """
    data = np.asarray(data, dtype=np.int64)
    tail = data[data >= fit.xmin]
    if tail.size < 5:
        return {}
    ll_pl = discrete_pl_logpmf(tail, fit.exponent, fit.xmin)
    out = {}
    for name, fn, nested in (
        ("exponential", lambda: _ll_exponential(tail, fit.xmin), False),
        ("lognormal", lambda: _ll_lognormal(tail, fit.xmin), False),
        ("truncated_pl", lambda: _ll_truncated_pl(tail, fit.xmin, fit.exponent), True),
    ):
        try:
            res = fn()
            if res is None:
                continue
            ll_alt, params = res
            R, z, p = _vuong(ll_pl, ll_alt, nested=nested)
            favors = ("power_law" if R > 0 else name) if (np.isfinite(p) and p < 0.10) \
                else "inconclusive"
            out[name] = dict(R=R, z=z, p=p, favors=favors, params=params)
        except Exception as e:                          # never let an alt fit break the report
            out[name] = dict(error=str(e))
    return out


# --------------------------------------------------------------------------------------
# 8. Top-level entry point — matches SPEC §7.4 `powerlaw_mle(sizes, xmin="ks")`
# --------------------------------------------------------------------------------------
def powerlaw_mle(data, xmin: str | int = "ks", gof: bool = True, ci: bool = True,
                 compare: bool = True, B_gof: int = 1000, B_ci: int = 1000,
                 min_tail: int = 10, seed: Optional[int] = 0) -> PLFit:
    """Full defensible discrete power-law fit.

    Parameters
    ----------
    data    : avalanche sizes (or durations), 1-D int array.
    xmin    : "ks" to select by KS minimisation, or an int to fix it.
    gof     : compute the bootstrap KS goodness-of-fit p-value (the SPEC criterion).
    ci      : compute the bootstrap CI on (τ, x_min).
    compare : run Vuong LR tests vs exponential / lognormal / truncated power law.
    B_gof, B_ci : bootstrap replica counts.
    seed    : RNG seed for reproducibility (bootstraps).

    Returns a PLFit (`.tau`/`.alpha` = exponent, `.xmin`, `.ks_stat`, `.ks_pvalue`,
    `.ci_exponent`, `.compare`, …).
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=np.int64)
    data = data[data >= 1]
    n_total = data.size
    if n_total < 2:
        return PLFit(float("nan"), 1, float("inf"), 0, n_total)

    xm, alpha, D, n_tail = fit_xmin_ks(data, min_tail=min_tail, xmin=xmin)
    tail = data[data >= xm]
    fit = PLFit(
        exponent=alpha, xmin=int(xm), ks_stat=D, n_tail=int(n_tail), n_total=int(n_total),
        exponent_continuous=(mle_exponent_continuous(tail, xm) if tail.size >= 2 else float("nan")),
        loglik=(float(np.sum(discrete_pl_logpmf(tail, alpha, xm))) if tail.size >= 2 else float("nan")),
    )
    if gof and np.isfinite(D):
        fit.ks_pvalue = gof_bootstrap_pvalue(data, fit, B=B_gof, rng=rng, min_tail=min_tail)
        fit.n_gof = B_gof
    if ci:
        tau_ci, xmin_ci, _, _ = bootstrap_ci(data, B=B_ci, rng=rng, min_tail=min_tail)
        fit.ci_exponent = tau_ci
        fit.ci_xmin = xmin_ci
    if compare and np.isfinite(alpha):
        fit.compare = compare_distributions(data, fit)
    return fit


# --------------------------------------------------------------------------------------
# 9. Human-readable report
# --------------------------------------------------------------------------------------
def format_report(fit: PLFit, label: str = "avalanche sizes",
                  tau_target: float = 1.5, band=(1.45, 1.55)) -> str:
    L = []
    L.append(f"== Discrete power-law fit: {label} ==")
    L.append(f"  n_total        = {fit.n_total}   (tail n≥xmin = {fit.n_tail})")
    L.append(f"  xmin (KS-sel)  = {fit.xmin}")
    ci = f"  CI95 [{fit.ci_exponent[0]:.3f}, {fit.ci_exponent[1]:.3f}]" if fit.ci_exponent else ""
    L.append(f"  tau (exact MLE)= {fit.exponent:.4f}{ci}")
    L.append(f"  tau (cont.aprx)= {fit.exponent_continuous:.4f}   <- what powerlaw_tau reports")
    if fit.ci_xmin:
        L.append(f"  xmin CI95      = [{fit.ci_xmin[0]:.0f}, {fit.ci_xmin[1]:.0f}]")
    L.append(f"  KS distance D  = {fit.ks_stat:.4f}")
    if fit.n_gof:
        verdict = "PLAUSIBLE (not rejected)" if fit.ks_pvalue > 0.10 else "REJECTED"
        L.append(f"  GOF p-value    = {fit.ks_pvalue:.3f}  [{fit.n_gof} reps]  -> power law {verdict}")
    in_band = band[0] <= fit.exponent <= band[1]
    L.append(f"  target tau={tau_target}: exponent {'IN' if in_band else 'OUT of'} band {band}")
    if fit.compare:
        L.append("  vs alternatives (Vuong LR; R>0 favours power law):")
        for name, c in fit.compare.items():
            if "error" in c:
                L.append(f"    - {name:12s}: (skipped: {c['error']})")
            else:
                L.append(f"    - {name:12s}: R={c['R']:+.2f}  p={c['p']:.3f}  -> {c['favors']}")
    # overall gate mirroring SPEC §7.4
    passes = (fit.n_gof and fit.ks_pvalue > 0.10 and in_band)
    L.append(f"  SPEC §7.4 gate : {'PASS' if passes else 'FAIL'} "
             f"(needs ks_pvalue>0.10 AND {band[0]}<=tau<={band[1]})")
    return "\n".join(L)


if __name__ == "__main__":
    # smoke demo on a clean critical branching surrogate
    rng = np.random.default_rng(0)
    x = sample_discrete_pl(5000, 1.5, 4, rng)
    print(format_report(powerlaw_mle(x, B_gof=300, B_ci=300, seed=1)))
