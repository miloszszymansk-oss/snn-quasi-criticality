"""
criticalcortex.criticality — SOC verification metrics.

Turns an AER spike stream into the statistics that diagnose the critical state:

  * branching_ratio  — MR/AR(1) estimate sigma = slope of A(t+1) on A(t); sigma -> 1 at
                       the critical point (Wilting & Priesemann 2018, single-step form).
  * avalanches       — neuronal-avalanche extraction: bin the population activity at the
                       mean inter-event interval (Beggs & Plenz 2003), an avalanche is a
                       maximal run of non-empty bins; returns sizes and durations.
  * powerlaw_tau     — Clauset-Shalizi-Newman MLE for P(S) ~ S^-tau (critical target ~3/2).

These are the same estimators calibrated in generate_fI_ref's synthetic-branching check;
here they consume the running network's output to test self-organized criticality.
"""

from __future__ import annotations

import numpy as np


def population_activity(aer_step, lo, hi) -> np.ndarray:
    """Spikes-per-step A(t) over the window [lo, hi)."""
    s = np.asarray(aer_step)
    m = (s >= lo) & (s < hi)
    return np.bincount(s[m] - lo, minlength=hi - lo).astype(np.float64)


def branching_ratio(aer_step, lo, hi) -> float:
    """sigma = slope of A(t+1) on A(t) over [lo, hi). ~1 at criticality, <1 subcritical."""
    A = population_activity(aer_step, lo, hi)
    if A.size < 3:
        return 0.0
    x, y = A[:-1], A[1:]
    xm = x.mean()
    denom = float(((x - xm) ** 2).sum())
    return float(((x - xm) * (y - y.mean())).sum() / denom) if denom > 0 else 0.0


def avalanches(aer_step, lo, hi):
    """Neuronal avalanches over [lo, hi): bin at the mean inter-event interval (~1 spike/
    bin), an avalanche is a maximal run of non-empty bins. Returns (sizes, durations)."""
    A = population_activity(aer_step, lo, hi)
    total = A.sum()
    if total < 10:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    bw = max(1, int(round((hi - lo) / total)))          # mean inter-event interval, in steps
    nb = A.size // bw
    B = A[: nb * bw].reshape(nb, bw).sum(axis=1)
    occupied = (B > 0).astype(np.int8)
    edges = np.diff(np.concatenate(([0], occupied, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    sizes = np.array([B[s:e].sum() for s, e in zip(starts, ends)], dtype=np.int64)
    durations = (ends - starts).astype(np.int64)
    return sizes, durations


def powerlaw_tau(sizes, xmin: int = 4):
    """Clauset-Shalizi-Newman discrete-MLE exponent for P(S) ~ S^-tau (>= xmin).
    Returns (tau, n_tail); tau is NaN if too few tail samples."""
    x = np.asarray(sizes, dtype=np.float64)
    x = x[x >= xmin]
    if x.size < 20:
        return float("nan"), int(x.size)
    tau = 1.0 + x.size / np.sum(np.log(x / (xmin - 0.5)))
    return float(tau), int(x.size)


def summarize(aer_step, lo, hi) -> dict:
    """Convenience bundle of the SOC diagnostics over a stationary window."""
    sizes, durs = avalanches(aer_step, lo, hi)
    tau, ntail = powerlaw_tau(sizes)
    A = population_activity(aer_step, lo, hi)
    return dict(
        branching_ratio=branching_ratio(aer_step, lo, hi),
        n_avalanches=int(sizes.size),
        max_avalanche=int(sizes.max()) if sizes.size else 0,
        tau=tau,
        tau_tail=ntail,
        mean_activity=float(A.mean()),
    )
