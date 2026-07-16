#!/usr/bin/env python3
"""
generate_fI_ref.py — Milestone M0 calibrator.

Produces two things from a single isolated Regular-Spiking (RS) Izhikevich neuron
(a=0.02, b=0.20, c=-65, d=8):

  (A) reference/izh_rs_fI.npz — the T1.1 GOLDEN MASTER (arrays `I`, `f`), computed
      with EXACTLY the arguments the test replays: dt=0.1 ms, T_ms=1000.
  (B) kappa — the single-neuron gain d(P_spike)/d(PSP) that SPEC §4.4 uses to set
      w0 = g/(K*kappa) so that the network branching ratio sigma == g by construction.

Derivation of kappa  (documented, not guessed)
----------------------------------------------
Branching bookkeeping (SPEC §1, §4.4):
    sigma = K_out * p_transmit ,  p_transmit = expected postsyn spikes per presyn spike.
We WANT p_transmit = kappa * w (linear small-signal gain); then
    w0 = g/(K*kappa)  =>  sigma = K*kappa*w0 = g.                      (sigma == g)

Map one EPSP -> expected extra spikes via linear (static-gain) response of the f–I curve:
  * A presyn spike adds weight w to the current-based synapse g_exc, which decays with
    tau_syn. The injected "current charge" is  Q = ∫ w e^{-t/tau_syn} dt = w*tau_syn
    [ (mV/ms)*ms = mV ].
  * In the adiabatic approximation the extra spikes from a slow current perturbation are
    Δspikes ≈ (dr/dI)*Q with r the rate in spikes/ms. Hence
        p_transmit = (dr/dI)*tau_syn*w   =>   kappa = tau_syn * (dr/dI).
    With f = 1000*r (Hz):  dr/dI = (df/dI)/1000, so
        kappa = tau_syn * (df/dI) / 1000        [ df/dI in Hz per current-unit; tau_syn in ms ].

Operating point  (empirically corrected — see §"MEASURED" below)
----------------------------------------------------------------
MEASURED at dt=0.1 ms: the deterministic (constant-current) RS f–I curve has a
*discontinuous onset* — the neuron is silent up to I≈3.7 and then jumps to ≈5.5 Hz;
there is no sustained sub-~5.5 Hz firing under constant drive. So the naive 5 Hz
operating point is BELOW the reachable deterministic domain, and a raw finite-difference
slope there is meaningless (its window straddles the silent zone). We therefore:
  * fit the class-1 square-root law  f^2 = m*(I - I_rheo)  over a band ABOVE the onset
    jump (6–20 Hz), where f^2 is linear in I with R^2 > 0.99;
  * take the operating rate f0 = 6 Hz — the lowest in-domain deterministic rate, and the
    closest clean proxy for the network's ~5 Hz FLUCTUATION-driven regime (SPEC §3.6);
  * evaluate the analytic gain  df/dI|_{f0} = m/(2*f0)  and cross-check it with a local
    linear fit of f vs I over a window CLIPPED to the firing branch (they agree to ~5%).

This kappa is a FIRST-ORDER seed. Its real validation is M2's test_locate_critical_point,
which measures the empirical g_c from the running network (where noise smooths the onset)
and asserts |g_c - 1| < 0.02. The kappa(f0) table below is stored for that reconciliation.
"""

import os
import numpy as np

from criticalcortex.numerics import measure_fI

# RS neuron (SPEC §7.2 / Izhikevich 2003 "regular spiking")
A, B, C, D = 0.02, 0.20, -65.0, 8.0
DT = 0.1                # ms  (SPEC §4.2)
TAU_SYN = 5.0           # ms  (SPEC §9 [dynamics].tau_syn_ms)
F_OP = 6.0              # Hz  operating rate: lowest in-domain deterministic rate,
                        #     closest proxy for the network's ~5 Hz (SPEC §3.6)
FIT_LO, FIT_HI = 6.0, 20.0   # Hz band for the sqrt-law fit (ABOVE the onset jump)
F0_TABLE = (6.0, 8.0, 10.0, 15.0, 20.0)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ref_dir = os.path.join(here, "reference")
    os.makedirs(ref_dir, exist_ok=True)

    I_grid = np.round(np.arange(0.0, 30.0 + 1e-9, 0.1), 2)   # current units (mV/ms)

    # (A) GOLDEN MASTER — exact args the T1.1 test will replay (dt=0.1, T_ms=1000).
    f_grid = measure_fI(A, B, C, D, I_grid, dt=DT, T_ms=1000.0)

    # (B) Calibration curve — longer integration => finer rate resolution near threshold.
    f_cal = measure_fI(A, B, C, D, I_grid, dt=DT, T_ms=4000.0, warmup_ms=1000.0)

    assert np.all(np.isfinite(f_grid)) and np.all(np.isfinite(f_cal)), "non-finite f — unstable integration"

    # Discontinuous onset: first firing current and the minimum sustained rate.
    fire = np.nonzero(f_cal > 0.0)[0]
    I_on = float(I_grid[fire[0]])
    f_min = float(f_cal[fire[0]])

    # sqrt-law fit  f^2 = m*I + q  over the low-rate band ABOVE the onset jump.
    band = (f_cal >= FIT_LO) & (f_cal <= FIT_HI)
    Ib, f2b = I_grid[band], f_cal[band] ** 2
    m, q = np.polyfit(Ib, f2b, 1)
    I_rheo = -q / m
    r2 = 1.0 - np.sum((f2b - (m * Ib + q)) ** 2) / np.sum((f2b - f2b.mean()) ** 2)

    # kappa(f0) table (kappa is rate-dependent for a class-1 neuron).
    f0s = np.array(F0_TABLE, dtype=np.float64)
    I0s = I_rheo + f0s ** 2 / m
    dfdI = m / (2.0 * f0s)
    kappa_tab = TAU_SYN * dfdI / 1000.0

    # Primary operating point (f0 = F_OP).
    k_op = int(np.argmin(np.abs(f0s - F_OP)))
    I_op, dfdI_op, kappa = float(I0s[k_op]), float(dfdI[k_op]), float(kappa_tab[k_op])

    # Cross-check: local linear fit of f vs I over a window CLIPPED to the firing branch.
    win = (I_grid >= max(I_op - 0.5, I_on)) & (I_grid <= I_op + 0.5)
    dfdI_local = float(np.polyfit(I_grid[win], f_cal[win], 1)[0])
    kappa_local = TAU_SYN * dfdI_local / 1000.0

    out = os.path.join(ref_dir, "izh_rs_fI.npz")
    np.savez(
        out,
        I=I_grid, f=f_grid,                          # <-- the two arrays T1.1 asserts on
        f_cal=f_cal,
        kappa=np.float64(kappa),                     # primary seed (f0 = F_OP)
        I_op=np.float64(I_op), f_op=np.float64(F_OP),
        dfdI_op=np.float64(dfdI_op),
        I_rheo=np.float64(I_rheo), I_onset=np.float64(I_on), f_min=np.float64(f_min),
        sqrt_m=np.float64(m), sqrt_r2=np.float64(r2),
        f0_table=f0s, kappa_table=kappa_tab, I0_table=I0s,
        tau_syn=np.float64(TAU_SYN),
        params=np.array([A, B, C, D], dtype=np.float64),
        dt=np.float64(DT),
    )

    print("== M0 f-I reference & kappa calibration ==")
    print(f"grid                 I in [0,30] step 0.1  ({I_grid.size} pts)")
    print(f"f range (T=1000)     {f_grid.min():.1f} .. {f_grid.max():.1f} Hz")
    print(f"DISCONTINUOUS onset  silent -> {f_min:.2f} Hz jump at I = {I_on:.2f}  (no sub-{f_min:.1f}Hz constant-current firing)")
    print(f"sqrt-law fit         f^2 = {m:.4f}(I - {I_rheo:.4f})   R^2 = {r2:.5f}   band {FIT_LO:.0f}-{FIT_HI:.0f} Hz")
    print(f"operating point      f0 = {F_OP:.1f} Hz  ->  I_op = {I_op:.4f}")
    print(f"gain df/dI @ f0      analytic {dfdI_op:.4f}  |  firing-clipped linfit {dfdI_local:.4f}  Hz/cu  (agree {100*abs(dfdI_op-dfdI_local)/dfdI_op:.1f}%)")
    print(f"KAPPA (primary)      = {kappa:.6e}   [cross-check {kappa_local:.6e}]")
    print(f"=> w0 = g/(K*kappa): K=1000, g=1  ->  w0 = {1.0 / (1000 * kappa):.6f}")
    print("kappa(f0) table:")
    for f0, I0, gg, kk in zip(f0s, I0s, dfdI, kappa_tab):
        print(f"   f0={f0:4.0f} Hz  I0={I0:6.3f}  df/dI={gg:6.4f}  kappa={kk:.6e}")
    print(f"saved                {out}")


if __name__ == "__main__":
    main()
