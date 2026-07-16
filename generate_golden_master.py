#!/usr/bin/env python3
"""
generate_golden_master.py — T0 generator.

Runs the composed reference network (M0 integrator + M1 CSR/ring/STDP) for a small
but representative configuration and streams the spikes to reference/golden_master.aer
in the SPEC §5 little-endian format. This file is the deterministic ground truth the
Rust `step_block` kernel is diffed against.

Config: N=1000, K=100, g=1.0, seed=42, 10_000 steps (1.0 s at dt=0.1 ms).
"""

import hashlib
import os

import numpy as np

from criticalcortex.connectome import build_connectome
from criticalcortex.simulation import SimParams, run_reference, aer_to_bytes

N, K, G, SEED, STEPS = 1000, 100, 1.0, 42, 10_000


def build_golden_bytes() -> tuple[bytes, dict]:
    """Deterministically produce the golden-master AER bytes plus run statistics."""
    conn = build_connectome(n=N, k=K, g=G, seed=SEED)
    params = SimParams(seed=SEED)                    # mu=3.5, sigma=3.0 (validated ~9.5 Hz)
    net = run_reference(conn, params, steps=STEPS)
    count = int(net.aer_count[0])
    data = aer_to_bytes(net.aer_step, net.aer_neuron, count,
                        n=N, dt_ms=params.dt, seed=SEED, g=G)
    stats = dict(
        spikes=count,
        rate_hz=count / (N * STEPS * params.dt / 1000.0),
        bytes=len(data),
        active_neurons=int(np.unique(net.aer_neuron[:count]).size),
        v_finite=bool(np.all(np.isfinite(net.v))),
        weight_finite=bool(np.all(np.isfinite(conn.weight))),
    )
    return data, stats


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ref_dir = os.path.join(here, "reference")
    os.makedirs(ref_dir, exist_ok=True)
    out = os.path.join(ref_dir, "golden_master.aer")

    data, stats = build_golden_bytes()
    with open(out, "wb") as f:
        f.write(data)

    assert stats["v_finite"] and stats["weight_finite"], "non-finite state — unstable run"
    assert len(data) == 64 + 8 * stats["spikes"], "AER size does not match header contract"

    print("== T0 golden master ==")
    print(f"config               N={N} K={K} g={G} seed={SEED} steps={STEPS} (1.0 s)")
    print(f"spikes               {stats['spikes']}  (~{stats['rate_hz']:.1f} Hz mean, "
          f"{stats['active_neurons']}/{N} neurons active)")
    print(f"file                 {out}  ({stats['bytes']} bytes = 64 header + 8*{stats['spikes']})")
    print(f"sha256               {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
