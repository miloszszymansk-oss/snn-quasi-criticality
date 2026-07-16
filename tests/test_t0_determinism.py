"""
tests/test_t0_determinism.py — Milestone T0 acceptance suite (SPEC §7.1).

Locks the determinism contract of the composed reference simulator (the golden-master
generator). Runs under pytest AND standalone. Regenerate the golden master first:
    python3 generate_golden_master.py
"""

import hashlib
import os
import struct
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from criticalcortex.connectome import build_connectome
from criticalcortex.simulation import SimParams, run_reference, hash_noise, AER_MAGIC
from generate_golden_master import build_golden_bytes, N, K, G, SEED

REF = os.path.join(_ROOT, "reference", "golden_master.aer")


# ===========================================================================
# T0.1 — bit reproducibility: identical config => byte-identical AER.
# ===========================================================================
def test_bit_reproducibility():
    a, _ = build_golden_bytes()
    b, _ = build_golden_bytes()
    assert a == b                                              # two runs, identical bytes
    with open(REF, "rb") as f:
        disk = f.read()
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(disk).hexdigest()   # matches committed


# ===========================================================================
# T0.2 — AER format contract (SPEC §5): header parses, records well-formed.
# ===========================================================================
def test_aer_header_and_records_contract():
    with open(REF, "rb") as f:
        data = f.read()
    magic, ver, n, dt, seed, g, n_events, flags = struct.unpack("<8sIQfQfQI", data[:48])
    assert magic == AER_MAGIC and ver == 1
    assert n == N and seed == SEED and abs(g - G) < 1e-9
    assert n_events == (len(data) - 64) // 8                   # header count == record count
    rec = np.frombuffer(data[64:], dtype=[("step", "<u4"), ("neuron", "<u4")])
    assert rec.size == n_events
    assert np.all(np.diff(rec["step"].astype(np.int64)) >= 0)  # sorted by step
    assert int(rec["neuron"].max()) < N                        # neuron ids in range


# ===========================================================================
# T0.3 — thread-count invariance (skeleton).
# True multi-thread equality is validated against the Rust parallel kernel. Here we
# lock the determinism that makes it possible: the streamed result is invariant to how
# n_steps is BLOCKED (state carried across calls), because the per-step Philox noise is
# keyed on the ABSOLUTE step — not on block position or thread partition (SPEC §4.3).
# ===========================================================================
def test_thread_count_invariance():
    conn = build_connectome(n=N, k=K, g=G, seed=SEED)
    p = SimParams(seed=SEED)
    one = run_reference(conn, p, steps=4000, block=None)       # one block of 4000
    many = run_reference(conn, p, steps=4000, block=337)       # awkward blocking on purpose
    c1, c2 = int(one.aer_count[0]), int(many.aer_count[0])
    assert c1 == c2
    assert np.array_equal(one.aer_step[:c1], many.aer_step[:c2])
    assert np.array_equal(one.aer_neuron[:c1], many.aer_neuron[:c2])
    # per-step noise is a pure function of (seed, step): reproducible AND step-dependent
    assert np.array_equal(hash_noise(SEED, 123, 64), hash_noise(SEED, 123, 64))
    assert not np.array_equal(hash_noise(SEED, 123, 64), hash_noise(SEED, 124, 64))


# ===========================================================================
# Standalone harness (pytest ignores __main__).
# ===========================================================================
if __name__ == "__main__":
    import traceback

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
    print(f"{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
