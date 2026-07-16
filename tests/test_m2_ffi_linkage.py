"""
tests/test_m2_ffi_linkage.py — FFI bridge contract (SPEC §2.2), signature-agnostic.

Verifies the zero-allocation Rust<->NumPy bridge through the production driver, so it
tracks the kernel signature as it evolves (M2 skeleton -> M3 physics -> SOC): the kernel
mutates the borrowed NumPy buffers IN PLACE (same data pointer, no copy) and reports
StepStats.alloc_bytes == 0.

BUILD FIRST:  maturin develop && python3 tests/test_m2_ffi_linkage.py
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


def test_ffi_zero_copy_and_zero_alloc():
    if not kernel_available():
        raise _Skip("extension not built (run `maturin develop`)")
    conn = build_connectome(n=64, k=8, g=1.0, seed=0)
    p = SimParams(seed=0)
    state = allocate_kernel_state(conn, p)
    net = state["net"]

    ptr_v, ptr_t = net.v.ctypes.data, net.t_last.ctypes.data
    v_before = net.v.copy()

    stats = step_kernel(state, 0, 25)

    # zero-allocation contract
    assert stats.alloc_bytes == 0
    # zero-copy: the SAME underlying buffers were mutated in place (no realloc/copy-back)
    assert net.v.ctypes.data == ptr_v
    assert net.t_last.ctypes.data == ptr_t
    # the kernel actually advanced the borrowed state in place
    assert not np.array_equal(net.v, v_before)
    assert np.all(np.isfinite(net.v))
    for attr in ("n_spikes", "sum_sq", "alloc_bytes"):
        assert hasattr(stats, attr)


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
