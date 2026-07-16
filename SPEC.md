# CriticalCortex — SPEC.md

**A million-neuron spiking network self-tuned to the edge of chaos.**
Spec-first architectural contract + executable test manifest. **No execution code ships until this manifest is green.**

Version 0.2 · status: **contract frozen for review** · integrator/kernel: unimplemented (all tests fail by construction). *(v0.2 — added the self-organized-criticality gate as milestone M3; former Performance→M4, Visualization→M5.)*

---

## 0. Purpose & non-goals

**Purpose.** Simulate `N ≈ 10⁶` Izhikevich neurons over a random sparse synaptic graph, drive the network across a control parameter `g`, and demonstrate — under falsifiable statistical tests — that at `g = g_c` the collective dynamics are those of a **critical branching process**: scale-free neuronal avalanches with the mean-field exponents `τ = 3/2`, `α = 2`, `γ = 2`, satisfying the crackling-noise scaling relation and a finite-size-scaling collapse.

**The thesis being tested (Cognitive Science payload).** Cortical computation is optimized at a second-order phase transition between quiescence and saturation (the *critical brain hypothesis*, Beggs & Plenz 2003). Criticality is not assumed — it is an **emergent, measured** property that the acceptance suite can reject. The *definitive* form of the claim (milestone M3) is stronger: the network must **self-organize** to the critical point with no external tuning, driven by homeostatic synaptic depression (Levina–Herrmann–Geisel) — criticality as a dynamical attractor, not a fine-tuned coincidence.

**Non-goals.** (1) Not a biophysically complete cortical column — no dendritic compartments, no detailed channel kinetics. (2) Not real-time BCI. (3) Not a machine-learning model — there is no loss function, no backprop, no dataset. The "learning" is unsupervised STDP and homeostasis only. (4) The frontend visualizes; it never feeds the analysis pipeline (see §3.6, subsampling constraint).

---

## 1. The central invariant (the whole contract in one line)

> **The system exists to produce, and be falsified by, the statistical signature of a critical branching process.**

Everything downstream — the integrator, the CSR scatter, the ring buffer, the Rust FFI — is *implementation detail in service of one measurable object*: the avalanche ensemble `{(Sᵢ, Tᵢ)}` and its exponents. If the exponents and their scaling relation do not hold at `g_c` (and do **not** hold away from it), the system is wrong regardless of how fast it runs.

**The bridge from micro-model to branching process.** For a sparse random graph with mean in-degree `K`, linearizing spike propagation around the quiescent state gives a branching ratio

```
σ = K · p_transmit(g)          where p_transmit = P(one presynaptic spike triggers a postsynaptic spike)
```

We **construct the weights so that `σ(g) = g` to linear order** (§4.4): each synaptic weight is normalized by `1/(K · ⟨PSP-to-spike gain⟩)`, making the control parameter *definitionally* the branching ratio. Criticality is therefore predicted at **`g_c = 1`**; the suite locates the empirical `g_c` and asserts it equals 1 within tolerance. This mapping is what makes the statistical-physics tests exact rather than aspirational.

---

## 2. Architecture & module boundaries

Six modules. Two run at **build time** (deterministic, seeded, NumPy). One is the **hot loop** (Rust). Three are **offline** (recorder, analysis, frontend). The boundaries are drawn so that the only performance-critical code is Module 3, and the only *correctness-critical statistics* live in Module 5 — they never share a process.

```
CONFIG (frozen TOML + seed) ─────────────┐
                                          ▼
┌──────────────────────────────┐  CSR + param arrays   ┌──────────────────────────────┐
│ 1  CONNECTOME BUILDER  (Py)  │ ───────────────────►  │  2  INTEGRATOR  (Py, thin)   │
│    seed → perm π → CSR(out)  │                        │     owns all arrays          │
│    → weights, delays          │  preallocated state   │     drives the step loop     │
└──────────────────────────────┘ ◄──── (borrowed) ────►│     calls kernel(n_steps)    │
             │                                          └───────────────┬──────────────┘
             │ layout_bytes(cfg)                     FFI: rust-numpy, borrowed slices │
             ▼                                                          ▼
      (memory contract §3)                            ┌──────────────────────────────┐
                                                      │  3  ROUTING KERNEL  (Rust)   │
                                                      │  subthreshold → threshold →  │
                                                      │  ring-deliver → CSR scatter →│
                                                      │  STDP RMW    [ZERO-ALLOC]    │
                                                      └──────┬──────────────────┬─────┘
                                       full AER spikes │                  │ rate frames
                                                       ▼                  ▼
                                        ┌───────────────────┐   ┌────────────────────┐
                                        │ 4  RECORDER       │   │ 6  FRONTEND BUS    │
                                        │  binary AER dump  │   │  quantized → WebGL │
                                        └─────────┬─────────┘   └────────────────────┘
                                                  ▼
                                        ┌───────────────────┐
                                        │ 5  ANALYSIS  (Py) │  ◄── the pytest manifest lives here
                                        │  σ, P(S), γ, FSS  │      (§7). Consumes FULL AER only.
                                        └───────────────────┘
```

### 2.1 Module contracts (interfaces are the API; violate them and the build fails)

| # | Module | Language | Owns | Consumes | Produces | Alloc policy |
|---|--------|----------|------|----------|----------|--------------|
| 1 | Connectome Builder | Python/NumPy | topology + params | `Config` | `CSR`, param SoA, permutation π | build-time only |
| 2 | Integrator | Python (thin) | all state arrays | `CSR`, `Config` | drives loop, owns lifetimes | **allocates once, before loop** |
| 3 | Routing Kernel | Rust (PyO3) | nothing (borrows all) | borrowed slices | mutates state in place, appends AER | **zero heap alloc in loop** |
| 4 | Recorder | Rust→disk | AER file handle | AER sink | `.aer` binary (§5) | streaming, bounded buffer |
| 5 | Analysis | Python/NumPy/SciPy | statistics | full `.aer` | exponents, p-values, verdict | offline, unbounded |
| 6 | Frontend Bus | Rust→bytes | frame ring | rate frames + subsampled spikes | quantized WebGL frames (§6) | preallocated ring |

**Hard rule:** Module 5 reads the **full AER dump** (Module 4), never the frontend feed (Module 6). Subsampling biases σ and the exponents (Priesemann); the two consumers are physically separated so this can never happen by accident.

### 2.2 The FFI contract (Rust ⇄ Python)

The entire hot loop is a **single entrypoint**. Python allocates everything; Rust borrows and mutates. No ownership transfer, no allocation, no callback into Python inside the loop.

```rust
// criticalcortex_kernel/src/lib.rs  (signature is the contract; body unimplemented)
#[pyfunction]
fn step_block(
    // --- neuron state (SoA, mutated in place) ---
    v:        PyReadwriteArray1<f32>,   // membrane potential      [N]
    u:        PyReadwriteArray1<f32>,   // recovery variable       [N]
    g_exc:    PyReadwriteArray1<f32>,   // exc. synaptic current   [N]
    g_inh:    PyReadwriteArray1<f32>,   // inh. synaptic current   [N]
    refrac:   PyReadwriteArray1<u16>,   // refractory countdown    [N]
    t_last:   PyReadwriteArray1<i32>,   // last-spike step (STDP)   [N]
    // --- connectivity (CSR out-edges; weight mutable under STDP) ---
    indptr:   PyReadonlyArray1<i64>,    //                         [N+1]
    indices:  PyReadonlyArray1<i32>,    // post neuron ids         [M]
    weight:   PyReadwriteArray1<f32>,   //                         [M]
    delay:    PyReadonlyArray1<u16>,    // axonal delay (steps)    [M]
    // --- delayed delivery + plasticity traces (preallocated arenas) ---
    ring:     PyReadwriteArray2<f32>,   // g_ring                  [D_max, N]
    x_pre:    PyReadwriteArray1<f32>,   // presyn STDP trace       [N]
    x_post:   PyReadwriteArray1<f32>,   // postsyn STDP trace      [N]
    // --- params + rng + output ---
    params:   &Params,                  // dt, a,b,c,d, tau_syn, A±, g, ...
    rng:      &mut Philox4x32,          // counter-based, deterministic
    n_steps:  u32,
    aer_out:  &mut SpikeSink,           // preallocated append buffer
) -> StepStats;                          // {n_spikes, sum_sq, alloc_bytes(must be 0)}
```

**Invariants the FFI enforces (tested in T0/T4):**
- *Borrow-only:* every array is owned by Python; Rust holds a mutable/immutable slice for the call duration.
- *Zero-alloc steady state:* `StepStats.alloc_bytes == 0` after warmup. Enforced by a bump-allocator guard in test builds.
- *Determinism under threads:* RNG is counter-based keyed on `(neuron_id, step)`; the scatter uses a deterministic reduction order (§3.4), so output is **bit-identical for any thread count**.

---

## 3. Memory layout (exact — this is where the bottleneck is won or lost)

All arrays are **Structure-of-Arrays**, `float32` unless noted, 64-byte aligned (one cache line; enables aligned AVX loads). Rationale is stated per object because the layout *is* the performance contract.

### 3.1 Neuron state (length `N`, contiguous)

| array | dtype | bytes/neuron | purpose |
|-------|-------|--------------|---------|
| `v` | f32 | 4 | membrane potential |
| `u` | f32 | 4 | Izhikevich recovery |
| `g_exc`, `g_inh` | f32 | 8 | current accumulators |
| `a,b,c,d` | f32 | 16 | per-neuron Izhikevich params (heterogeneous → RS/IB/CH mix) |
| `refrac` | u16 | 2 | refractory countdown |
| `t_last` | i32 | 4 | last spike step (STDP) |
| `is_inhib` | bit | ⅛ | Dale's-law sign, packed bitset |
| **total** | | **≈ 38 B** | |

At `N = 10⁶` → **~38 MB**. Exceeds L3, so the subthreshold sweep is a **streaming, fully sequential** read/write of 38 MB/step — bandwidth-bound but hardware-prefetchable (near-peak). This is the *easy* half. The scatter (§3.3) is the hard half.

### 3.2 Sparse connectivity — CSR, out-edge oriented (length `M = N·K`)

Row `i` = presynaptic neuron; its slice `indices[indptr[i]:indptr[i+1]]` = postsynaptic targets. This orientation is chosen because forward spike propagation iterates over *spiking presynaptic* neurons and writes to *their targets* — the natural gather-then-scatter.

| array | dtype | width | why this width |
|-------|-------|-------|----------------|
| `indptr` | **i64** | N+1 | `M ≈ 10⁹ > 2³¹`; row offsets **must** be 64-bit or they overflow |
| `indices` | **i32** | M | neuron ids `< 2³¹`, so 32-bit is safe and halves bandwidth vs i64 |
| `weight` | f32 | M | PSP amplitude; **mutable** under STDP |
| `delay` | **u16** | M | axonal delay in steps; `D_max < 65536` (@dt=0.1ms → 6.5 s max, ample) |

**Bytes per synapse: 4 (idx) + 4 (w) + 2 (delay) = 10 B.** At `M = 10⁹` → **~10 GB** — the dominant object; it sets the machine requirement (≥ 24 GB RAM at full scale). Tests run at `M = 10⁶–10⁷`.

> **Design decision — single CSR, no transpose.** STDP naïvely needs in-edges (for potentiation on postsynaptic spikes), which would force a second CSC copy and *double* connectivity memory to ~20 GB. We avoid this: STDP is **pair-based with per-neuron traces** (§3.5) evaluated entirely along the out-edge list at spike-delivery time (Morrison, Diesmann & Gerstner 2008, "online/axonal" scheme). Correctness of this transpose-free path is pinned by test **T2.3** against a dense brute-force reference. Cost of the alternative (exact per-post-spike potentiation) is documented: +10 GB for the CSC.

### 3.3 Delayed delivery — dense conductance ring buffer

Fixed max delay `D_max` steps. `ring` is `f32[D_max, N]`: slot `s = (step + delay) mod D_max` holds pending current per neuron.

- **Deliver** (the hot path): for each out-edge `(j, w, d)` of a spiking neuron, `ring[(step+d) % D_max, j] += w`. This `+=` into a random `j` across a `D_max·N·4` = ~80 MB structure (`D_max=20`) is **the random scatter** — the cache-hostile, DRAM-bound core operation, and exactly the CSR-bandwidth bottleneck to beat.
- **Drain** (sequential, cheap): at each step, `g_exc/g_inh += ring[step % D_max]`, then zero that row.

Chosen over an event-list ring because it is **zero-allocation** and bandwidth-predictable (no per-slot dynamic arenas, no reallocation under bursty load). The tradeoff — it touches `N` even when few spikes are pending — is accepted; at criticality the network is never that sparse.

### 3.4 Cache-locality permutation (baked into the layout at build time)

The scatter's miss rate is dominated by how far apart `j` targets land in memory. The **Connectome Builder applies a permutation `π` to neuron ids** (space-filling / METIS graph partition minimizing edge-cut) so presynaptic neurons and their targets are near-neighbors in the address space. CSR is materialized *in permuted order*; π is stored so AER can be de-permuted for analysis. Deterministic parallel scatter uses **tile-owned target ranges** (each thread owns a contiguous neuron block) so writes never race and the reduction order is fixed → bit-reproducibility (T0).

### 3.5 STDP & homeostatic state (per-neuron traces — O(N), not O(M))

Pair-based STDP with exponential traces needs only **two length-`N` arrays**: `x_pre`, `x_post`. On a spike, the trace increments; between spikes it decays `x *= exp(-dt/τ_stdp)`. The weight update for synapse `i→j` is evaluated when `i` fires: `Δw_ij = A₊·x_pre[i] − A₋·x_post[j]`, applied along the out-edge list. The **per-synapse cost is one RMW of `weight[e]`** (read 4 B + write 4 B); the traces stay O(N) in L2. This is the single trick that keeps plasticity affordable at `M = 10⁹`.

**Homeostatic driver (SOC, milestone M3).** The Levina–Herrmann–Geisel mechanism adds dynamical synaptic *depression*: a resource variable depletes on transmission and recovers on a slow timescale `τ_homeo ≫ τ_syn`, the negative feedback that pulls the branching ratio `σ → 1` with no external tuning. Following the same O(N)-not-O(M) discipline as the STDP traces, resources are **pooled per presynaptic neuron** — one `f32[N]` array `r_avail`, depleted when neuron `i` fires and recovered as `r += (1 − r)·dt/τ_homeo`; the transmitted weight becomes `w_ij · r_i · u_release`. This adds a single sequential O(N) stream (~4 MB at `N = 10⁶`), negligible against the scatter. **Fallback (mirrors §3.2):** if per-neuron pooling fails to reproduce self-organization under test T3-SOC, revert to per-synapse resources — O(M), +4 GB — documented and gated by that same test.

### 3.6 Memory roofline (the performance target derives from this, not from vibes)

Bytes moved per **delivered synaptic event**, scatter + STDP:

```
read  indices[e]        4 B
read  weight[e]         4 B     (RMW: +4 B write under STDP)
RMW   ring[s, j]        8 B     (4 read + 4 write)
read  x_post[j]         4 B     (STDP)
------------------------------
≈ 24–28 B / event
```

At a measured `B_stream` (STREAM triad; ~50 GB/s on a dual-channel DDR4 desktop): ceiling ≈ `50e9 / 26 ≈ 1.9×10⁹ events/s`. Random-access efficiency ~30–50% of peak → **realistic 0.5–0.9×10⁹ events/s single-socket.** Full-scale biological load = `N·rate·K = 10⁶·5Hz·10³ = 5×10⁹ events/s`, so real-time requires ~6–10 cores; the SLO (§8) is stated as a *fraction of measured roofline*, never an absolute, because it is hardware-bound by construction.

---

## 4. Integrator contract (Module 2 + the numeric core of Module 3)

### 4.1 Model — Izhikevich (heterogeneous)

```
v' = 0.04 v² + 5 v + 140 − u + I_syn(t) + I_noise
u' = a (b v − u)
if v ≥ +30 mV:   emit spike;  v ← c;  u ← u + d
```

### 4.2 Discretization (fixed `dt = 0.1 ms`)
- `v`: forward Euler in **two half-`dt` substeps** (Izhikevich 2003 — required for stability near the quadratic blow-up). Global order 1; verified by Richardson extrapolation (T1.3).
- `u`: forward Euler, one `dt`.
- Synaptic current `g_{exc,inh}`: **exponential Euler**, `g *= exp(-dt/τ_syn)` — *exact* for linear decay (verified to fp-eps, T1.2).

### 4.3 Determinism contract
Given `(seed, Config)`, the entire spike train is **bit-reproducible** across runs and thread counts (T0). Counter-based RNG (Philox), deterministic reduction (§3.4). This is non-negotiable: golden-master tests depend on it.

### 4.4 Weight normalization (the σ=g construction)
At build, excitatory weights are scaled so the linearized branching ratio equals `g`: `w₀ = g / (K_exc · κ)`, where `κ = ∂(P_spike)/∂(PSP)` is the measured single-neuron gain at the operating point (calibrated once by T1.1's f–I curve). Inhibition set for an E/I balance target. **Consequence:** the criticality knob `g` is the branching ratio by construction; `g_c = 1` is a prediction, not a fit.

---

## 5. Recorder & AER format (Module 4)

Address-Event Representation, little-endian, streamed:

```
HEADER (64 B): magic "CCXAER01" | u32 version | u64 N | f32 dt_ms | u64 seed
             | f32 g | u64 n_events | u32 flags | pad
RECORD (8 B):  u32 step | u32 neuron_id            // sorted by step, then id
```

Full fidelity (every spike) — this file is the **sole input to the analysis manifest**. Downsampling happens only in Module 6. A 2M-step run at 5 Hz mean rate over 10⁶ neurons ≈ 10¹⁰ events ≈ 80 GB → recorder supports chunked/compressed shards; analysis streams them.

---

## 6. Frontend data bus (Module 6 → WebGL)

Two channels, both **quantized and preallocated** (zero-alloc, mirroring the N-body frontend):
- **Rate field:** neurons binned to a `256×256` grid, per-frame `u8` activity (population rate) → a single `RG8`/`R8` texture. 60 fps decoupled from sim `dt`.
- **Spike stream (subsample):** a fixed-capacity ring of `(x, y)` for a deterministic subsample (≤ 2¹⁶ neurons), appended to a persistent GPU vertex buffer; decay handled in-shader. **This subsample never reaches Module 5.**

Byte layout is a frozen contract so the WebGL client is a pure consumer:
```
FRAME: u32 step | u16 grid_w | u16 grid_h | u8[grid_w*grid_h] rate | u16 n_spk | (u16 x,u16 y)[n_spk]
```

---

## 7. Test manifest

Every test below **fails today** — the modules it imports are unimplemented. That is the point: the manifest is the contract, written before the code. A test suite that only asserts positive results is worthless, so §7.4 includes **negative controls** that must *fail off-criticality* — the tests must have teeth.

**Avalanche extraction protocol (frozen, shared by all §7.4 tests).** From the full AER: (1) bin spikes into windows of width `Δ = ⟨ISI⟩` (mean inter-spike interval across the network); (2) an **avalanche** = a maximal run of consecutive non-empty bins bracketed by empty bins; (3) **size** `S` = total spikes in the run, **duration** `T` = number of bins. Bin width is itself swept (T3.7) because true criticality is robust to it; a spurious power law is not.

```python
# conftest.py — fixtures. Signatures are the contract; bodies raise NotImplementedError.
import numpy as np, pytest
from criticalcortex import Config, run, run_and_dump
from criticalcortex.analysis import (
    avalanches_from_aer, powerlaw_mle, branching_ratio_MR, branching_ratio_naive,
    mean_size_given_duration_slope, collapse_shapes, fss_collapse, is_bimodal,
)

G_C = 1.0   # predicted critical branching ratio (weights normalized so σ=g, §4.4)

@pytest.fixture(scope="session")
def crit(tmp_path_factory):
    cfg = Config(seed=2025, n=100_000, k=1000, g=G_C,
                 steps=2_000_000, warmup=200_000, dt_ms=0.1)
    aer = run_and_dump(cfg, tmp_path_factory.mktemp("crit") / "c.aer")
    return avalanches_from_aer(aer, bin_dt="mean_isi")   # full AER only (§3.6)
```

### 7.1 — T0 · Determinism & golden master

```python
def test_bit_reproducibility(tmp_path):
    cfg = Config(seed=1234, n=10_000, k=100, steps=50_000)
    a = run_and_dump(cfg, tmp_path / "a.aer").sha256()
    b = run_and_dump(cfg, tmp_path / "b.aer").sha256()
    assert a == b                                   # identical seed → identical stream

def test_thread_count_invariance():
    cfg = Config(seed=1, n=10_000, k=100, steps=20_000)
    assert run(cfg, threads=1).spike_hash == run(cfg, threads=8).spike_hash
    # deterministic parallel scatter (§3.4) — bit-identical across core counts
```

### 7.2 — T1 · Single-neuron numerics (no network)

```python
def test_izhikevich_fI_matches_reference():
    # isolated regular-spiking neuron; f–I curve calibrates κ used in §4.4
    ref = np.load("reference/izh_rs_fI.npz")
    f = measure_fI(a=.02, b=.2, c=-65, d=8, I=ref["I"], dt=0.1, T_ms=1000)
    assert np.allclose(f, ref["f"], atol=0.5)       # Hz

def test_exponential_synapse_is_exact():
    tau, dt, g0 = 5.0, 0.1, 1.0
    g = simulate_decay(g0, tau, dt, n=1000)
    t = np.arange(1, 1001) * dt
    assert np.max(np.abs(g - g0 * np.exp(-t / tau))) < 1e-6   # exact to fp-eps

def test_integrator_convergence_order():
    e1, e2 = traj_error(dt=0.1), traj_error(dt=0.05)
    assert 1.7 < e1 / e2 < 2.3                       # global order 1 (half-step Euler on v)
```

### 7.3 — T2 · CSR routing, delays, STDP (small nets, dense reference)

```python
def test_csr_scatter_matches_dense():
    net = random_net(n=200, k=20, seed=3)
    spikes = np.array([5, 17, 88])
    got = csr_deliver(net.csr, spikes)               # kernel path
    W = net.to_dense()                               # (post × pre)
    assert np.allclose(got, W[:, spikes].sum(axis=1), atol=1e-6)

def test_delay_ring_delivers_at_exact_step():
    ring = DelayRing(d_max=32, n=10)
    ring.schedule(target=7, weight=0.5, at_step=12)
    for s in range(12):
        assert ring.drain(s)[7] == 0.0
    assert ring.drain(12)[7] == 0.5                  # not one step early, not one late

def test_stdp_pair_window_matches_analytic():
    # transpose-free out-edge STDP (§3.5) must reproduce the exact pair window
    for lag_ms in (-40, -20, -5, 5, 20, 40):
        dw = single_pair_experiment(lag_ms)
        assert np.isclose(dw, stdp_window(lag_ms), rtol=1e-3)

def test_stdp_transpose_free_equals_dense_reference():
    # the CSR-only online scheme == brute-force in-edge computation, over a full run
    net = random_net(n=300, k=30, seed=11)
    w_kernel = run_stdp(net, steps=5_000).weights
    w_dense  = run_stdp_dense_reference(net, steps=5_000).weights
    assert np.allclose(w_kernel, w_dense, rtol=1e-4)
```

### 7.4 — T3 · Statistical-physics invariants (the crown jewels)

Tolerances are **calibrated against a synthetic critical Galton–Watson process** (Poisson offspring, mean 1; see Appendix A). That calibration showed: the size exponent recovers to <0.2%, while the duration exponent and γ carry a ~5% finite-window/discreteness bias — so the suite pins `τ` tightly, `α` loosely, and tests the **scaling relation as a self-consistency check between independently measured exponents**, which closed to 3.2% on synthetic ground truth. This is the statistically honest formulation.

```python
def test_locate_critical_point():
    # sweep g; σ(g) must cross 1 at g_c ≈ 1 (the §4.4 construction)
    gs = np.linspace(0.90, 1.10, 21)
    sig = [branching_ratio_MR(run(Config(seed=7, n=50_000, k=1000, g=g,
                                         steps=500_000)).rate) for g in gs]
    g_c = gs[np.argmin(np.abs(np.array(sig) - 1.0))]
    assert abs(g_c - 1.0) < 0.02

def test_branching_parameter_sigma(crit):
    sigma_mr    = branching_ratio_MR(crit.rate)      # Wilting & Priesemann 2018: subsampling-invariant
    sigma_naive = branching_ratio_naive(crit)
    assert abs(sigma_mr    - 1.0) < 0.02             # the rigorous estimator
    assert abs(sigma_naive - 1.0) < 0.05             # naïve is biased; looser bound

def test_size_distribution_is_powerlaw(crit):
    fit = powerlaw_mle(crit.sizes, xmin="ks")        # Clauset–Shalizi–Newman
    assert 1.45 <= fit.tau <= 1.55                   # target 3/2
    assert fit.ks_pvalue > 0.10                      # power law NOT rejected (1000-resample bootstrap)
    assert fit.vuong_p_vs_lognormal > 0.10           # lognormal not a significantly better fit

def test_duration_distribution_is_powerlaw(crit):
    fit = powerlaw_mle(crit.durations, xmin="ks")
    assert 1.80 <= fit.alpha <= 2.15                 # target 2, loose (borderline-exponent bias)
    assert fit.ks_pvalue > 0.10

def test_crackling_scaling_relation(crit):
    tau   = powerlaw_mle(crit.sizes).tau
    alpha = powerlaw_mle(crit.durations).alpha
    gamma_pred = (alpha - 1.0) / (tau - 1.0)         # Sethna crackling-noise relation
    gamma_meas = mean_size_given_duration_slope(crit)  # ⟨S⟩(T) ~ T^γ, independent measurement
    # the DEEP test: two independently measured exponents must satisfy the relation
    assert abs(gamma_meas - gamma_pred) / gamma_meas < 0.10

def test_avalanche_shape_collapse(crit):
    # rescaled mean temporal profiles collapse onto one universal curve;
    # best-fit collapse exponent must equal γ − 1
    exp_collapse, residual = collapse_shapes(crit, durations=range(20, 120))
    gamma = mean_size_given_duration_slope(crit)
    assert abs(exp_collapse - (gamma - 1.0)) < 0.15
    assert residual < 0.05

def test_finite_size_scaling_collapse():
    runs = [avalanches_from_aer(run(Config(seed=s, n=n, k=1000, g=G_C,
                                           steps=1_000_000)).spikes)
            for n, s in [(20_000, 1), (50_000, 2), (100_000, 3), (200_000, 4)]]
    tau, D, residual = fss_collapse(runs)            # P(S,N)=S^{-τ} F(S/N^D)
    assert 1.45 <= tau <= 1.55
    assert abs(D - 2.0/3.0) < 0.10                   # mean-field cutoff S_c ~ N^{2/3} (ER critical window)
    assert residual < 0.05

def test_exponents_robust_to_bin_width():
    raw = run(Config(seed=5, n=100_000, k=1000, g=G_C, steps=1_000_000)).spikes
    taus = [powerlaw_mle(avalanches_from_aer(raw, bin_dt=f).sizes).tau
            for f in (0.5, 1.0, 2.0)]               # fractions of ⟨ISI⟩
    assert np.ptp(taus) < 0.10                        # real criticality is bin-robust

# ---- NEGATIVE CONTROLS: the suite MUST reject criticality away from g_c ----
@pytest.mark.parametrize("g", [0.90, 1.10])
def test_offcriticality_is_rejected(g):
    av = avalanches_from_aer(run(Config(seed=9, n=100_000, k=1000, g=g,
                                        steps=1_000_000)).spikes)
    fit   = powerlaw_mle(av.sizes, xmin="ks")
    sigma = branching_ratio_MR(av.rate)
    if g < 1.0:                                      # subcritical → exponential, σ<1
        assert sigma < 0.98 and fit.ks_pvalue < 0.10
    else:                                            # supercritical → bump/bimodal, σ>1
        assert sigma > 1.02 and is_bimodal(av.sizes)
```

#### T3-SOC · Self-organized criticality — the M3 acceptance gate

These tests are **gated to milestone M3, deliberately not M1/M2.** M1/M2 establish criticality by *manually tuning* `g → g_c`, which de-risks the Rust kernel against a known target first. The tests below verify the deeper, definitive claim: the network **finds `g_c` on its own** through the homeostatic Levina–Herrmann–Geisel depressing-synapse driver (§3.5) — the cognitive-science payload, criticality as an emergent attractor rather than a coincidence. **Precondition:** separation of timescales `τ_homeo ≫ τ_syn, τ_mem` (§9); without it there is no self-organization, only slow drift.

```python
# ============================================================================
# T3-SOC — SELF-ORGANIZED CRITICALITY.  Milestone M3 gate (NOT M1/M2).
# Homeostasis ON; g0 is only an INITIAL CONDITION, never a tuned target.
# ============================================================================

@pytest.mark.milestone("M3")
@pytest.mark.parametrize("g0", [0.80, 1.30])          # start sub- AND super-critical
def test_self_organizes_to_criticality(g0, tmp_path):
    cfg = Config(seed=31, n=100_000, k=1000, g=g0, dt_ms=0.1,
                 steps=6_000_000, warmup=0,
                 homeostasis="depressing_synapses_lhg",   # the SOC driver (§3.5)
                 tau_homeo_ms=10_000.0)                    # slow ⇒ timescale separation
    trace = run(cfg)                                       # records rate(t) across the run

    # (1) rule out the trivial DEAD fixed point (σ→0 is also stable)
    assert trace.mean_rate_hz[-1] > 0.5

    # (2) the effective branching ratio self-organizes to a neighborhood of 1
    sigma = branching_ratio_MR(trace.rate[-2_000_000:])   # stationary tail only
    assert abs(sigma - 1.0) < 0.03

    # (3) it reached GENUINE criticality, not σ≈1 by coincidence:
    #     the stationary avalanche law is the mean-field power law
    av  = avalanches_from_aer(dump_tail(trace, tmp_path / "soc.aer"), bin_dt="mean_isi")
    fit = powerlaw_mle(av.sizes, xmin="ks")
    assert 1.45 <= fit.tau <= 1.55
    assert fit.ks_pvalue > 0.10

@pytest.mark.milestone("M3")
def test_soc_fixed_point_is_initial_condition_independent():
    # the definitive SOC signature: ONE attractor, reached from both directions
    def final_sigma(g0):
        t = run(Config(seed=31, n=100_000, k=1000, g=g0, dt_ms=0.1, steps=6_000_000,
                       homeostasis="depressing_synapses_lhg", tau_homeo_ms=10_000.0))
        return branching_ratio_MR(t.rate[-2_000_000:])
    assert abs(final_sigma(0.80) - final_sigma(1.30)) < 0.03

@pytest.mark.milestone("M3")
def test_soc_approaches_criticality_from_above_with_N():
    # LHG finite-size law: the self-organized state is slightly SUPER-critical at
    # finite N and → exactly critical as N→∞. The sharpest (most fragile) claim.
    excess = {}
    for n in (25_000, 50_000, 100_000):
        t = run(Config(seed=8, n=n, k=1000, g=1.20, dt_ms=0.1, steps=4_000_000,
                       homeostasis="depressing_synapses_lhg", tau_homeo_ms=10_000.0))
        excess[n] = branching_ratio_MR(t.rate[-1_500_000:]) - 1.0
    assert excess[25_000] > excess[50_000] > excess[100_000] >= -0.01   # monotone → 0⁺
```

### 7.5 — T4 · Performance & resource SLOs

```python
BYTES_PER_EVENT = 26   # §3.6 roofline

def test_scatter_throughput(reference_hw):
    ev_s      = bench_scatter(n=1_000_000, k=1000, active_frac=0.05)
    stream_bw = measure_stream_triad()               # measured, not assumed
    assert ev_s >= 5e8                               # ≥ 0.5 G events/s single-socket
    assert ev_s * BYTES_PER_EVENT >= 0.40 * stream_bw  # ≥ 40% of memory roofline

def test_hot_loop_zero_allocation():
    with allocation_guard() as guard:
        run(Config(seed=1, n=50_000, k=200, steps=100_000))
    assert guard.heap_bytes_during_steady_state == 0  # §2.2 zero-alloc contract

def test_memory_footprint_matches_layout():
    cfg = Config(n=1_000_000, k=1000)
    predicted = layout_bytes(cfg)                    # 10·M + 38·N + D_max·N·4 + traces
    assert abs(peak_rss(cfg) - predicted) / predicted < 0.10
```

---

## 8. Milestones & Definition of Done

| Milestone | Gate (all tests green) | Proves |
|-----------|------------------------|--------|
| **M0 — Numeric core** | T1.* | integrator + synapse are exact; κ calibrated |
| **M1 — Routing + plasticity** | T0.*, T2.* | CSR scatter, delay ring, transpose-free STDP correct & deterministic |
| **M2 — Criticality (manual tuning)** | T3.* core (incl. negative controls) | network *measurably* critical at a **tuned** g_c and *measurably not* off it — de-risks the kernel against a known target |
| **M3 — Self-organized criticality** | T3-SOC.* | network **finds** g_c with no tuning (LHG homeostasis), from both sub- and super-critical starts — *the cognitive-science payload* |
| **M4 — Performance** | T4.* | ≥ 40% roofline, zero-alloc loop, predicted footprint |
| **M5 — Visualization** | frontend bus contract (§6) round-trips | zero-alloc WebGL renders the rate field + spikes |

**Definition of Done:** every manifest test green, every negative control red-off-criticality, **and** the SOC attractor reached from both directions (M3). Nothing about frame rate or neuron count substitutes for the exponents, their scaling relation, and demonstrated self-organization. *If it cannot be tested, it does not ship.*

---

## 9. Config contract (single frozen source of truth)

Determinism (§4.3) depends on one immutable config object; all six modules read it, none mutate it.

```toml
[run]
seed = 2025
dt_ms = 0.1
steps = 2_000_000
warmup = 200_000

[network]
n = 1_000_000
k = 1000                 # mean in-degree → M = 1e9
ei_ratio = 0.8           # Dale's law, 80% excitatory
topology = "random_fixed_indegree"
permutation = "metis"    # §3.4 cache-locality reorder

[dynamics]
model = "izhikevich"
g = 1.0                  # control parameter ≡ branching ratio (§4.4)
tau_syn_ms = 5.0
d_max_steps = 20

[plasticity]
enabled = true
rule = "pair_stdp_online"    # transpose-free (§3.5)
a_plus = 0.01
a_minus = 0.0105
tau_stdp_ms = 20.0

[homeostasis]                          # self-organized-criticality driver (M3; OFF for M1/M2)
enabled = false                        # M1/M2 tune g by hand; M3 flips this on and g becomes an initial condition
driver = "depressing_synapses_lhg"     # Levina–Herrmann–Geisel dynamical synapses
resource_pool = "per_presyn_neuron"    # O(N) state (§3.5); fallback "per_synapse" (+4 GB)
tau_homeo_ms = 10000.0                 # SLOW: τ_homeo ≫ τ_syn, τ_mem ⇒ timescale separation (required for SOC)
u_release = 0.2                        # utilization / release fraction per spike
```

---

## 10. Build & CI contract

- **Kernel:** Rust crate `criticalcortex_kernel`, `maturin` build, `rust-numpy` + `pyo3` for borrowed-slice interop, `rayon` for the tiled scatter. `#![forbid(unsafe_op_in_unsafe_fn)]`; the only `unsafe` permitted is the deterministic atomic-free tile reduction, isolated behind a `SAFETY:`-documented module and covered by T0.
- **CI stages:** `M0→M5` gates run in order; a merge is blocked unless the manifest for the touched milestone is green. T3 (the expensive statistical suite) runs nightly at `n=10⁵`; a reduced `n=2×10⁴` smoke version runs per-PR. **T3-SOC is the most expensive of all** — multi-million-step runs at the slow homeostatic timescale — so it runs nightly only (never per-PR), and CI asserts the timescale-separation precondition `τ_homeo ≥ 100·max(τ_syn, τ_mem)` before launching it. Milestone markers (`@pytest.mark.milestone`) are registered in `pyproject.toml`.
- **Roofline gate:** T4 records `ev_s / stream_bw` as a tracked metric; a regression >10% fails CI (catches accidental cache-hostile refactors).

---

## Appendix A — Calibration of the acceptance tolerances

The §7.4 tolerances are not guessed. A pure critical Galton–Watson process (Poisson offspring, mean 1 — the exact generative model the simulator must reproduce at `g_c`) was simulated and passed through the *same estimators* the manifest specifies. `N = 1.2×10⁶` avalanches, high size-cap with censored (cap-truncated) avalanches excluded from fits.

| quantity | mean-field target | recovered | method | manifest bound |
|----------|-------------------|-----------|--------|----------------|
| `τ` (size) | 1.5000 | **1.5026** | CSN MLE, xmin via KS | `[1.45, 1.55]` |
| `α` (duration) | 2.0000 | **1.9050** | CSN MLE | `[1.80, 2.15]` |
| `γ` from ⟨S⟩(T) | 2.0000 | **1.8600** | log–log regression, untruncated window | — |
| `(α−1)/(τ−1)` vs `γ` | 0 gap | **3.2% gap** | scaling-relation consistency | `<10%` |
| cutoff `S_c(N)` | `∼ N^{2/3}` | (FSS) | data collapse | `|D−2/3|<0.10` |

**Reading of the calibration.** The size exponent is recoverable to <0.2%; the duration exponent and γ carry a shared ~5% finite-window/discreteness bias, so pinning either to exactly 2.0 would be statistically naïve. The **scaling relation, however, closes to 3.2%** because the biases in `α` and `γ` are correlated and cancel in the ratio test — which is precisely why the crackling test is written as a self-consistency check, and why it is the strongest single line of evidence for genuine criticality. (Calibration script: `calibrate.py`, seed 7, reproducible.)

## Appendix B — Exponent reference & literature

| exponent | value | source |
|----------|-------|--------|
| avalanche size `τ` | 3/2 | Beggs & Plenz 2003; Borel dist. of critical Poisson-GW total progeny |
| avalanche duration `α` | 2 | Kolmogorov survival `P(T>t)∼2/t` |
| `⟨S⟩(T) ~ T^γ` | γ = 2 | scaling relation `(α−1)/(τ−1)` |
| mean shape | `T^{γ−1}` | Friedman et al. 2012 (universal collapse) |
| branching ratio | σ → 1 | critical branching; MR estimator Wilting & Priesemann 2018 |
| FSS cutoff | `S_c ∼ N^{2/3}` | Erdős–Rényi critical window |

Key references: Beggs & Plenz (2003) *J. Neurosci.*; Clauset, Shalizi & Newman (2009) *SIAM Rev.*; Friedman et al. (2012) *PRL*; Wilting & Priesemann (2018) *Nat. Commun.*; Levina, Herrmann & Geisel (2007) *Nat. Phys.*; Morrison, Diesmann & Gerstner (2008) *Biol. Cybern.*; Izhikevich (2003) *IEEE TNN*; Sethna, Dahmen & Myers (2001) *Nature*.
