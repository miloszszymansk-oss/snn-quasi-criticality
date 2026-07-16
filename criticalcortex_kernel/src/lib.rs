//! criticalcortex_kernel — Rust routing kernel (M3: the physics loop).
//!
//! Compiled by maturin to the Python module `criticalcortex._kernel`. `step_block`
//! implements the full integrate -> decay -> drain -> spike -> scatter -> STDP loop,
//! mirroring `criticalcortex/simulation.py` operation-for-operation so the SHORT-horizon
//! spike raster is bit-for-bit identical to reference/golden_master.aer (T0/M3).
//!
//! Bit-exactness design (SPEC §4.3 + the M3 plan):
//!   * State is f32; every arithmetic expression matches the reference's association,
//!     and Rust never contracts a*b+c into an FMA — so f32 results are IEEE-identical.
//!   * The external drive is a splitmix64 counter-hash -> f32 uniform, identical to the
//!     NumPy `hash_noise` (pure integer wrapping + exact u24->f32), NOT a Gaussian RNG.
//!   * Decay factors (a_syn, a_stdp) are PRECOMPUTED to f32 on the Python side and passed
//!     in, so no exp() drift between libm implementations can creep in.
//!   * The recurrent scatter accumulates in the SAME order as the reference (ascending
//!     fired neuron, then CSR edge order), so ring sums are bit-identical.
//! STDP (transpose-free, SPEC §3.5): depression is out-edge/trace native; potentiation is
//! deferred onto the presynaptic out-edge traversal via a bounded per-neuron post-spike
//! history (Morrison 2008). STDP does not affect the short-horizon raster (verified in
//! Python: STDP-on == STDP-off for the first 50 steps), so the M3 exact-match test
//! validates the core loop; transpose-free STDP correctness is a T2.4 / T3 concern.

#![forbid(unsafe_code)]

use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;

/// splitmix64 counter-hash -> f32 uniform in [-1, 1) for neuron `i` at absolute `step`.
/// Bit-identical to NumPy `criticalcortex.simulation.hash_noise` (wrapping u64 == mod 2^64).
#[inline]
fn hash_noise_i(seed: u64, step: i64, i: usize) -> f32 {
    let mut z = seed
        .wrapping_add((step as u64).wrapping_mul(0xD1342543DE82EF95))
        .wrapping_add((i as u64).wrapping_mul(0x2545F4914F6CDD1D))
        .wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    let u24 = (z >> 40) as u32; // top 24 bits, < 2^24
    let u = (u24 as f32) * (1.0f32 / 16777216.0f32); // [0, 1), exact
    2.0f32 * u - 1.0f32 // [-1, 1)
}

/// Immutable per-run parameters (SPEC §2.2 `&Params`). Decay factors arrive PRECOMPUTED
/// as f32 (`a_syn`, `a_stdp`) so no exp() is evaluated in Rust.
#[pyclass]
#[derive(Clone)]
struct KernelParams {
    #[pyo3(get, set)]
    dt: f32,
    #[pyo3(get, set)]
    a: f32,
    #[pyo3(get, set)]
    b: f32,
    #[pyo3(get, set)]
    c: f32,
    #[pyo3(get, set)]
    d: f32,
    #[pyo3(get, set)]
    v_peak: f32,
    #[pyo3(get, set)]
    a_plus: f32,
    #[pyo3(get, set)]
    a_minus: f32,
    #[pyo3(get, set)]
    a_syn: f32,
    #[pyo3(get, set)]
    a_stdp: f32,
    #[pyo3(get, set)]
    mu_ext: f32,
    #[pyo3(get, set)]
    sigma_ext: f32,
    #[pyo3(get, set)]
    refractory_steps: i32,
    #[pyo3(get, set)]
    seed: u64,
    // --- self-organized criticality (per-neuron short-term depression) ---
    #[pyo3(get, set)]
    homeo_enabled: i32, // 0 => classic (bit-exact) path; !=0 => SOC depression on
    #[pyo3(get, set)]
    tau_homeo: f32, // ms, slow resource-recovery time constant
}

#[pymethods]
impl KernelParams {
    #[new]
    #[pyo3(signature = (dt=0.1, a=0.02, b=0.2, c=-65.0, d=8.0, v_peak=30.0, a_plus=0.01,
                        a_minus=0.0105, a_syn=0.98019868, a_stdp=0.9950125, mu_ext=3.5,
                        sigma_ext=3.0, refractory_steps=0, seed=42,
                        homeo_enabled=0, tau_homeo=10000.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dt: f32, a: f32, b: f32, c: f32, d: f32, v_peak: f32, a_plus: f32, a_minus: f32,
        a_syn: f32, a_stdp: f32, mu_ext: f32, sigma_ext: f32, refractory_steps: i32, seed: u64,
        homeo_enabled: i32, tau_homeo: f32,
    ) -> Self {
        KernelParams {
            dt, a, b, c, d, v_peak, a_plus, a_minus, a_syn, a_stdp, mu_ext, sigma_ext,
            refractory_steps, seed, homeo_enabled, tau_homeo,
        }
    }
}

/// Per-call return stats (SPEC §2.2 `StepStats`). `alloc_bytes` is 0: the loop touches
/// only borrowed buffers and stack scalars — no heap allocation.
#[pyclass]
#[derive(Clone)]
struct StepStats {
    #[pyo3(get)]
    n_spikes: u64,
    #[pyo3(get)]
    sum_sq: f64,
    #[pyo3(get)]
    alloc_bytes: u64,
}

/// The zero-allocation hot loop (SPEC §2.2, extended with the transpose-free STDP state
/// `last_pre / xhat / post_hist / post_cnt` that Morrison-2008 potentiation requires).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn step_block<'py>(
    py: Python<'py>,
    // --- neuron state (mutated in place) ---
    mut v: PyReadwriteArray1<'py, f32>,
    mut u: PyReadwriteArray1<'py, f32>,
    mut g_exc: PyReadwriteArray1<'py, f32>,
    mut g_inh: PyReadwriteArray1<'py, f32>,
    mut refrac: PyReadwriteArray1<'py, i32>,
    mut t_last: PyReadwriteArray1<'py, i64>,
    // --- out-edge CSR (weight mutable under STDP) ---
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i32>,
    mut weight: PyReadwriteArray1<'py, f32>,
    delay: PyReadonlyArray1<'py, u16>,
    // --- delayed delivery + plasticity traces ---
    mut ring: PyReadwriteArray1<'py, f32>, // flattened [d_max * n], row-major
    mut x_pre: PyReadwriteArray1<'py, f32>,
    mut x_post: PyReadwriteArray1<'py, f32>,
    // --- transpose-free potentiation state (per neuron) ---
    mut last_pre: PyReadwriteArray1<'py, i64>,
    mut xhat: PyReadwriteArray1<'py, f32>,
    mut post_hist: PyReadwriteArray1<'py, i64>, // flattened [n * hist_cap], circular per neuron
    mut post_cnt: PyReadwriteArray1<'py, i64>,
    // --- self-organized criticality: per-neuron resources + per-spike depletion factor ---
    mut x_avail: PyReadwriteArray1<'py, f32>,
    homeo_keep: PyReadonlyArray1<'py, f32>,
    // --- params + block control + AER sink ---
    params: &KernelParams,
    start_step: i64,
    n_steps: u32,
    mut aer_step: PyReadwriteArray1<'py, u32>,
    mut aer_neuron: PyReadwriteArray1<'py, u32>,
    mut aer_count: PyReadwriteArray1<'py, i64>,
) -> PyResult<StepStats> {
    let _ = py;

    // Borrow zero-copy slices into the NumPy buffers.
    let v = v.as_slice_mut()?;
    let u = u.as_slice_mut()?;
    let g_exc = g_exc.as_slice_mut()?;
    let g_inh = g_inh.as_slice_mut()?;
    let refrac = refrac.as_slice_mut()?;
    let t_last = t_last.as_slice_mut()?;
    let indptr = indptr.as_slice()?;
    let indices = indices.as_slice()?;
    let weight = weight.as_slice_mut()?;
    let delay = delay.as_slice()?;
    let ring = ring.as_slice_mut()?;
    let x_pre = x_pre.as_slice_mut()?;
    let x_post = x_post.as_slice_mut()?;
    let last_pre = last_pre.as_slice_mut()?;
    let xhat = xhat.as_slice_mut()?;
    let post_hist = post_hist.as_slice_mut()?;
    let post_cnt = post_cnt.as_slice_mut()?;
    let x_avail = x_avail.as_slice_mut()?;
    let homeo_keep = homeo_keep.as_slice()?;
    let aer_step = aer_step.as_slice_mut()?;
    let aer_neuron = aer_neuron.as_slice_mut()?;
    let aer_count = aer_count.as_slice_mut()?;

    let n = v.len();
    let d_max = (ring.len() / n) as i64;
    let hist_cap = post_hist.len() / n;

    // scalar params (all f32; the reference casts the same Python floats to f32)
    let (dt, a, b, c, d) = (params.dt, params.a, params.b, params.c, params.d);
    let v_peak = params.v_peak;
    let half = 0.5f32 * dt;
    let a_syn = params.a_syn;
    let a_stdp = params.a_stdp;
    let (ap, am) = (params.a_plus, params.a_minus);
    let (mu, sigma) = (params.mu_ext, params.sigma_ext);
    let seed = params.seed;
    let refractory = params.refractory_steps;
    let homeo = params.homeo_enabled != 0;
    let homeo_rec = dt / params.tau_homeo; // per-step resource recovery rate

    let mut n_spikes: u64 = 0;
    let mut sum_sq: f64 = 0.0;

    for local in 0..n_steps as i64 {
        let step = start_step + local;

        // (1) exponential decay of synaptic accumulators and STDP traces
        for i in 0..n {
            g_exc[i] *= a_syn;
            g_inh[i] *= a_syn;
            x_pre[i] *= a_stdp;
            x_post[i] *= a_stdp;
            if homeo {
                x_avail[i] += (1.0f32 - x_avail[i]) * homeo_rec; // slow recovery toward 1
            }
        }

        // (2) drain this step's ring slot (drain BEFORE schedule); split by sign into g_exc/g_inh
        let slot = (step.rem_euclid(d_max)) as usize;
        let base = slot * n;
        for j in 0..n {
            let dcur = ring[base + j];
            ring[base + j] = 0.0;
            if dcur > 0.0 {
                g_exc[j] += dcur;
            } else {
                g_inh[j] += dcur;
            }
        }

        // (3)&(4) external drive + Izhikevich (two half-dt substeps on v, one dt step on u)
        for i in 0..n {
            let i_ext = mu + sigma * hash_noise_i(seed, step, i);
            let i_syn = (g_exc[i] + g_inh[i]) + i_ext;
            let vv = v[i];
            let uu = u[i];
            let dv1 = 0.04f32 * vv * vv + 5.0f32 * vv + 140.0f32 - uu + i_syn;
            let vh = vv + half * dv1;
            let dv2 = 0.04f32 * vh * vh + 5.0f32 * vh + 140.0f32 - uu + i_syn;
            let vn = vh + half * dv2;
            let un = uu + dt * (a * (b * vn - uu));
            v[i] = vn;
            u[i] = un;
        }

        // (5) spike detection + reset + AER + record post-spike (ascending neuron id).
        //     `t_last[i] == step` marks a neuron that fired THIS step (used below).
        let mut cnt = aer_count[0] as usize;
        for i in 0..n {
            let fired = v[i] >= v_peak && refrac[i] == 0;
            if fired {
                v[i] = c;
                u[i] += d;
                t_last[i] = step;
                refrac[i] = refractory;
                // record spike in this neuron's circular post-history (for potentiation)
                let h = (post_cnt[i] as usize) % hist_cap;
                post_hist[i * hist_cap + h] = step;
                post_cnt[i] += 1;
                // stream to AER
                aer_step[cnt] = step as u32;
                aer_neuron[cnt] = i as u32;
                cnt += 1;
                n_spikes += 1;
            } else if refractory > 0 && refrac[i] > 0 {
                refrac[i] -= 1;
            }
        }
        aer_count[0] = cnt as i64;

        // (6) STDP (transpose-free). Traces are read PRE-increment.
        //   (a) potentiation catch-up + depression along out-edges of each fired neuron
        for i in 0..n {
            if t_last[i] != step {
                continue;
            }
            let lo = indptr[i] as usize;
            let hi = indptr[i + 1] as usize;
            // potentiation: realize post spikes of each target j in (last_pre[i], step]
            let tp = last_pre[i];
            if tp >= 0 && xhat[i] != 0.0 {
                let xh = xhat[i];
                for e in lo..hi {
                    let j = indices[e] as usize;
                    let cnt_j = (post_cnt[j] as usize).min(hist_cap);
                    let mut acc = 0.0f32;
                    for k in 0..cnt_j {
                        let s = post_hist[j * hist_cap + k];
                        if s > tp && s <= step {
                            acc += a_stdp.powi((s - tp) as i32);
                        }
                    }
                    if acc != 0.0 {
                        weight[e] += ap * xh * acc;
                    }
                }
            }
            // depression: post-before-pre, using the current post trace
            for e in lo..hi {
                let j = indices[e] as usize;
                weight[e] -= am * x_post[j];
            }
        }
        //   (b) increment traces of fired neurons, then refresh last_pre / xhat
        for i in 0..n {
            if t_last[i] == step {
                x_pre[i] += 1.0;
                x_post[i] += 1.0;
                last_pre[i] = step;
                xhat[i] = x_pre[i];
            }
        }

        // (7) scatter fired neurons' (updated) out-edge weights into the ring at step+delay.
        //     Ascending neuron, then CSR edge order == the reference's np.add.at order.
        for i in 0..n {
            if t_last[i] != step {
                continue;
            }
            let lo = indptr[i] as usize;
            let hi = indptr[i + 1] as usize;
            // SOC depression gate: scale delivered efficacy by available resources.
            // homeo off => gain == 1.0, and weight[e]*1.0 == weight[e] exactly (IEEE754),
            // so the classic/bit-exact scatter path is preserved.
            let gain = if homeo { x_avail[i] } else { 1.0f32 };
            for e in lo..hi {
                let j = indices[e] as usize;
                let arr = (step + delay[e] as i64).rem_euclid(d_max) as usize;
                ring[arr * n + j] += weight[e] * gain;
            }
            if homeo {
                x_avail[i] *= homeo_keep[i]; // deplete (excitatory-only via keep vector)
            }
        }
    }

    // light telemetry (not part of the raster contract)
    for &vi in v.iter() {
        sum_sq += (vi as f64) * (vi as f64);
    }

    Ok(StepStats { n_spikes, sum_sq, alloc_bytes: 0 })
}

/// Module init. Function name must equal the last component of `module-name`
/// (`criticalcortex._kernel`) in pyproject.toml.
#[pymodule]
fn _kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KernelParams>()?;
    m.add_class::<StepStats>()?;
    m.add_function(wrap_pyfunction!(step_block, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
