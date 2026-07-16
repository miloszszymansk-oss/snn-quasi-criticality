"""
criticalcortex.simulation — composed pure-NumPy reference network (M0 + M1).

`step_block_reference(...)` mirrors the SPEC §2.2 `step_block` FFI signature and its
in-place state-mutation contract: Python allocates every array once, the function
borrows and mutates them, and advances `n_steps` timesteps, streaming spikes into a
preallocated AER buffer. This is the golden-master generator the Rust kernel is diffed
against (T0). Per absolute step:

    1. decay synaptic accumulators (g *= exp(-dt/tau_syn)) and STDP traces.
    2. DRAIN the delay-ring slot for this step -> g_exc/g_inh (drain-before-schedule).
    3. form I_syn = g_exc + g_inh + I_ext(seed, step)   [deterministic Philox noise].
    4. integrate Izhikevich: v via TWO half-dt Euler substeps, u via ONE dt step (M0).
    5. detect v >= V_PEAK, emit spikes, reset (v<-c, u<-u+d), set refractory / t_last.
    6. STDP (traces read pre-increment): depression along out-edges (transpose-free);
       potentiation via a precomputed in-edge index. The in-edge (eager) realization is
       byte-for-byte equivalent to the transpose-free out-edge scheme (locked by
       test_m1_reference::T2.4, <=5e-15); the Rust kernel uses the transpose-free form
       for memory. STDP pairing uses somatic spike steps (delay affects delivery only).
    7. SCATTER the fired neurons' (updated) weights into the ring at step+delay.
    8. append fired (step, neuron_id) to the AER buffer.

Determinism: external drive is counter-based Philox keyed on (seed, absolute step), so
the noise for step t is a pure function of (seed, t) — independent of block boundaries
and of any future thread partition (SPEC §4.3). State is float32 to mirror §3.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .connectome import Connectome

F32 = np.float32

# splitmix64 counter-hash constants — the deterministic external drive is a pure
# function of (seed, step, neuron) implemented with IDENTICAL integer + f32 ops in
# NumPy and Rust, so the two languages produce bit-identical stimulus (numpy's
# Philox+ziggurat Gaussian is NOT practically reproducible in Rust; this is).
_P1 = np.uint64(0xD1342543DE82EF95)
_P2 = np.uint64(0x2545F4914F6CDD1D)
_S = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)

# sentinel "step" for the one-shot initial-voltage spread (distinct from any run step)
INIT_STEP = 1 << 32


@dataclass
class SimParams:
    dt: float = 0.1                 # ms
    # Izhikevich RS (homogeneous; SPEC allows heterogeneous a,b,c,d — reference uses RS)
    a: float = 0.02
    b: float = 0.20
    c: float = -65.0
    d: float = 8.0
    v_peak: float = 30.0
    tau_syn: float = 5.0            # ms
    # pair-STDP (SPEC §9)
    a_plus: float = 0.01
    a_minus: float = 0.0105
    tau_stdp: float = 20.0         # ms
    # deterministic external drive  I_ext = mu + sigma * N(0,1)
    mu_ext: float = 3.5            # I_ext = mu + sigma*U[-1,1); tuned to ~9 Hz mean rate
    sigma_ext: float = 3.0
    v0_spread: float = 45.0        # initial v spread over [c, c+v0_spread) -> early async firing
    refractory_steps: int = 0      # 0 => pure Izhikevich; >0 => absolute refractory
    seed: int = 42
    # --- self-organized criticality: per-neuron short-term depression (LHG variant) ---
    homeo_enabled: bool = False    # off => bit-exact M3/T0 path unchanged
    tau_homeo: float = 10000.0     # ms, SLOW resource recovery (timescale separation)
    u_release: float = 0.2         # fraction of available resources consumed per spike
    # --- separation of timescales: drive-in-silence (BTW/LHG strict-SOC protocol) ---
    # When enabled: the continuous external drive (mu_ext, sigma_ext) is SUPPRESSED, and a
    # single deterministic neuron is forced to spike ONLY when the network has been globally
    # quiescent for `quiet_gap` consecutive steps (all in-flight ring activity has drained).
    # This yields non-overlapping avalanches by construction. Default off => existing path
    # (continuous Poisson drive) is untouched and remains bit-exact for T0/M3.
    drive_when_silent: bool = False
    quiet_gap: int = 0             # consecutive silent steps before a drive; 0 => use d_max (ring size)
    # POISED variant: a QUASI-STATIC sub-threshold bias holds neurons near (but below)
    # threshold so a single EPSP can recruit the small near-threshold fraction (σ≈1), while
    # the network still relaxes to silence between avalanches (frozen ⇒ no spontaneous firing).
    #   bias[i] = drive_tonic + bias_spread · ξ_i ,   ξ_i ∈ [-1,1) frozen per-neuron hash.
    drive_tonic: float = 0.0       # DC poising level (sub-rheobase); only used in drive_when_silent
    bias_spread: float = 0.0       # frozen per-neuron heterogeneity in the poising bias


def hash_noise(seed: int, step: int, n: int) -> np.ndarray:
    """Deterministic external-drive noise in [-1, 1), one f32 per neuron.

    noise[i] = 2*u24(seed, step, i)/2^24 - 1, where u24 are the top 24 bits of a
    splitmix64 hash of (seed, step, i). Pure integer mixing (wrapping mod 2^64, which
    matches Rust `wrapping_add/mul`) followed by an exact u24->f32 conversion — so the
    Rust kernel reproduces this bit-for-bit, which is what makes short-horizon exact
    raster matching possible (unlike numpy's Philox+ziggurat Gaussian). A pure function
    of (seed, step, i): block- and thread-partition invariant.
    """
    with np.errstate(over="ignore"):                    # uint64 overflow == mod-2^64 wrap
        i = np.arange(n, dtype=np.uint64)
        z = np.uint64(seed) + np.uint64(step) * _P1 + i * _P2 + _S
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        z = z ^ (z >> np.uint64(31))
        u24 = (z >> np.uint64(40)).astype(np.uint32)     # top 24 bits
        u = u24.astype(F32) * F32(1.0 / 16777216.0)      # [0, 1), exact (u24 < 2^24)
        return F32(2.0) * u - F32(1.0)                    # [-1, 1)


def pick_drive_neuron(seed: int, step: int, n: int) -> int:
    """Deterministic single-neuron perturbation target for the drive-in-silence protocol.

    A scalar splitmix64 hash of (seed, step) reduced mod N — the SAME integer mixing as
    `hash_noise`, so the Rust kernel reproduces the identical drive sequence bit-for-bit
    (which is what keeps the separation-of-timescales run deterministic and diff-able).
    """
    with np.errstate(over="ignore"):
        z = np.uint64(seed) + np.uint64(step) * _P1 + _S
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        z = z ^ (z >> np.uint64(31))
        return int(z % np.uint64(n))


def build_in_edges(conn: Connectome):
    """Reference scaffolding: group edge ids by postsynaptic target (the 'transpose').

    Returns (in_indptr[N+1], in_edge_ids[M], in_src[M]) such that the in-edges of
    neuron j are in_edge_ids[in_indptr[j]:in_indptr[j+1]] with presynaptic sources
    in_src[...]. Potentiation modifies the SAME weight[e] the out-edge scatter reads.
    """
    n = conn.n
    src = conn.edge_sources()
    dst = conn.indices.astype(np.int64)
    order = np.argsort(dst, kind="stable")
    in_indptr = np.zeros(n + 1, dtype=np.int64)
    in_indptr[1:] = np.cumsum(np.bincount(dst, minlength=n))
    return in_indptr, order.astype(np.int64), src[order].astype(np.int64)


@dataclass
class Network:
    """All preallocated state for a run (fields map 1:1 to the §2.2 FFI arrays)."""
    conn: Connectome
    params: SimParams
    # mutable plastic weights (a COPY of conn.weight; the connectome stays immutable)
    weight: np.ndarray
    # neuron state (SoA, float32)
    v: np.ndarray
    u: np.ndarray
    g_exc: np.ndarray
    g_inh: np.ndarray
    refrac: np.ndarray
    t_last: np.ndarray
    # delayed delivery + plasticity traces
    ring: np.ndarray
    x_pre: np.ndarray
    x_post: np.ndarray
    x_avail: np.ndarray            # per-neuron available synaptic resources in (0,1] (SOC)
    homeo_keep: np.ndarray         # per-spike depletion factor: (1-U) excitatory, 1.0 inhibitory
    # in-edge scaffolding (reference-only; Rust is transpose-free)
    in_indptr: np.ndarray
    in_edge_ids: np.ndarray
    in_src: np.ndarray
    # AER sink
    aer_step: np.ndarray
    aer_neuron: np.ndarray
    aer_count: np.ndarray          # 1-element int64 (mutable running total)
    # drive-in-silence state: soc_state[0] = consecutive silent steps (carried across blocks)
    soc_state: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    next_step: int = 0             # absolute step of the next block


def build_network(conn: Connectome, params: SimParams, aer_capacity: int = 2_000_000) -> Network:
    """Allocate every array ONCE (mirrors the Python-side allocation before the FFI loop)."""
    n = conn.n
    d_max = int(conn.meta.get("d_max", conn.delay.max()))
    # deterministic initial-voltage spread (hash-seeded, Rust-reproducible): heterogeneous
    # v0 gives immediate ASYNCHRONOUS firing instead of a degenerate synchronized onset.
    if params.drive_when_silent:
        # start from rest so the network is silent and the first perturbation seeds avalanche 1
        v = np.full(n, F32(params.c), dtype=F32)
    else:
        u01 = (hash_noise(params.seed, INIT_STEP, n) + F32(1.0)) * F32(0.5)   # [0, 1)
        v = (F32(params.c) + F32(params.v0_spread) * u01).astype(F32)
    in_indptr, in_edge_ids, in_src = build_in_edges(conn)
    d_max0 = int(conn.meta.get("d_max", conn.delay.max()))
    # seed soc_state at the gap so step 0 drives immediately (network starts quiescent)
    quiet_gap0 = params.quiet_gap if params.quiet_gap > 0 else d_max0
    return Network(
        conn=conn, params=params,
        weight=conn.weight.copy(),                   # mutable STDP copy; conn stays pristine
        v=v,
        u=(F32(params.b) * v).astype(F32),
        g_exc=np.zeros(n, dtype=F32),
        g_inh=np.zeros(n, dtype=F32),
        refrac=np.zeros(n, dtype=np.int32),
        t_last=np.full(n, -1, dtype=np.int64),
        ring=np.zeros((d_max, n), dtype=F32),
        x_pre=np.zeros(n, dtype=F32),
        x_post=np.zeros(n, dtype=F32),
        x_avail=np.ones(n, dtype=F32),          # resources start full
        # excitatory-only depression: inhibitory efficacy is never depressed (keep=1)
        homeo_keep=np.where(conn.is_inhib, F32(1.0), F32(1.0 - params.u_release)).astype(F32),
        in_indptr=in_indptr, in_edge_ids=in_edge_ids, in_src=in_src,
        aer_step=np.empty(aer_capacity, dtype=np.uint32),
        aer_neuron=np.empty(aer_capacity, dtype=np.uint32),
        aer_count=np.zeros(1, dtype=np.int64),
        soc_state=np.array([quiet_gap0], dtype=np.int64),   # >= gap => drive at step 0
    )


def step_block_reference(
    # --- neuron state (mutated in place) ---
    v, u, g_exc, g_inh, refrac, t_last,
    # --- out-edge CSR (weight mutable under STDP) ---
    indptr, indices, weight, delay,
    # --- in-edge index (reference scaffolding for eager==transpose-free potentiation) ---
    in_indptr, in_edge_ids, in_src,
    # --- delayed delivery + plasticity traces ---
    ring, x_pre, x_post,
    # --- params + block control + AER sink ---
    params: SimParams, start_step: int, n_steps: int,
    aer_step, aer_neuron, aer_count,
    homeo_x=None, homeo_keep=None,   # per-neuron resources + depletion factor (SOC)
    soc_state=None,                  # [silent_run] counter for the drive-in-silence protocol
) -> int:
    """Advance `n_steps` steps from absolute `start_step`; returns spikes emitted."""
    dt = params.dt
    half = 0.5 * dt
    a, b, c, d = params.a, params.b, params.c, params.d
    v_peak = params.v_peak
    # decay factors rounded to f32 explicitly (the Rust kernel receives these exact
    # f32 constants, so the decay multiply is bit-identical and free of exp() drift).
    a_syn = F32(np.exp(-dt / params.tau_syn))
    a_stdp = F32(np.exp(-dt / params.tau_stdp))
    ap, am = params.a_plus, params.a_minus
    mu, sigma, seed = params.mu_ext, params.sigma_ext, params.seed
    refractory = params.refractory_steps
    homeo = bool(params.homeo_enabled) and (homeo_x is not None)
    homeo_rec = F32(dt / params.tau_homeo)      # per-step resource recovery rate
    if homeo and homeo_keep is None:            # fallback: deplete every neuron equally
        homeo_keep = np.full(v.shape[0], F32(1.0 - params.u_release), dtype=F32)
    d_max = ring.shape[0]
    # --- separation-of-timescales (drive-in-silence) setup ---
    dws = bool(params.drive_when_silent)
    quiet_gap = params.quiet_gap if params.quiet_gap > 0 else d_max
    if dws and soc_state is None:               # persistent silent-run counter (start quiescent)
        soc_state = np.array([quiet_gap], dtype=np.int64)
    n_neurons = v.shape[0]
    if dws:                                     # frozen quasi-static poising bias (computed once)
        if params.bias_spread != 0.0:
            xi = hash_noise(seed, INIT_STEP + 1, n_neurons)          # frozen per-neuron ξ∈[-1,1)
            poise_bias = (F32(params.drive_tonic) + F32(params.bias_spread) * xi).astype(F32)
        else:
            poise_bias = F32(params.drive_tonic)

    spikes_emitted = 0
    for local in range(n_steps):
        step = start_step + local
        # decide the drive-in-silence perturbation for THIS step (before integration)
        drive_now = dws and (int(soc_state[0]) >= quiet_gap)
        j_drive = pick_drive_neuron(seed, step, n_neurons) if drive_now else -1

        # (1) exponential decay of synaptic accumulators and STDP traces
        g_exc *= a_syn
        g_inh *= a_syn
        x_pre *= a_stdp
        x_post *= a_stdp
        if homeo:                                       # slow resource recovery toward 1
            homeo_x += (F32(1.0) - homeo_x) * homeo_rec

        # (2) drain the delay-ring slot for this step (drain BEFORE schedule)
        slot = step % d_max
        drained = ring[slot].copy()
        ring[slot] = 0.0
        g_exc += np.maximum(drained, F32(0.0))
        g_inh += np.minimum(drained, F32(0.0))

        # (3) synaptic + external current. In drive-in-silence mode the continuous Poisson
        #     drive is SUPPRESSED — the only exogenous input is the sparse when-silent kick.
        if dws:
            i_syn = g_exc + g_inh + poise_bias      # quasi-static poising, no per-step noise
        else:
            i_ext = F32(mu) + F32(sigma) * hash_noise(seed, step, v.shape[0])
            i_syn = g_exc + g_inh + i_ext

        # (4) Izhikevich: two half-dt substeps on v (u frozen), one dt step on u
        dv1 = F32(0.04) * v * v + F32(5.0) * v + F32(140.0) - u + i_syn
        v_half = v + F32(half) * dv1
        dv2 = F32(0.04) * v_half * v_half + F32(5.0) * v_half + F32(140.0) - u + i_syn
        v[:] = v_half + F32(half) * dv2
        u[:] = u + F32(dt) * (F32(a) * (F32(b) * v - u))

        # (4b) drive-in-silence perturbation: force the chosen neuron over threshold so it is
        #      detected as fired this step (one "grain" dropped into a quiescent network).
        if drive_now:
            v[j_drive] = F32(v_peak)
            refrac[j_drive] = 0

        # (5) spike detection (respect refractory) + reset
        fired = (v >= v_peak) & (refrac == 0)
        F = np.nonzero(fired)[0]
        if refractory:
            dec = (refrac > 0) & ~fired
            refrac[dec] -= 1
        if dws:                                     # advance the quiescence counter
            soc_state[0] = 0 if F.size else int(soc_state[0]) + 1
        if F.size == 0:
            continue
        v[F] = F32(c)
        u[F] += F32(d)
        t_last[F] = step
        refrac[F] = refractory

        # (6) STDP — traces read BEFORE the same-step increment (coincident pair -> 0)
        #   potentiation (eager in-edges == transpose-free out-edge scheme, T2.4)
        segs = [in_edge_ids[in_indptr[j]:in_indptr[j + 1]] for j in F]
        srcs = [in_src[in_indptr[j]:in_indptr[j + 1]] for j in F]
        e_pot = np.concatenate(segs)
        s_pot = np.concatenate(srcs)
        weight[e_pot] += F32(ap) * x_pre[s_pot]
        #   depression (out-edges: contiguous edge range per presynaptic neuron)
        for i in F:
            lo, hi = indptr[i], indptr[i + 1]
            weight[lo:hi] -= F32(am) * x_post[indices[lo:hi]]
        #   increment the traces of the neurons that just fired
        x_pre[F] += F32(1.0)
        x_post[F] += F32(1.0)

        # (7) scatter the fired neurons' (now-updated) out-edge weights into the ring,
        #     gated by available synaptic resources (SOC depression) when enabled.
        e_all = np.concatenate([np.arange(indptr[i], indptr[i + 1]) for i in F])
        tgt = indices[e_all]
        slots = (step + delay[e_all].astype(np.int64)) % d_max
        if homeo:
            degs = indptr[F + 1] - indptr[F]
            src_eall = np.repeat(F, degs)               # presynaptic neuron of each edge
            np.add.at(ring, (slots, tgt), weight[e_all] * homeo_x[src_eall])
            homeo_x[F] *= homeo_keep[F]                  # deplete (excitatory-only via keep vec)
        else:
            np.add.at(ring, (slots, tgt), weight[e_all])

        # (8) stream spikes to the preallocated AER buffer (ascending neuron id within step)
        nf = F.size
        cnt = int(aer_count[0])
        aer_step[cnt:cnt + nf] = step
        aer_neuron[cnt:cnt + nf] = F
        aer_count[0] = cnt + nf
        spikes_emitted += nf

    return spikes_emitted


def run_reference(conn: Connectome, params: SimParams, steps: int,
                  block: int | None = None, aer_capacity: int = 2_000_000) -> Network:
    """Convenience driver: allocate a Network and run `steps` (optionally in blocks of
    `block` to exercise the streaming/state-carry contract). Returns the Network with
    its AER sink populated (net.aer_step/net.aer_neuron[:net.aer_count[0]])."""
    net = build_network(conn, params, aer_capacity)
    block = steps if block is None else block
    done = 0
    while done < steps:
        nb = min(block, steps - done)
        step_block_reference(
            net.v, net.u, net.g_exc, net.g_inh, net.refrac, net.t_last,
            net.conn.indptr, net.conn.indices, net.weight, net.conn.delay,
            net.in_indptr, net.in_edge_ids, net.in_src,
            net.ring, net.x_pre, net.x_post,
            net.params, done, nb,
            net.aer_step, net.aer_neuron, net.aer_count,
            net.x_avail, net.homeo_keep, net.soc_state,
        )
        done += nb
    net.next_step = steps
    return net


# ---------------------------------------------------------------------------
# AER serialization (SPEC §5): 64-byte little-endian header + 8-byte records.
# ---------------------------------------------------------------------------
AER_MAGIC = b"CCXAER01"
_HEADER_FMT = "<8sIQfQfQI"     # magic|version|N|dt_ms|seed|g|n_events|flags  = 48 bytes


def aer_to_bytes(aer_step, aer_neuron, count, n, dt_ms, seed, g) -> bytes:
    """Serialize the AER sink to the exact SPEC §5 byte layout (records sorted by
    step, then neuron id — which is the emission order). Little-endian, fixed width."""
    import struct
    count = int(count)
    header = struct.pack(_HEADER_FMT, AER_MAGIC, 1, int(n), float(dt_ms),
                         int(seed), float(g), count, 0)
    header += b"\x00" * (64 - len(header))          # pad to 64 bytes
    rec = np.empty(count, dtype=[("step", "<u4"), ("neuron", "<u4")])
    rec["step"] = aer_step[:count]
    rec["neuron"] = aer_neuron[:count]
    return header + rec.tobytes()
