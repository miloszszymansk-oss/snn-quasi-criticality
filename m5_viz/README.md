# M5 — Real-time SOqC visualization

Live view of the E/I-balanced Izhikevich network self-organizing to quasi-criticality:
a WebGL neuron cloud lighting up on spikes, plus streaming rate, ⟨x⟩, and the avalanche
size distribution building up in real time.

## Folder structure

```
New project - Claude/
├── criticalcortex/            # existing package (Rust kernel, sim, criticality, avalanche_stats)
└── m5_viz/                     # this milestone
    ├── server.py               # FastAPI + WebSocket backend; paced 30 FPS broadcaster
    ├── sim_engine.py           # backend-agnostic streaming stepper (Rust kernel or NumPy reference)
    ├── protocol.py             # binary render-frame wire format (single source of truth)
    ├── requirements.txt
    ├── README.md
    └── static/
        ├── index.html          # UI shell (Three.js + uPlot from CDN)
        └── app.js              # WS client, neuron cloud, live charts
```

## Run (on the Mac, Rust kernel auto-detected)

```bash
cd "New project - Claude/m5_viz"
python3 -m pip install -r requirements.txt          # into the same venv where criticalcortex is built
python3 server.py                                    # or: uvicorn server:app --port 8000
# open http://127.0.0.1:8000
```

`sim_engine` imports the compiled Rust kernel when it is available (your build) and falls back
to the pure-NumPy reference otherwise — the banner printed on startup says which is live.

## Configuration (env vars — defaults are the validated SOqC operating point)

| var | default | meaning |
|-----|---------|---------|
| `M5_N` | 1000 | neuron count (1000–2000) |
| `M5_G` | 3.5 | control parameter g |
| `M5_MU_EXT` | 3.3 | tonic external drive |
| `M5_EI_RATIO` | 0.8 | excitatory fraction (0.8 = balanced SOqC regime) |
| `M5_TAU_HOMEO` | 50000 | homeostatic time constant (ms) |
| `M5_STEPS_PER_FRAME` | 20 | sim steps advanced per broadcast frame (playback speed; also live via the slider) |
| `M5_FPS` | 30 | broadcast cap |
| `M5_SPATIAL` | 1 | distance-embedded connectome (Fibonacci sphere) → avalanches propagate as spatial waves; `0` = random graph |
| `M5_LOCALITY` | 0.35 | exponential length scale of the distance kernel (chord units ~[0,2]); larger → back toward random |
| `M5_DIST_DELAYS` | 1 | axonal delays scale with distance (finite conduction velocity → visible ripples) |

Example: `M5_N=2000 M5_STEPS_PER_FRAME=40 M5_LOCALITY=0.25 python3 server.py`

## Architecture

- **Decoupled ticks.** One shared `SimEngine` advances `STEPS_PER_FRAME` steps per frame; a single
  paced `broadcaster()` task emits one render frame per client at ≤ `FPS`. Many simulation ticks
  map to one render tick, so the kernel is never throttled by the browser. Stepping runs in a worker
  thread (`asyncio.to_thread`) so the event loop stays responsive.
- **Bounded memory.** The AER sink is reset every batch (`aer_count[0] ← 0`), so an indefinite live
  run reuses a fixed buffer — nothing grows without bound, and the zero-allocation kernel hot loop
  is untouched (this module only acts at batch boundaries).
- **Wire format.** Hot path is a compact little-endian binary frame (see `protocol.py`): header
  (frame id, abs step, rate, ⟨x⟩, N, n_glow) + `uint16` spike ids. Control/analytics (the one-time
  `hello` with the E/I mask + histogram edges, and the ~1 Hz avalanche histogram) go as JSON on the
  same socket. The browser distinguishes by `typeof event.data`.
- **Controls.** The playback slider and pause button send `{cmd:…}` JSON back over the socket.

## Spatial mode (default) — the propagating-wave money shot

`M5_SPATIAL=1` (default) uses `criticalcortex.spatial_connectome`: neurons are embedded on the
Fibonacci sphere and each still draws EXACTLY K inputs with the exact E/I split, but partners are
sampled with probability ∝ exp(−distance/`locality`) (Gumbel-top-k), inhibition is interspersed
(not clumped at a pole), and delays scale with distance. The frontend renders the backend's
positions and a faint mesh of the shortest local edges (toggle in the panel), so cascades spread
as spatially-contiguous ripples instead of global flicker. The Rust kernel needs **no change** —
it only ever reads the CSR arrays and is topology-agnostic.

**SOqC is preserved under embedding** (measured, reference backend, N=1000): σ_MR self-organizes to
≈0.97 for BOTH the spatial and random connectomes — the homeostatic operating point is topology-
robust. Still, re-run `validate_m3_criticality.py` on the spatial variant for the exact exponent /
GOF (locality can shift the avalanche exponent even when σ_MR does not).

## Notes / honest caveats

- Near-critical avalanches are **fractal, not clean expanding rings** — with locality you see
  spatially-contiguous local cascades (a real improvement over global flicker), but not textbook
  circular wavefronts every time. Lower `M5_LOCALITY` (tighter kernel) makes propagation more
  wave-like; larger values fade back toward the random graph.
- Tested here: spatial connectome (fixed in-degree, exact E/I split, interspersed inhibition,
  edge locality, distance–delay correlation 0.997), `sim_engine` stepping in spatial mode, the
  rolling histogram, the binary encode/decode roundtrip, and `app.js`/`server.py` syntax — all
  against the NumPy reference. The full FastAPI server can't run in the build sandbox (no network to
  install FastAPI); first real end-to-end run is on the Mac.
