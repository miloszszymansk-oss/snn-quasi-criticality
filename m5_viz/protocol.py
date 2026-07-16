"""
m5_viz.protocol — the binary wire format for 30 FPS render frames (single source of truth;
the frontend in static/app.js mirrors this exact layout).

Two message kinds share one WebSocket:
  * TEXT (JSON)   — low-frequency control/analytics: the "hello" metadata on connect (N, E/I mask,
                    sphere positions, local-edge mesh, histogram edges) and the ~1 Hz avalanche
                    histogram. Parse cost is irrelevant at that cadence.
  * BINARY (this) — the hot path: one compact little-endian frame at up to 60 FPS.

Frame layout (little-endian), type byte = 1:
    offset       type      field
    0            u8        msg_type (1 = FRAME)
    1            u32       frame_id
    5            u32       abs_step
    9            f32       rate_hz
    13           f32       xbar               (mean ⟨x⟩ — redundant scalar for the plots)
    17           u16       N
    19           u32       n_glow
    23           u8[N]     x_u8               (per-neuron ⟨x⟩ scaled 0..255 — the heatmap)
    23+N         u16[n]    neuron_ids         (which neurons spiked this batch -> glow)
Header = 23 bytes; body = N (resource heatmap) + 2·n_glow (spike ids) bytes.

The per-neuron resource is uint8 (⟨x⟩∈(0,1] → 0..255): ~1 byte/neuron/frame ≈ 60 KB/s at N=2000,
30 FPS — negligible, and it lets the client switch spike/heatmap modes with zero renegotiation.
"""

from __future__ import annotations

import struct

import numpy as np

MSG_FRAME = 1
_HEADER = struct.Struct("<B I I f f H I")   # 23 bytes
HEADER_SIZE = _HEADER.size


def encode_frame(frame_id: int, batch: dict, N: int) -> bytes:
    """Encode a SimEngine.step() result into a binary FRAME.
    `batch['x_u8']` is a length-N uint8 array; `batch['neurons']` is uint16 spike ids."""
    xb = np.ascontiguousarray(batch["x_u8"], dtype=np.uint8)
    ids = np.ascontiguousarray(batch["neurons"], dtype="<u2")
    header = _HEADER.pack(MSG_FRAME, frame_id & 0xFFFFFFFF, batch["abs_step"] & 0xFFFFFFFF,
                          float(batch["rate"]), float(batch["xbar"]), int(N), int(ids.size))
    return header + xb.tobytes() + ids.tobytes()


def decode_frame(buf: bytes) -> dict:
    """Decode a binary FRAME (tests / any Python client; the browser mirrors this in app.js)."""
    msg_type, frame_id, abs_step, rate, xbar, N, n_glow = _HEADER.unpack_from(buf, 0)
    x = np.frombuffer(buf, dtype=np.uint8, count=N, offset=HEADER_SIZE)
    ids = np.frombuffer(buf, dtype="<u2", count=n_glow, offset=HEADER_SIZE + N)
    return dict(msg_type=msg_type, frame_id=frame_id, abs_step=abs_step,
                rate=rate, xbar=xbar, N=N, x=x, ids=ids)
