"""
m5_viz.server — FastAPI + WebSocket backend for the live SOqC visualization.

One shared SimEngine advances the (Rust or reference) hot loop. A single paced broadcaster
task steps the sim `STEPS_PER_FRAME` steps per frame and pushes a binary render frame to every
connected client at a capped `FPS` — the simulation tick and the render tick are DECOUPLED
(many sim steps per broadcast). The ~1 Hz avalanche histogram and the one-time "hello" go as
JSON on the same socket. Stepping runs in a worker thread (asyncio.to_thread) so the event
loop stays responsive even for the pure-Python reference backend.

Run:
    pip install fastapi "uvicorn[standard]" websockets numpy
    python server.py                 # or: uvicorn server:app --host 127.0.0.1 --port 8000
    open http://127.0.0.1:8000

Config via environment variables (defaults = the validated SOqC operating point):
    M5_N, M5_G, M5_MU_EXT, M5_EI_RATIO, M5_TAU_HOMEO, M5_STEPS_PER_FRAME, M5_FPS
"""

from __future__ import annotations

import asyncio
import contextlib
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sim_engine import SimEngine
from protocol import encode_frame

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")


def _envf(name, default):
    return float(os.environ.get(name, default))


def _envb(name, default):
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes", "on")


CONFIG = dict(
    N=int(_envf("M5_N", 1000)),
    g=_envf("M5_G", 3.5),
    mu_ext=_envf("M5_MU_EXT", 3.3),
    ei_ratio=_envf("M5_EI_RATIO", 0.8),
    tau_homeo=_envf("M5_TAU_HOMEO", 50_000.0),
    spatial=_envb("M5_SPATIAL", True),               # spatial money-shot on by default in M5
    locality=_envf("M5_LOCALITY", 0.35),
    distance_delays=_envb("M5_DIST_DELAYS", True),
)
FPS = _envf("M5_FPS", 30.0)
HIST_EVERY = int(FPS)                     # broadcast the avalanche histogram ~once per second


class Hub:
    """Shared simulation + connected clients + adjustable playback state."""
    def __init__(self):
        self.engine = SimEngine(**CONFIG)
        self.clients: set[WebSocket] = set()
        self.steps_per_frame = int(_envf("M5_STEPS_PER_FRAME", 20))
        self.paused = False
        self.frame_id = 0

    async def broadcast_bytes(self, data: bytes):
        for ws in list(self.clients):
            try:
                await ws.send_bytes(data)
            except Exception:
                self.clients.discard(ws)

    async def broadcast_json(self, obj: dict):
        for ws in list(self.clients):
            try:
                await ws.send_json(obj)
            except Exception:
                self.clients.discard(ws)


hub: Hub | None = None


async def broadcaster():
    """Paced render loop: step -> encode -> broadcast, capped at FPS. Idle when nobody watches."""
    loop = asyncio.get_event_loop()
    period = 1.0 / FPS
    while True:
        t0 = loop.time()
        if hub.clients and not hub.paused:
            batch = await asyncio.to_thread(hub.engine.step, hub.steps_per_frame)
            await hub.broadcast_bytes(encode_frame(hub.frame_id, batch, hub.engine.N))
            if hub.frame_id % HIST_EVERY == 0:
                h = hub.engine.avalanche_histogram()
                h["type"] = "hist"
                await hub.broadcast_json(h)
            hub.frame_id += 1
        dt = loop.time() - t0
        await asyncio.sleep(max(0.0, period - dt))


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global hub
    hub = Hub()
    task = asyncio.create_task(broadcaster())
    print(f"[m5] SimEngine ready  backend={hub.engine.meta()['backend']}  "
          f"N={hub.engine.N}  cfg={CONFIG}  FPS={FPS}  steps/frame={hub.steps_per_frame}")
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    hub.clients.add(websocket)
    await websocket.send_json(hub.engine.meta())          # one-time hello (N, is_inhib, edges, cfg)
    try:
        while True:
            msg = await websocket.receive_json()          # control channel (speed / pause)
            cmd = msg.get("cmd")
            if cmd == "speed":
                hub.steps_per_frame = max(1, min(2000, int(msg.get("value", 20))))
            elif cmd == "pause":
                hub.paused = bool(msg.get("value", False))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        hub.clients.discard(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("M5_HOST", "127.0.0.1"),
                port=int(os.environ.get("M5_PORT", 8000)), log_level="info")
