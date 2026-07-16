/* m5_viz/static/app.js — live SOqC visualizer client.
 * Decodes the binary FRAME protocol (mirror of protocol.py), lights up a Three.js neuron
 * cloud on spikes, and streams rate/⟨x⟩/avalanche-distribution into uPlot charts. */
'use strict';

const $ = (id) => document.getElementById(id);
const DECAY = 0.86;              // per-rAF glow decay
const STREAM_W = 500;            // rolling points kept in the time-series charts

// ---- state -------------------------------------------------------------------------
let N = 0, dtMs = 0.1, isInhib = null, act = null;
let THREEok = (typeof THREE !== 'undefined'), UPok = (typeof uPlot !== 'undefined');
let points = null, geom = null, colorAttr = null, baseCol = null, peakCol = null;
let renderer, scene, camera, world = null, edgeLines = null, cloudPos = null;
let edgesVisible = true, mode = 'spikes', xU8 = null;
const RADIUS = 30;
const EMPTY16 = new Uint16Array(0);

// inferno-style colormap: depleted ⟨x⟩ → dark purple, recovered → bright yellow
const CMAP = [[0.00,[0.02,0.01,0.09]],[0.25,[0.30,0.05,0.36]],[0.50,[0.68,0.12,0.30]],
              [0.75,[0.96,0.42,0.09]],[1.00,[0.99,0.92,0.55]]];
function colormap(t) {
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  for (let k = 1; k < CMAP.length; k++) {
    if (t <= CMAP[k][0]) {
      const a = CMAP[k-1], b = CMAP[k], f = (t - a[0]) / (b[0] - a[0] || 1);
      return [a[1][0]+f*(b[1][0]-a[1][0]), a[1][1]+f*(b[1][1]-a[1][1]), a[1][2]+f*(b[1][2]-a[1][2])];
    }
  }
  return CMAP[CMAP.length-1][1];
}

// ---- Three.js neuron cloud ---------------------------------------------------------
function sprite() {
  const c = document.createElement('canvas'); c.width = c.height = 64;
  const g = c.getContext('2d').createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(0.35, 'rgba(255,255,255,.55)');
  g.addColorStop(1, 'rgba(255,255,255,0)');
  const ctx = c.getContext('2d'); ctx.fillStyle = g; ctx.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

function buildCloud(positionsFlat) {
  if (!THREEok) return;
  const stage = $('stage');
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(55, stage.clientWidth / stage.clientHeight, 0.1, 5000);
  camera.position.set(0, 0, 78);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(stage.clientWidth, stage.clientHeight);
  renderer.setClearColor(0x0a0e14, 0);
  stage.appendChild(renderer.domElement);
  world = new THREE.Group(); scene.add(world);

  cloudPos = new Float32Array(N * 3);
  const col = new Float32Array(N * 3);
  const GA = Math.PI * (1 + Math.sqrt(5));
  baseCol = new Float32Array(N * 3); peakCol = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    if (positionsFlat && positionsFlat.length >= N * 3) {           // backend geometry (spatial)
      cloudPos[i*3]   = positionsFlat[i*3]   * RADIUS;
      cloudPos[i*3+1] = positionsFlat[i*3+1] * RADIUS;
      cloudPos[i*3+2] = positionsFlat[i*3+2] * RADIUS;
    } else {                                                        // fallback Fibonacci sphere
      const phi = Math.acos(1 - 2 * (i + 0.5) / N), th = GA * (i + 0.5);
      cloudPos[i*3] = RADIUS*Math.sin(phi)*Math.cos(th);
      cloudPos[i*3+1] = RADIUS*Math.sin(phi)*Math.sin(th);
      cloudPos[i*3+2] = RADIUS*Math.cos(phi);
    }
    const inh = isInhib && isInhib[i];
    const b = inh ? [0.42, 0.13, 0.17] : [0.09, 0.32, 0.46];   // dim base
    const p = inh ? [1.0, 0.55, 0.62]  : [0.55, 0.90, 1.0];    // hot peak
    for (let c = 0; c < 3; c++) { baseCol[i*3+c] = b[c]; peakCol[i*3+c] = p[c]; col[i*3+c] = b[c]; }
  }
  geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(cloudPos, 3));
  colorAttr = new THREE.BufferAttribute(col, 3);
  geom.setAttribute('color', colorAttr);
  const mat = new THREE.PointsMaterial({ size: 1.7, map: sprite(), vertexColors: true,
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true });
  points = new THREE.Points(geom, mat);
  world.add(points);
}

function buildEdges(edgesFlat) {
  if (!THREEok || !world || !edgesFlat || edgesFlat.length < 2) return;
  const M = edgesFlat.length >> 1;
  const verts = new Float32Array(M * 6);
  for (let e = 0; e < M; e++) {
    const i = edgesFlat[e*2], j = edgesFlat[e*2+1];
    verts[e*6]   = cloudPos[i*3];   verts[e*6+1] = cloudPos[i*3+1]; verts[e*6+2] = cloudPos[i*3+2];
    verts[e*6+3] = cloudPos[j*3];   verts[e*6+4] = cloudPos[j*3+1]; verts[e*6+5] = cloudPos[j*3+2];
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  const m = new THREE.LineBasicMaterial({ color: 0x2f6da8, transparent: true, opacity: 0.09,
    blending: THREE.AdditiveBlending, depthWrite: false });
  edgeLines = new THREE.LineSegments(g, m);
  edgeLines.visible = edgesVisible;
  world.add(edgeLines);
}

function animate() {
  requestAnimationFrame(animate);
  if (!points || !act) return;
  const col = colorAttr.array;
  if (mode === 'heatmap') {
    for (let i = 0; i < N; i++) {
      const a = act[i] *= DECAY;                        // spike -> brief white flash
      const c = colormap(xU8 ? xU8[i] / 255 : 1);       // steady color = current resource
      const w = a > 1 ? 1 : a, i3 = i * 3;
      col[i3]   = c[0] + (1 - c[0]) * w;
      col[i3+1] = c[1] + (1 - c[1]) * w;
      col[i3+2] = c[2] + (1 - c[2]) * w;
    }
  } else {
    for (let i = 0; i < N; i++) {
      const a = act[i] *= DECAY;
      const i3 = i * 3;
      col[i3]   = baseCol[i3]   + a * (peakCol[i3]   - baseCol[i3]);
      col[i3+1] = baseCol[i3+1] + a * (peakCol[i3+1] - baseCol[i3+1]);
      col[i3+2] = baseCol[i3+2] + a * (peakCol[i3+2] - baseCol[i3+2]);
    }
  }
  colorAttr.needsUpdate = true;
  if (world) { world.rotation.y += 0.0016; world.rotation.x = 0.18 * Math.sin(performance.now() * 0.00007); }
  renderer.render(scene, camera);
}

// ---- uPlot charts ------------------------------------------------------------------
function mkStream(el, stroke) {
  if (!UPok) return null;
  const w = el.clientWidth || 340;
  const opts = { width: w, height: el.clientHeight || 120, cursor: { show: false }, legend: { show: false },
    scales: { x: { time: false } },
    axes: [{ stroke: '#6b7c90', grid: { stroke: '#1e2a3a' }, size: 24 },
           { stroke: '#6b7c90', grid: { stroke: '#1e2a3a' }, size: 38 }],
    series: [{}, { stroke, width: 1.5, points: { show: false } }] };
  const u = new uPlot(opts, [[], []], el);
  return { u, xs: [], ys: [] };
}
function pushStream(s, t, y) {
  if (!s) return;
  s.xs.push(t); s.ys.push(y);
  if (s.xs.length > STREAM_W) { s.xs.shift(); s.ys.shift(); }
  s.u.setData([s.xs, s.ys]);
}
function mkHist(el) {
  if (!UPok) return null;
  const opts = { width: el.clientWidth || 340, height: el.clientHeight || 150,
    cursor: { show: false }, legend: { show: false },
    scales: { x: { distr: 3 }, y: { distr: 3 } },
    axes: [{ stroke: '#6b7c90', grid: { stroke: '#1e2a3a' }, size: 24 },
           { stroke: '#6b7c90', grid: { stroke: '#1e2a3a' }, size: 38 }],
    series: [{}, { stroke: '#f2c14e', width: 1.5, points: { show: true, size: 4, fill: '#f2c14e' } }] };
  return new uPlot(opts, [[1], [1]], el);
}
let cRate, cXbar, cHist;

// ---- WebSocket ---------------------------------------------------------------------
function decodeFrame(buf) {
  const dv = new DataView(buf);
  const absStep = dv.getUint32(5, true);
  const rate = dv.getFloat32(9, true), xbar = dv.getFloat32(13, true);
  const Nf = dv.getUint16(17, true), nGlow = dv.getUint32(19, true);
  const x = new Uint8Array(buf, 23, Nf);                                 // resource heatmap (1-byte)
  const io = 23 + Nf;                                                    // ids may sit at odd offset
  const ids = nGlow ? new Uint16Array(buf.slice(io, io + 2 * nGlow)) : EMPTY16;  // copy -> aligned
  return { rate, xbar, absStep, x, ids };
}

function onFrame(f) {
  xU8 = f.x;                                                             // latest per-neuron ⟨x⟩
  if (act) for (let k = 0; k < f.ids.length; k++) act[f.ids[k]] = 1.0;   // ignite glow / flash
  $('s-rate').textContent = f.rate.toFixed(1);
  $('s-xbar').textContent = f.xbar.toFixed(3);
  const t = f.absStep * dtMs / 1000;                                     // bio-seconds
  pushStream(cRate, t, f.rate); pushStream(cXbar, t, f.xbar);
}

function setMode(m) {
  mode = m;
  if (points) {
    const mat = points.material;
    if (m === 'heatmap') { mat.blending = THREE.NormalBlending; mat.depthWrite = true; mat.size = 2.3; }
    else { mat.blending = THREE.AdditiveBlending; mat.depthWrite = false; mat.size = 1.7; }
    mat.needsUpdate = true;
  }
  const le = $('legend-ei'), lh = $('legend-heat');
  if (le) le.style.display = m === 'heatmap' ? 'none' : '';
  if (lh) lh.style.display = m === 'heatmap' ? '' : 'none';
  const bs = $('mode-spikes'), bh = $('mode-heat');
  if (bs) bs.classList.toggle('active', m === 'spikes');
  if (bh) bh.classList.toggle('active', m === 'heatmap');
}

function onHello(m) {
  N = m.N; dtMs = m.dt_ms; isInhib = m.is_inhib; act = new Float32Array(N);
  const mode = m.spatial ? `spatial λ=${m.cfg.locality}` : 'random graph';
  $('sub').textContent = `N=${N} · ${m.cfg.ei_ratio*100|0}% exc · g=${m.cfg.g} · μ=${m.cfg.mu_ext} · ${mode}`;
  $('backend').textContent = `backend: ${m.backend}`;
  buildCloud(m.positions);
  buildEdges(m.edges);
  setMode(mode);                                          // apply render mode to the new points
  const et = $('edges-toggle');
  if (et) { et.disabled = !(m.edges && m.edges.length); et.checked = edgesVisible && !et.disabled; }
}

function onHist(m) {
  $('s-naval').textContent = m.n_aval; $('s-max').textContent = m.max_size;
  if (cHist) {
    const xs = m.edges, ys = m.counts.map((c) => (c > 0 ? c : null));
    cHist.setData([xs, ys]);
  }
}

function connect() {
  const host = location.host || '127.0.0.1:8000';        // file:// has empty host -> fall back
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url = `${proto}://${host}/ws`;
  console.log('[m5] connecting to', url);
  $('sub').textContent = 'connecting to ' + url + ' …';
  let ws;
  try { ws = new WebSocket(url); }
  catch (e) { console.error('[m5] WebSocket construction failed:', e);
    $('sub').textContent = 'WS construction error: ' + e.message; setTimeout(connect, 1500); return; }
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { $('conn').className = 'dot on'; window._ws = ws; console.log('[m5] connected'); };
  ws.onerror = () => { $('conn').className = 'dot off';
    $('sub').textContent = 'cannot reach ' + url + ' — is the server up there? (see console)'; };
  ws.onclose = () => { $('conn').className = 'dot off'; setTimeout(connect, 1000); };
  ws.onmessage = (e) => {
    try {
      if (typeof e.data === 'string') { const m = JSON.parse(e.data);
        if (m.type === 'hello') onHello(m); else if (m.type === 'hist') onHist(m); }
      else onFrame(decodeFrame(e.data));
    } catch (err) { console.error('[m5] message handler error:', err); }
  };
}

// ---- controls + resize -------------------------------------------------------------
function initControls() {
  const sp = $('speed');
  sp.addEventListener('input', () => {
    $('speed-val').textContent = sp.value + ' st/f';
    if (window._ws && _ws.readyState === 1) _ws.send(JSON.stringify({ cmd: 'speed', value: +sp.value }));
  });
  let paused = false;
  $('pause').addEventListener('click', () => {
    paused = !paused; $('pause').textContent = paused ? 'resume' : 'pause';
    if (window._ws && _ws.readyState === 1) _ws.send(JSON.stringify({ cmd: 'pause', value: paused }));
  });
  const et = $('edges-toggle');
  if (et) et.addEventListener('change', () => {
    edgesVisible = et.checked; if (edgeLines) edgeLines.visible = edgesVisible;
  });
  const bs = $('mode-spikes'), bh = $('mode-heat');
  if (bs) bs.addEventListener('click', () => setMode('spikes'));
  if (bh) bh.addEventListener('click', () => setMode('heatmap'));
  window.addEventListener('resize', () => {
    if (renderer) { const s = $('stage'); camera.aspect = s.clientWidth / s.clientHeight;
      camera.updateProjectionMatrix(); renderer.setSize(s.clientWidth, s.clientHeight); }
    if (cRate) cRate.u.setSize({ width: $('c-rate').clientWidth, height: 120 });
    if (cXbar) cXbar.u.setSize({ width: $('c-xbar').clientWidth, height: 120 });
    if (cHist) cHist.setSize({ width: $('c-hist').clientWidth, height: 150 });
  });
}

// ---- boot --------------------------------------------------------------------------
function boot() {
  connect();                                     // FIRST — a chart/3D error must never block the socket
  try {
    cRate = mkStream($('c-rate'), '#39c2ff');
    cXbar = mkStream($('c-xbar'), '#37d67a');
    cHist = mkHist($('c-hist'));
  } catch (e) { console.error('[m5] chart init failed (uPlot):', e); }
  try {
    if (!THREEok) console.warn('[m5] three.js not loaded — cloud disabled (CDN blocked?)');
    initControls();
    animate();
  } catch (e) { console.error('[m5] 3D/controls init failed (three.js):', e); }
}
// DOMContentLoaded may already have fired (script is at end of body) — handle both cases
if (document.readyState === 'loading') window.addEventListener('DOMContentLoaded', boot);
else boot();
