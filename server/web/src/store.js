import { ref, computed } from 'vue';
import {
  MAGIC, T_REQ_ALL, T_REQ_MESH, T_START, T_BATCH, T_DEL, T_END, T_MESH,
  T_META, T_MESH_BATCH, REC_SIZE, gzipEncode, gzipDecode, parseRecord,
  brushColor, parseMetaLayers, parseMeshEntry, parseMesh,
} from './proto.js';

// ---- shared viewer state (chunks, selection, filters, mesh cache) ----
export const chunks = ref(new Map());
export const connected = ref(false);
export const selectedId = ref(null);
// Per-column filters (one search box per table column). The chunkLoD column
// defaults to '1', so the list AND the rendered chunks start on the frontier
// rung; clearing it (or typing a range like 1-3) renders more rungs. The same
// match set gates the table AND the rendered chunks (boxes, picking,
// wireframe) alike.
export const colFilter = ref({ id: '', layer: '', level: '', size: '', version: '', lod: '', chunkLod: '1', brush: '' });
export const sortKey = ref('level');
export const layerVisibility = ref({ 0: true, 1: true });
export const showWireframe = ref(true);
export const renderAll = ref(false);
export const allMeshCount = ref(0);
// Tessellation cache fed by pushed geometry (snapshot mesh batches + live
// upsert batches, plus any single replies). Entry:
// { pos, idx, brush, version, layer, gl, span, firstBrush, lastRenderer,
//   brushList }. GPU objects (.gl) are owned by gl.js.
export const allMeshes = new Map();
export const fps = ref(0);
export const culled = ref(0);
export const lastMsg = ref('no data yet');
export const wsTarget = ref('…');
export const wsError = ref('connecting…');
export const wsAttempts = ref(0);
export const meshStatus = ref('no selection');
export const expectedTotal = ref(0);
// Layer directory from SCENE_META (id -> { name, renderer }). Empty when the
// server predates the extension; every lookup below falls back to the
// historic 0/1 assumptions in that case.
export const layerInfo = ref({});
// Single-chunk view (selected wireframe). Drawn from the cache entry, so
// these are just an id + version pointer plus status text.
export const meshSel = { id: null, version: -1 };

let ws = null, chunksVersion = 0, reconnectTimer = 0, metaRev = 0;
export function getMetaRev() { return metaRev; }
export function getChunksVersion() { return chunksVersion; }
export function bumpChunksVersion() { chunksVersion++; }

// GL-eviction hook (installed by gl.js): frees an entry's VBO/IBO when the
// entry is replaced or dropped. Keeps this module free of GL imports.
let evictHook = null;
export function setMeshEvictHook(fn) { evictHook = fn; }
function evictGL(m) { if (evictHook) { try { evictHook(m); } catch (e) { /* GL teardown must not break message handling */ } } }

export function rendererOf(layer) {
  const info = layerInfo.value[layer];
  if (info) return info.renderer;
  return layer === 1 ? 'water' : 'solid';
}
export function layerName(l) {
  const info = layerInfo.value[l];
  if (info && info.name) return info.name;
  if (l === 0) return 'opaque';
  if (l === 1) return 'transparent';
  return 'layer ' + l;
}
export function layerTag(l) {
  const info = layerInfo.value[l];
  if (info && info.name) {
    const n = info.name.toLowerCase();
    if (n === 'opaque') return 'opq';
    if (n === 'transparent') return 'trn';
    return info.name.slice(0, 3).toLowerCase();
  }
  return l === 0 ? 'opq' : l === 1 ? 'trn' : 'L' + l;
}
export const layerList = computed(() => {
  const s = new Set(Object.keys(layerInfo.value).map(Number));
  for (const c of chunks.value.values()) s.add(c.layer);
  return [...s].sort((a, b) => a - b);
});
export const layerCounts = computed(() => {
  const m = {};
  for (const c of chunks.value.values()) m[c.layer] = (m[c.layer] || 0) + 1;
  return m;
});

export function normFilters() {
  const f = colFilter.value, n = {};
  for (const k of ['id', 'layer', 'level', 'size', 'version', 'lod', 'chunkLod', 'brush'])
    n[k] = (f[k] || '').trim().toLowerCase();
  return n;
}
export function matchNum(v, q) {
  if (!q) return true;
  let m = q.match(/^(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)$/);
  if (m) {
    const n = parseFloat(m[2]);
    if (m[1] === '>') return v > n;
    if (m[1] === '<') return v < n;
    if (m[1] === '>=') return v >= n;
    return v <= n;
  }
  m = q.match(/^(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)$/);
  if (m) {
    const a = parseFloat(m[1]), b = parseFloat(m[2]);
    return v >= Math.min(a, b) && v <= Math.max(a, b);
  }
  return String(v).includes(q);
}
export function matchesFilters(c, f) {
  if (f.id && !c.id.toLowerCase().includes(f.id)) return false;
  if (f.layer && !(String(c.layer) === f.layer || layerName(c.layer).toLowerCase().includes(f.layer))) return false;
  if (f.level && !matchNum(c.level, f.level)) return false;
  if (f.size && !matchNum(c.size, f.size)) return false;
  if (f.version && !matchNum(c.version, f.version)) return false;
  if (f.lod && !matchNum(c.lod, f.lod)) return false;
  if (f.chunkLod && !matchNum(c.chunkLod, f.chunkLod)) return false;
  if (f.brush && !matchNum(c.brushIndex, f.brush)) return false;
  return true;
}
export const filtered = computed(() => {
  const f = normFilters();
  const vis = layerVisibility.value;
  let arr = [...chunks.value.values()];
  arr = arr.filter(c => vis[c.layer] !== false && matchesFilters(c, f));
  const k = sortKey.value;
  arr.sort((a, b) => (a[k] < b[k] ? -1 : a[k] > b[k] ? 1 : 0));
  return arr;
});
export const selected = computed(() => selectedId.value ? (chunks.value.get(selectedId.value) || null) : null);

export function shortId(id) { return id.length > 10 ? '…' + id.slice(-8) : id; }
export function fmtF(x) { return (typeof x === 'number' && isFinite(x)) ? x.toFixed(1) : String(x); }
export function fmt3(a, b, c) { return `${fmtF(a)}, ${fmtF(b)}, ${fmtF(c)}`; }
export function spaceName(t) { return t === 0 ? 'Empty' : t === 1 ? 'Surface' : t === 2 ? 'Solid' : '?'; }
export function sortBy(k) { sortKey.value = k; }

// Store one decoded mesh in the render-all cache (replaces stale versions).
// Returns true when the cache changed.
export function cacheMesh(m) {
  const prev = allMeshes.get(m.id);
  if (prev && m.version < prev.version) return false;
  if (prev) evictGL(prev); // version bump: drop the old VBO/IBO
  allMeshes.set(m.id, {
    pos: m.pos, idx: m.idx,
    brush: (m.brush && m.brush.length === m.pos.length / 3) ? m.brush : null,
    version: m.version, layer: m.layer,
    gl: null, span: null, firstBrush: null, lastRenderer: null, brushList: null,
  });
  allMeshCount.value = allMeshes.size;
  return true;
}

export function clearMesh(silent) {
  meshSel.id = null; meshSel.version = -1;
  if (!silent) meshStatus.value = 'no selection';
}
export function ensureSelectedMesh() {
  // Geometry arrives pushed (snapshot + live batches land in the cache);
  // adopt it locally, requesting only on a miss or a stale version
  // (e.g. selected in the ms between a record batch and its mesh batch).
  const id = selectedId.value;
  if (id == null) return;
  const c = chunks.value.get(id);
  const hit = allMeshes.get(id);
  if (hit && c && hit.version >= c.version && hit.idx && hit.idx.length > 0) {
    meshSel.id = id; meshSel.version = hit.version;
    const nTris = hit.idx.length / 3;
    meshStatus.value = nTris > 0
      ? `${nTris} tris · ${(hit.pos.length / 3)} verts · v${hit.version} (cached)`
      : `empty (non-surface v${hit.version})`;
  } else {
    requestMesh(id);
  }
}
export function select(id) {
  selectedId.value = id;
  chunksVersion++;
  ensureSelectedMesh();
}
export function clearAll() {
  chunks.value.clear(); selectedId.value = null; chunksVersion++;
  clearMesh();
  for (const m of allMeshes.values()) evictGL(m);
  allMeshes.clear(); allMeshCount.value = 0;
}
export function toggleRenderAll() {
  // Purely local view switch: geometry arrives pushed (snapshot mesh
  // batches + live upsert batches), so this never sends requests.
  lastMsg.value = renderAll.value
    ? `render-all on (${allMeshes.size} cached)`
    : 'render-all off';
}

export async function requestAll() {
  if (!ws || ws.readyState !== 1) return;
  const gz = await gzipEncode(new Uint8Array([T_REQ_ALL]));
  ws.send(gz);
  lastMsg.value = 'REQUEST_ALL sent (gzip binary)';
}

export async function requestMesh(idStr) {
  if (idStr == null) { clearMesh(); return; }
  if (!ws || ws.readyState !== 1) { meshStatus.value = 'offline — no socket'; return; }
  try {
    const p = new Uint8Array(9);
    new DataView(p.buffer).setBigUint64(1, BigInt(idStr), true);
    p[0] = T_REQ_MESH;
    meshStatus.value = 'loading…';
    ws.send(await gzipEncode(p));
    lastMsg.value = `REQUEST_MESH ${shortId(idStr)} sent (gzip binary)`;
  } catch (e) { meshStatus.value = 'request failed: ' + e; }
}

function handleMessage(raw) {
  if (raw.length < 5) return;
  for (let i = 0; i < 4; i++) if (raw[i] !== MAGIC[i]) return;
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const type = dv.getUint8(4);
  if (type === T_START) {
    expectedTotal.value = dv.getUint32(5, true);
    chunks.value.clear(); selectedId.value = null; chunksVersion++;
    clearMesh();
    lastMsg.value = `snapshot start: expect ${expectedTotal.value}`;
  } else if (type === T_BATCH) {
    const n = dv.getUint16(5, true);
    let selTouched = false;
    for (let i = 0; i < n; i++) {
      const off = 7 + i * REC_SIZE;
      if (off + REC_SIZE > raw.length) break;
      const c = parseRecord(dv, off);
      chunks.value.set(c.id, c);
      if (!(c.layer in layerVisibility.value)) layerVisibility.value[c.layer] = true;
      if (c.id === selectedId.value) selTouched = true;
    }
    chunksVersion++;
    lastMsg.value = `batch +${n} (total ${chunks.value.size}, gzip ${raw.length}B raw)`;
    // Live update of the selected chunk: adopt the pushed geometry
    // (its mesh batch follows the record batch), requesting on a miss.
    if (selTouched && selectedId.value != null) ensureSelectedMesh();
  } else if (type === T_DEL) {
    const n = dv.getUint16(5, true);
    for (let i = 0; i < n; i++) {
      const off = 7 + i * 8;
      if (off + 8 > raw.length) break;
      const id = dv.getBigUint64(off, true).toString();
      chunks.value.delete(id);
      const evicted = allMeshes.get(id);
      if (evicted) evictGL(evicted);
      if (allMeshes.delete(id)) allMeshCount.value = allMeshes.size;
      if (selectedId.value === id) { selectedId.value = null; clearMesh(); }
    }
    chunksVersion++;
    lastMsg.value = `delete −${n} (total ${chunks.value.size})`;
  } else if (type === T_END) {
    lastMsg.value = `snapshot end: ${chunks.value.size} chunks live`;
  } else if (type === T_META) {
    // Layer directory (replaces the previous map): id -> { name, renderer }.
    // Layers with no chunks yet still appear in pills/checkboxes with 0.
    const info = parseMetaLayers(raw);
    layerInfo.value = info;
    metaRev++;
    for (const id of Object.keys(info)) {
      if (!(id in layerVisibility.value)) layerVisibility.value[id] = true;
    }
    chunksVersion++;
    lastMsg.value = `layers: ${Object.keys(info).map(id => `${info[id].name || 'layer ' + id}(${info[id].renderer})`).join(', ') || 'none'}`;
  } else if (type === T_MESH) {
    const m = parseMesh(raw);
    if (!m) { lastMsg.value = 'mesh: short frame ignored'; return; }
    // Every reply feeds the render-all cache; the selected-only path below
    // keeps its exact-match rule for the single-chunk view.
    cacheMesh(m);
    if (renderAll.value) {
      const nTris = m.idx.length / 3;
      meshStatus.value = `${allMeshes.size} meshes cached`;
      lastMsg.value = `mesh ${shortId(m.id)}: ${nTris} tris (render-all)`;
      return;
    }
    if (m.id !== selectedId.value) return; // stale reply for an old selection
    meshSel.id = m.id; meshSel.version = m.version;
    const nTris = m.idx.length / 3;
    meshStatus.value = nTris > 0
      ? `${nTris} tris · ${(m.pos.length / 3)} verts · v${m.version}`
      : `empty (non-surface v${m.version})`;
    lastMsg.value = `mesh ${shortId(m.id)}: ${nTris} tris`;
  } else if (type === T_MESH_BATCH) {
    // Pushed geometry (snapshot + live upserts): cache every entry.
    // Empty meshes are never sent; a missing id means "no geometry".
    if (raw.length < 7) { lastMsg.value = 'mesh batch: short frame ignored'; return; }
    const bdv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const n = bdv.getUint16(5, true);
    let off = 7, cached = 0;
    for (let i = 0; i < n; i++) {
      const e = parseMeshEntry(bdv, raw.length, off);
      if (!e) break;
      off = e.next;
      if (cacheMesh(e.m)) cached++;
    }
    meshStatus.value = `${allMeshes.size} meshes cached`;
    lastMsg.value = `mesh batch +${cached} (total ${allMeshes.size})`;
  }
}

// WS endpoint resolution: ?ws= full override, ?port= override, same-host
// /ws by default, localhost:8080 fallback for file:// (location.host is
// empty there, which previously produced the broken URL "ws:///ws").
// Bind-all hosts (0.0.0.0, ::) are not dialable, so they map to localhost:
// opening the page as http://0.0.0.0:8080/ must still reach ws://localhost:8080/ws.
function dialableHostname() {
  const h = location.hostname || 'localhost';
  if (h === '0.0.0.0' || h === '::' || h === '[::]') return 'localhost';
  return h;
}
export function resolveWsUrl() {
  const scheme = location.protocol === 'https:' ? 'wss://' : 'ws://';
  try {
    const q = new URLSearchParams(location.search);
    const o = q.get('ws');
    if (o) return o;
    const p = q.get('port');
    if (p) return scheme + dialableHostname() + ':' + p + '/ws';
  } catch (e) { /* malformed query — fall through to defaults */ }
  if (location.protocol === 'file:' || !location.host) return 'ws://localhost:8080/ws';
  return scheme + dialableHostname() + (location.port ? ':' + location.port : '') + '/ws';
}

export function connect() {
  const url = resolveWsUrl();
  wsTarget.value = url;
  wsAttempts.value++;
  let opened = false;
  try {
    ws = new WebSocket(url);
  } catch (e) {
    wsError.value = 'bad URL (' + e + ')';
    scheduleReconnect();
    return;
  }
  ws.binaryType = 'arraybuffer';
  ws.onopen = async () => {
    opened = true;
    connected.value = true;
    wsError.value = 'connected';
    await requestAll();
  };
  ws.onclose = (ev) => {
    connected.value = false;
    const code = ev && ev.code ? ' (code ' + ev.code + (ev.reason ? ': ' + ev.reason : '') + ')' : '';
    wsError.value = opened
      ? 'connection lost' + code + ' — is the server still running?'
      : 'connection refused' + code + ' — is the server running on that host:port?';
    scheduleReconnect();
  };
  ws.onerror = () => { try { ws.close(); } catch (e) { /* already closing */ } };
  ws.onmessage = async (ev) => {
    try {
      const raw = await gzipDecode(ev.data);
      handleMessage(raw);
    } catch (e) { lastMsg.value = 'gzip decode error: ' + e; }
  };
}
function scheduleReconnect() {
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(connect, 1500);
}
export function reconnect() {
  clearTimeout(reconnectTimer);
  try { if (ws) ws.close(); } catch (e) { /* no socket */ }
  connect();
}
