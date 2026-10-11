import {
  chunks, layerVisibility, matchesFilters, normFilters, selectedId, renderAll,
  allMeshes, showWireframe, layerInfo, layerList, colFilter, culled,
  getChunksVersion, getMetaRev, setMeshEvictHook, select, sortKey,
} from './store.js';
import { brushColor } from './proto.js';
import { V3, aabbVisible } from './math3d.js';

// WebGL renderer. Passes per frame:
//   1. deferred depth pre-pass: mesh triangles (VBO + IBO) with color writes
//      off, so the depth buffer holds the solid surface;
//   2. main pass: frustum-culled chunk boxes (LINES) + wireframes (LINES)
//      with depth test on, so hidden edges are occluded correctly.
// Render-state snapshot for automated tests (read-only).
export const dbg = {
  meshVisible: false, meshTriCount: 0, meshLineCount: 0, vertCount: 0, culled: 0,
  lineMin: 0, lineMax: 0, lineSum: 0,
};

let gl = null, prog = null, posBuf = null, colBuf = null, uVP = null;
let uintIndexExt = null; // OES_element_index_uint for Uint32 IBOs (else Uint16/soup fallback)
let vertCount = 0, meshTriCount = 0, meshLineCount = 0;
let builtKey = '';

export function initGL(canvas) {
  gl = canvas.getContext('webgl', { antialias: true });
  if (!gl) return false;
  const vs = 'attribute vec3 aPos; attribute vec3 aCol; uniform mat4 uVP; varying vec3 vC;' +
    'void main(){ gl_Position = uVP * vec4(aPos, 1.0); vC = aCol; }';
  const fs = 'precision mediump float; varying vec3 vC; void main(){ gl_FragColor = vec4(vC, 1.0); }';
  function sh(t, s) { const h = gl.createShader(t); gl.shaderSource(h, s); gl.compileShader(h); return h; }
  prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(prog);
  gl.useProgram(prog);
  posBuf = gl.createBuffer(); colBuf = gl.createBuffer();
  uintIndexExt = gl.getExtension('OES_element_index_uint');
  uVP = gl.getUniformLocation(prog, 'uVP');
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL); // wireframe sits exactly on the pre-pass surface
  return true;
}

function layerColor(c) {
  if (c.spaceType === 0) return [0.42, 0.45, 0.5];
  if (c.spaceType === 2) return [0.96, 0.62, 0.04];
  // Box tint follows the layer's renderer (green = solid, cyan = water);
  // layers unknown to the directory keep the historic 0/1 colors, and
  // anything else gets a distinct golden-ratio hue.
  const info = layerInfo.value[c.layer];
  if (!info) {
    if (c.layer === 0) return [0.29, 0.87, 0.5];
    if (c.layer === 1) return [0.13, 0.83, 0.93];
  } else if (info.renderer === 'water') {
    return [0.13, 0.83, 0.93];
  } else {
    return [0.29, 0.87, 0.5];
  }
  const hh = (((c.layer * 0.61803398875) % 1) + 1) % 1;
  const t = Math.PI * 2;
  return [0.0, 0.33, 0.67].map(o => Math.min(1, Math.max(0, 0.35 + 0.65 * (0.5 + 0.5 * Math.cos(t * (hh + o))))));
}

const EDGES = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
function bindArr(buf, data, loc, size) {
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
}

// Rebuilds box line buffers, skipping chunks whose AABB sits fully outside
// the view frustum (CPU frustum culling). Returns culled count.
function rebuildBuffers(planes) {
  if (!gl) return 0;
  const pos = [], col = [];
  let culled = 0;
  const q = normFilters();
  for (const c of chunks.value.values()) {
    if (layerVisibility.value[c.layer] === false) continue;
    if (!matchesFilters(c, q)) continue; // column filters gate rendered chunks
    const mn = [c.minX, c.minY, c.minZ], mx = [c.minX + c.size, c.minY + c.size, c.minZ + c.size];
    if (planes && !aabbVisible(planes, mn, mx)) { culled++; continue; }
    const sel = selectedId.value === c.id;
    const cc = sel ? [0.98, 0.8, 0.08] : layerColor(c);
    const cx = [];
    for (let i = 0; i < 8; i++)
      cx.push([c.minX + ((i & 4) ? c.size : 0), c.minY + ((i & 2) ? c.size : 0), c.minZ + ((i & 1) ? c.size : 0)]);
    for (const [a, b] of EDGES) {
      pos.push(...cx[a], ...cx[b]);
      col.push(...cc, ...cc);
    }
  }
  vertCount = pos.length / 3;
  const aPos = gl.getAttribLocation(prog, 'aPos');
  const aCol = gl.getAttribLocation(prog, 'aCol');
  bindArr(posBuf, new Float32Array(pos), aPos, 3);
  bindArr(colBuf, new Float32Array(col), aCol, 3);
  return culled;
}

// Per-mesh GPU objects (VBO + IBO, STATIC_DRAW): uploaded ONCE per mesh
// version on first draw, then only bound — the triangle soup is never
// re-expanded or re-uploaded per frame. Cache entry shape:
// { pos, idx, brush, version, layer, gl, span, firstBrush, lastRenderer,
//   brushList } with gl = { posVBO, triIBO, triIdxCount, triType,
//   triSoupVBO, lineVBO, lineColVBO, lineCount, ver, colRev } | null.
export function freeMeshGL(m) {
  if (!gl || !m || !m.gl) return;
  const g = m.gl;
  m.gl = null;
  for (const b of [g.posVBO, g.triIBO, g.triSoupVBO, g.lineVBO, g.lineColVBO])
    if (b) gl.deleteBuffer(b);
}
export function installEvictHook() { setMeshEvictHook(freeMeshGL); }

function uploadStaticVBO(data) {
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  return buf;
}

// Bake line colors for one mesh (positions + per-endpoint brush colors).
// Pure CPU side; upload happens in ensureMeshGL/refreshMeshColors.
function bakeMeshLines(m, chunk, rendererOf) {
  const nVerts = m.pos.length / 3;
  const brushArr = (m.brush && m.brush.length === nVerts) ? m.brush : null;
  const fallback = chunk ? chunk.brushIndex : 0;
  const renderer = rendererOf(chunk ? chunk.layer : 0);
  const lp = [], lc = [];
  const brushSet = new Set();
  let lmin = Infinity, lmax = -Infinity, lsum = 0, firstBrush = null;
  const tris = [];
  for (let t = 0; t + 2 < m.idx.length; t += 3) {
    const a = m.idx[t], b = m.idx[t + 1], d = m.idx[t + 2];
    if (a >= nVerts || b >= nVerts || d >= nVerts) continue; // corrupt index: drop the tri
    tris.push(a, b, d);
  }
  for (let t = 0; t + 2 < tris.length; t += 3) {
    const vs = [tris[t], tris[t + 1], tris[t + 1], tris[t + 2], tris[t + 2], tris[t]];
    for (const vi of vs) {
      const vx = m.pos[vi * 3], vy = m.pos[vi * 3 + 1], vz = m.pos[vi * 3 + 2];
      lp.push(vx, vy, vz);
      if (vx < lmin) lmin = vx;
      if (vx > lmax) lmax = vx;
      if (vy < lmin) lmin = vy;
      if (vy > lmax) lmax = vy;
      if (vz < lmin) lmin = vz;
      if (vz > lmax) lmax = vz;
      lsum += vx + vy + vz;
      const bi = brushArr ? brushArr[vi] : fallback;
      if (firstBrush === null) firstBrush = bi;
      const cc = brushColor(bi, renderer);
      lc.push(cc[0], cc[1], cc[2]);
      brushSet.add(bi);
    }
  }
  return {
    tris, lp: new Float32Array(lp), lc: new Float32Array(lc),
    brushes: [...brushSet].sort((x, y) => x - y), firstBrush, renderer,
    span: { min: lmin, max: lmax, sum: lsum },
  };
}

// Upload all GPU objects for a cache entry (once per version). chunk supplies
// the brush fallback + renderer for the baked line colors. Colors rebake when
// the layer directory (metaRev) changes under us.
function ensureMeshGL(m, chunk, metaRev, rendererOf) {
  if (!gl || !m.pos || !m.idx || m.idx.length === 0) return null;
  if (m.gl && m.gl.ver === m.version) {
    if (m.gl.colRev !== metaRev) {
      const baked = bakeMeshLines(m, chunk, rendererOf);
      gl.bindBuffer(gl.ARRAY_BUFFER, m.gl.lineColVBO);
      gl.bufferData(gl.ARRAY_BUFFER, baked.lc, gl.STATIC_DRAW);
      m.gl.colRev = metaRev;
      m.span = baked.span;
      m.firstBrush = baked.firstBrush;
      m.lastRenderer = baked.renderer;
      m.brushList = baked.brushes;
    }
    return m.gl;
  }
  freeMeshGL(m);
  const nVerts = m.pos.length / 3;
  const g = {
    posVBO: null, triIBO: null, triIdxCount: 0, triType: 0,
    triSoupVBO: null, lineVBO: null, lineColVBO: null, lineCount: 0,
    ver: m.version, colRev: metaRev,
  };
  g.posVBO = uploadStaticVBO(m.pos);
  const baked = bakeMeshLines(m, chunk, rendererOf);
  if (baked.tris.length > 0) {
    g.triIdxCount = baked.tris.length;
    if (uintIndexExt || nVerts <= 65535) {
      // Indexed triangles: shared position VBO + IBO (the fast path).
      g.triIBO = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, g.triIBO);
      if (nVerts <= 65535) {
        g.triType = gl.UNSIGNED_SHORT;
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, Uint16Array.from(baked.tris), gl.STATIC_DRAW);
      } else {
        g.triType = gl.UNSIGNED_INT;
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, Uint32Array.from(baked.tris), gl.STATIC_DRAW);
      }
    } else {
      // No uint-index support and >64k verts: expanded soup fallback.
      const soup = new Float32Array(baked.tris.length * 3);
      for (let i = 0; i < baked.tris.length; i++) {
        soup[i * 3] = m.pos[baked.tris[i] * 3];
        soup[i * 3 + 1] = m.pos[baked.tris[i] * 3 + 1];
        soup[i * 3 + 2] = m.pos[baked.tris[i] * 3 + 2];
      }
      g.triSoupVBO = uploadStaticVBO(soup);
    }
  }
  g.lineVBO = uploadStaticVBO(baked.lp);
  g.lineColVBO = uploadStaticVBO(baked.lc);
  g.lineCount = baked.lp.length / 3;
  m.span = baked.span;
  m.firstBrush = baked.firstBrush;
  m.lastRenderer = baked.renderer;
  m.brushList = baked.brushes;
  m.gl = g;
  return g;
}

// One frame: rebuild culled box buffers when inputs change, bind + draw mesh
// VBO/IBOs. Returns nothing; updates culled + dbg.
export function renderFrame(vp, planes, aspect, rendererOf, poseKey) {
  // Box buffers depend on data + filters + frustum, so the rebuild key
  // includes a quantized camera pose (rebuilt while moving, cached idle).
  const visKey = layerList.value.map(l => layerVisibility.value[l] ? 1 : 0).join(',');
  const metaRev = getMetaRev();
  const key = [getChunksVersion(), visKey, metaRev, JSON.stringify(colFilter.value), selectedId.value, aspect,
    poseKey || ''].join('|');
  if (key !== builtKey) {
    culled.value = rebuildBuffers(planes);
    builtKey = key;
  }
  // Visible meshes: render-all draws every cached mesh passing filters +
  // frustum, otherwise just the selected chunk's cached entry. GPU objects
  // are uploaded once (ensureMeshGL) — per frame each mesh is only
  // frustum-tested on CPU, then its VBO/IBO are bound and drawn.
  const meshJobs = [];
  if (showWireframe.value) {
    const q = normFilters();
    const consider = (id) => {
      const m = allMeshes.get(id);
      if (!m) return;
      const c = chunks.value.get(id);
      if (!c || layerVisibility.value[c.layer] === false) return;
      if (!matchesFilters(c, q)) return;
      if (!aabbVisible(planes, [c.minX, c.minY, c.minZ],
        [c.minX + c.size, c.minY + c.size, c.minZ + c.size])) return;
      const g = ensureMeshGL(m, c, metaRev, rendererOf);
      if (g && (g.triIdxCount > 0 || g.lineCount > 0)) meshJobs.push({ g, m });
    };
    if (renderAll.value) {
      for (const id of allMeshes.keys()) consider(id);
    } else if (selectedId.value != null) {
      consider(selectedId.value);
    }
  }
  meshTriCount = 0; meshLineCount = 0;
  let lmin = Infinity, lmax = -Infinity, lsum = 0;
  let wireColor = [1.0, 1.0, 1.0], haveWireColor = false;
  const wireBrushSet = new Set();
  const wireRenderers = new Set();
  for (const job of meshJobs) {
    meshTriCount += job.g.triIdxCount / 3;
    meshLineCount += job.g.lineCount;
    if (job.m.span) {
      if (job.m.span.min < lmin) lmin = job.m.span.min;
      if (job.m.span.max > lmax) lmax = job.m.span.max;
      lsum += job.m.span.sum;
    }
    if (job.m.brushList) for (const b of job.m.brushList) wireBrushSet.add(b);
    if (job.m.lastRenderer) wireRenderers.add(job.m.lastRenderer);
    if (!haveWireColor && job.m.firstBrush !== undefined && job.m.firstBrush !== null) {
      wireColor = brushColor(job.m.firstBrush, job.m.lastRenderer || 'solid');
      haveWireColor = true;
    }
  }
  dbg.wireBrushes = [...wireBrushSet].sort((x, y) => x - y);
  if (wireRenderers.size === 1) dbg.wireRenderer = [...wireRenderers][0];
  else if (wireRenderers.size > 1) dbg.wireRenderer = 'mixed';
  dbg.wireColor = wireColor;
  dbg.lineMin = lmin; dbg.lineMax = lmax; dbg.lineSum = lsum;
  dbg.meshTriCount = meshTriCount; dbg.meshLineCount = meshLineCount; dbg.vertCount = vertCount;
  dbg.meshVisible = meshJobs.length > 0; dbg.culled = culled.value;
  const cv = gl.canvas;
  const aPos = gl.getAttribLocation(prog, 'aPos');
  const aCol = gl.getAttribLocation(prog, 'aCol');
  gl.viewport(0, 0, cv.width, cv.height);
  gl.clearColor(0.008, 0.02, 0.05, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.uniformMatrix4fv(uVP, false, vp);
  function bindVBO(buf, loc) {
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
  }
  // 1. deferred depth pre-pass: mesh triangles (VBO + IBO) write depth only.
  if (meshJobs.length > 0) {
    gl.colorMask(false, false, false, false);
    gl.depthMask(true);
    for (const job of meshJobs) {
      const g = job.g;
      if (g.triIdxCount === 0) continue;
      if (g.triIBO) {
        bindVBO(g.posVBO, aPos);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, g.triIBO);
        gl.drawElements(gl.TRIANGLES, g.triIdxCount, g.triType, 0);
      } else {
        bindVBO(g.triSoupVBO, aPos);
        gl.drawArrays(gl.TRIANGLES, 0, g.triIdxCount);
      }
    }
    gl.colorMask(true, true, true, true);
  }
  // 2. main pass: culled boxes + depth-tested wireframe (line VBOs).
  gl.depthMask(true);
  bindVBO(posBuf, aPos);
  bindVBO(colBuf, aCol);
  if (vertCount) gl.drawArrays(gl.LINES, 0, vertCount);
  for (const job of meshJobs) {
    const g = job.g;
    if (g.lineCount === 0) continue;
    bindVBO(g.lineVBO, aPos);
    bindVBO(g.lineColVBO, aCol);
    gl.drawArrays(gl.LINES, 0, g.lineCount);
  }
}

// Camera-pose part of the box rebuild key is passed per frame by the loop
// (see ViewportCanvas): boxes rebuild while moving, cached when idle.

// Click picking: nearest filtered chunk box along the NDC ray. Routes through
// select() so a canvas pick requests the mesh exactly like a row click.
export function pick(nx, ny, unproj) {
  if (!gl) return;
  const p0 = unproj(nx, ny, -1), p1 = unproj(nx, ny, 1);
  const dir = V3.norm(V3.sub(p1, p0));
  let best = null, bestT = 1e30;
  const pq = normFilters();
  for (const c of chunks.value.values()) {
    if (layerVisibility.value[c.layer] === false) continue;
    if (!matchesFilters(c, pq)) continue; // filtered out: not pickable
    const mn = [c.minX, c.minY, c.minZ], mx = [c.minX + c.size, c.minY + c.size, c.minZ + c.size];
    let t0 = 0, t1 = 1e30, ok = true;
    for (let i = 0; i < 3; i++) {
      const o = p0[i], dd = dir[i];
      if (Math.abs(dd) < 1e-9) { if (o < mn[i] || o > mx[i]) { ok = false; break; } }
      else {
        let a = (mn[i] - o) / dd, b = (mx[i] - o) / dd;
        if (a > b) { const t = a; a = b; b = t; }
        t0 = Math.max(t0, a); t1 = Math.min(t1, b);
        if (t0 > t1) { ok = false; break; }
      }
    }
    if (ok && t0 < bestT && t0 > 0) { bestT = t0; best = c.id; }
  }
  if (best) select(best);
}
