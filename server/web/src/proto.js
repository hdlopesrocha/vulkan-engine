const MAGIC = [0x43, 0x48, 0x4B, 0x31]; // 'CHK1'
const T_REQ_ALL = 0x01, T_REQ_MESH = 0x02;
const T_START = 0x10, T_BATCH = 0x11, T_DEL = 0x12, T_END = 0x13, T_MESH = 0x14, T_META = 0x15,
  T_MESH_BATCH = 0x16;
const REC_SIZE = 76;

async function gzipEncode(u8) {
  const cs = new CompressionStream('gzip');
  const src = new Blob([u8]).stream().pipeThrough(cs);
  return new Uint8Array(await new Response(src).arrayBuffer());
}
async function gzipDecode(ab) {
  const ds = new DecompressionStream('gzip');
  const src = new Blob([ab]).stream().pipeThrough(ds);
  return new Uint8Array(await new Response(src).arrayBuffer());
}
function parseRecord(dv, off) {
  const id = dv.getBigUint64(off, true).toString();
  const minX = dv.getFloat32(off + 8, true), minY = dv.getFloat32(off + 12, true), minZ = dv.getFloat32(off + 16, true);
  const size = dv.getFloat32(off + 20, true);
  const level = dv.getUint32(off + 24, true);
  const lod = dv.getUint8(off + 28), chunkLod = dv.getUint8(off + 29);
  const isLeaf = dv.getUint8(off + 30), spaceType = dv.getUint8(off + 31), layer = dv.getUint8(off + 32);
  const version = dv.getUint32(off + 36, true);
  const brushIndex = dv.getInt32(off + 40, true);
  const sdf = [];
  for (let i = 0; i < 8; i++) sdf.push(dv.getFloat32(off + 44 + 4 * i, true));
  return { id, minX, minY, minZ, size, level, lod, chunkLod, isLeaf, spaceType, layer, version, brushIndex, sdf };
}
// Debug brush colors — exact mirror of the Vulkan DEBUG_MODE_MATERIAL_INDEX
// views, so the wireframe shows the same color a brush has in the app:
// solid uses the 16-entry palette in shaders/includes/solid/SolidSurface.glsl
// (floored mod, so negative ids wrap), water the golden-ratio hue in
// shaders/includes/water/WaterSurface.glsl (negative ids show magenta).
const BRUSH_PALETTE = [
  [0.90,0.10,0.10],[0.10,0.90,0.10],[0.10,0.10,0.90],[0.90,0.90,0.10],
  [0.90,0.10,0.90],[0.10,0.90,0.90],[1.00,0.55,0.10],[0.55,0.35,0.15],
  [0.60,0.20,0.80],[1.00,0.50,0.70],[0.70,1.00,0.30],[0.00,0.45,0.55],
  [0.05,0.10,0.35],[0.45,0.50,0.10],[0.60,0.60,0.60],[1.00,1.00,1.00],
];
function brushColor(brushIndex, renderer) {
  // renderer is 'water' or 'solid' (resolved from the layer directory, not
  // the numeric id, so added layers color correctly).
  if (renderer === 'water') {
    if (brushIndex < 0) return [1.0, 0.0, 1.0];
    const hh = (((brushIndex * 0.61803398875) % 1) + 1) % 1;
    const t = Math.PI * 2;
    return [0.0, 0.33, 0.67].map(o => Math.min(1, Math.max(0, 0.5 + 0.5 * Math.cos(t * (hh + o)))));
  }
  return BRUSH_PALETTE[((brushIndex % 16) + 16) % 16];
}
// SCENE_META extension section: u8 layer, u8 renderer (0 Solid, 1 Water),
// u8 nameLen, name bytes. Returns { id: { name, renderer } } ('solid'|'water').
function parseMetaLayers(raw) {
  const out = {};
  if (raw.length < 9) return out;
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const n = dv.getUint32(5, true);
  let off = 9 + n * 24; // skip the fixed LoD-gate entries
  const dec = new TextDecoder();
  for (let i = 0; i < n && off + 3 <= raw.length; i++) {
    const id = dv.getUint8(off), renderer = dv.getUint8(off + 1), nameLen = dv.getUint8(off + 2);
    off += 3;
    if (off + nameLen > raw.length) break;
    let name = '';
    try { name = dec.decode(raw.subarray(off, off + nameLen)); } catch (e) {}
    off += nameLen;
    out[id] = { name, renderer: renderer === 1 ? 'water' : 'solid' };
  }
  return out;
}
function parseMeshEntry(dv, rawLen, off) {
  // One MeshData entry at absolute offset off (entry-relative layout, no
  // magic/type): u64 id, u32 version, u8 layer, pad[3], u32 nVerts, u32 nIdx,
  // verts, idx, trailing brush array (may be absent). Returns { m, next }
  // with next = first byte after the entry, or null when truncated.
  if (off + 24 > rawLen) return null;
  const id = dv.getBigUint64(off, true).toString();
  const version = dv.getUint32(off + 8, true);
  const layer = dv.getUint8(off + 12);
  const nVerts = dv.getUint32(off + 16, true), nIdx = dv.getUint32(off + 20, true);
  const body = 24 + nVerts * 12 + nIdx * 4;
  if (off + body > rawLen) return null;
  const pos = new Float32Array(nVerts * 3);
  let o = off + 24;
  for (let i = 0; i < nVerts * 3; i++, o += 4) pos[i] = dv.getFloat32(o, true);
  const idx = new Uint32Array(nIdx);
  for (let i = 0; i < nIdx; i++, o += 4) idx[i] = dv.getUint32(o, true);
  let brush = null;
  if (rawLen >= o + nVerts * 4) {
    brush = new Int32Array(nVerts);
    for (let i = 0; i < nVerts; i++, o += 4) brush[i] = dv.getInt32(o, true);
  }
  return { m: { id, version, layer, pos, idx, brush }, next: o };
}
function parseMesh(raw) {
  // MESH_DATA: magic[4], type, then one entry (see parseMeshEntry). The
  // trailing brush array may be absent (old senders) -> brush is null and
  // callers fall back to the chunk record's brush.
  if (raw.length < 29) return null;
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const e = parseMeshEntry(dv, raw.length, 5);
  return e ? e.m : null;
}
export { MAGIC, T_REQ_ALL, T_REQ_MESH, T_START, T_BATCH, T_DEL, T_END, T_MESH,
  T_META, T_MESH_BATCH, REC_SIZE, gzipEncode, gzipDecode, parseRecord,
  BRUSH_PALETTE, brushColor, parseMetaLayers, parseMeshEntry, parseMesh };
