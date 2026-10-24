#pragma once

// Binary + gzipped wire protocol between the headless server and the web UI.
//
// Transport: WebSocket binary frames. EVERY frame payload is a gzip blob
// (zlib gzip wrapper). The browser uses CompressionStream/DecompressionStream.
//
// Client -> Server (gunzipped):
//   byte[0] = 0x01  REQUEST_ALL  (no further payload)
//   byte[0] = 0x02  REQUEST_MESH (bytes[1..8] = u64 chunk id, LE)
//   byte[0] = 0x03  REQUEST_ALL_MESHES (no further payload: stream one
//                           MESH_DATA per live chunk, skipping empties)
//
// Server -> Client (gunzipped):
//   bytes[0..3] = 'C','H','K','1'  magic
//   byte[4]     = message type
//   payload ... (all integers little-endian, floats IEEE754 LE)
//     0x10 SNAPSHOT_START : u32 totalChunks
//     0x11 CHUNK_BATCH    : u16 count, then count x 76-byte ChunkRecord
//     0x12 DELETE_BATCH   : u16 count, then count x u64 nodeId
//     0x13 SNAPSHOT_END   : no payload
//     0x14 MESH_DATA      : u64 chunk id, u32 version, u8 layer, u8 pad[3],
//                           u32 vertCount, u32 idxCount,
//                           vertCount x 3 f32 positions, idxCount x u32 indices,
//                           vertCount x i32 per-vertex brushIndex.
//                           Kept for protocol compatibility: the server is
//                           SDF-only, so it always sends vertCount == 0 and
//                           clients build vertices on the GPU from the chunk
//                           records (cube + eight corner samples). The trailing
//                           brush array may be absent (old senders).
//     0x15 SCENE_META     : u32 layerCount, then per layer:
//                           u8 layer, f32 rootMin xyz, f32 rootLen,
//                           f32 chunkSize, u8 rootChunkLod (stored, +1-shifted),
//                           u8 pad[2];
//                           then an extension section, per layer:
//                           u8 layer, u8 renderer (0 Solid, 1 Water),
//                           u8 nameLen, name bytes (UTF-8, not NUL-terminated).
//                           Sent once per snapshot after START so render
//                           clients can set up the LoD rung gate (root lattice
//                           origin + ladder depth) exactly and name layers.
//                           Receivers that predate the extension stop after
//                           the fixed entries and ignore the tail.
//     0x16 MESH_BATCH      : u16 count, then count x MeshData entries.
//                           Each entry is encoded exactly like the 0x14
//                           payload (u64 id, u32 version, u8 layer, pad[3],
//                           u32 vertCount, u32 idxCount, verts, idx, brush
//                           array). Pushed unsolicited: once per snapshot
//                           after the CHUNK_BATCH frames, and after every
//                           live upsert flush for the upserted ids. Empty
//                           meshes are omitted; receivers treat a missing id
//                           as "no geometry". Old receivers ignore the type.
//
// Chunk ids are OPAQUE server-assigned monotonic numbers (1, 2, 3, ...).
// They are never node pointers and never reused while the mapping lives:
// deleting a chunk retires its id, so allocator recycling of node memory
// cannot alias a new chunk onto an old id. 0 means invalid/unknown.
//
// ChunkRecord (76 bytes, packed, little-endian):
//   0  u64 id            opaque chunk id (see above), session-scoped
//   8  f32 minX,minY,minZ
//   20 f32 size          cube edge length
//   24 u32 level
//   28 u8  lod           (+1-shifted stored ladder level, 0 = unset)
//   29 u8  chunkLod      (+1-shifted, 0 = unset)
//   30 u8  isLeaf        (0/1)
//   31 u8  spaceType     (0 Empty, 1 Surface, 2 Solid -- matches SpaceType enum)
//   32 u8  layer         (0 opaque, 1 transparent)
//   33 u8  pad[3]
//   36 u32 version
//   40 i32 brushIndex
//   44 f32 sdf[8]

#include <cstdint>
#include <string>
#include <vector>

namespace chunkproto {

constexpr uint8_t MSG_REQUEST_ALL   = 0x01;
constexpr uint8_t MSG_REQUEST_MESH  = 0x02;
constexpr uint8_t MSG_REQUEST_ALL_MESHES = 0x03;

constexpr uint8_t MSG_SNAPSHOT_START = 0x10;
constexpr uint8_t MSG_CHUNK_BATCH    = 0x11;
constexpr uint8_t MSG_DELETE_BATCH   = 0x12;
constexpr uint8_t MSG_SNAPSHOT_END   = 0x13;
constexpr uint8_t MSG_MESH_DATA      = 0x14;
constexpr uint8_t MSG_SCENE_META     = 0x15;
constexpr uint8_t MSG_MESH_BATCH     = 0x16;

constexpr uint8_t LAYER_OPAQUE      = 0;
constexpr uint8_t LAYER_TRANSPARENT = 1;

constexpr size_t CHUNK_RECORD_SIZE = 76;

struct ChunkRecord {
    uint64_t id = 0;
    float minX = 0.0f, minY = 0.0f, minZ = 0.0f;
    float size = 0.0f;
    uint32_t level = 0;
    uint8_t lod = 0;
    uint8_t chunkLod = 0;
    uint8_t isLeaf = 0;
    uint8_t spaceType = 1;
    uint8_t layer = 0;
    uint32_t version = 0;
    int32_t brushIndex = 0;
    float sdf[8] = {0,0,0,0,0,0,0,0};
};

// ---- gzip (zlib gzip wrapper) ----
std::vector<uint8_t> gzipCompress(const uint8_t* data, size_t len);
inline std::vector<uint8_t> gzipCompress(const std::vector<uint8_t>& in) {
    if (in.empty()) return gzipCompress(nullptr, 0);
    return gzipCompress(in.data(), in.size());
}
std::vector<uint8_t> gzipDecompress(const uint8_t* data, size_t len);
inline std::vector<uint8_t> gzipDecompress(const std::vector<uint8_t>& in) {
    if (in.empty()) return {};
    return gzipDecompress(in.data(), in.size());
}

// ---- message builders (raw, UNCOMPRESSED; caller gzips before WS send) ----
std::vector<uint8_t> buildRequestAll();
std::vector<uint8_t> buildRequestMesh(uint64_t chunkId);
std::vector<uint8_t> buildRequestAllMeshes();
std::vector<uint8_t> buildSnapshotStart(uint32_t total);
std::vector<uint8_t> buildSnapshotEnd();
std::vector<uint8_t> buildChunkBatch(const ChunkRecord* recs, size_t count);
std::vector<uint8_t> buildDeleteBatch(const uint64_t* ids, size_t count);

// Scene-level metadata for render clients (LoD rung gate setup + layer
// directory). One entry per layer; rootChunkLod uses the stored
// +1-shifted representation; renderer is 0 (Solid) / 1 (Water).
struct SceneMetaLayer {
    uint8_t layer = 0;
    float rootMinX = 0.0f, rootMinY = 0.0f, rootMinZ = 0.0f;
    float rootLen = 0.0f;
    float chunkSize = 0.0f;
    uint8_t rootChunkLod = 0;
    uint8_t renderer = 0;
    std::string name;
};
std::vector<uint8_t> buildSceneMeta(const SceneMetaLayer* layers, size_t count);

// On-demand tessellated chunk geometry (positions are xyz triplets, brush
// holds one brushIndex per vertex for per-vertex debug coloring; it may be
// shorter than vertCount when the sender has no per-vertex data).
struct MeshData {
    uint64_t id = 0;
    uint32_t version = 0;
    uint8_t layer = 0;
    std::vector<float> positions;   // vertCount * 3
    std::vector<uint32_t> indices;
    std::vector<int32_t> brush;     // vertCount entries (may be empty)
};
std::vector<uint8_t> buildMeshData(const MeshData& mesh);
// One MESH_BATCH frame: count entries back-to-back (same per-entry layout
// as buildMeshData's payload). Callers chunk large sets into several frames
// to keep individual WS frames small.
std::vector<uint8_t> buildMeshBatch(const MeshData* meshes, size_t count);

// ---- record codec ----
void encodeRecord(const ChunkRecord& r, uint8_t* out76);
bool decodeRecord(const uint8_t* in, size_t avail, ChunkRecord& out);

} // namespace chunkproto
