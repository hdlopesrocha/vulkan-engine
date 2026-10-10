#pragma once

// On-demand tessellation of a single chunk for remote render clients.
// Unlike LocalScene::requestModel3D (which dedups via an emitted-version
// cache for the Vulkan renderer), this always tessellates fresh so repeated
// requests always return geometry.
//
// The id is OPAQUE (see ChunkIds.hpp): it is resolved to a live node through
// the scene's registry, verified before AND after tessellation, and a stale
// id simply yields "not found" instead of a dangling pointer.

#include "ChunkProtocol.hpp"
#include <cstdint>

class ChunkIdRegistry;
class LocalScene;

namespace chunkmesh {

// Tessellates the chunk with the given opaque id into `out`. Returns true
// when the id names a live chunk node (`out` may still carry zero verts when
// the chunk is non-surface); false when the id is unknown/stale (caller
// should reply with an empty MESH_DATA).
bool buildChunkMesh(ChunkIdRegistry& ids, LocalScene& scene, uint64_t id,
                    chunkproto::MeshData& out);

} // namespace chunkmesh
