#pragma once

// Protocol-compatible empty mesh reply for remote clients.
//
// The server is SDF-only: `ChunkRecord` already carries each chunk's cube and
// eight corner samples, and the client GPU runs the topology walk. MESH_DATA
// therefore never carries Cartesian vertices; the host does not evaluate or
// tessellate them.
//
// The id is OPAQUE (see ChunkIds.hpp): it is resolved to a live node through
// the scene's registry and verified after a structural walk, so a stale id
// simply yields "not found" instead of a dangling pointer.

#include "ChunkProtocol.hpp"
#include <cstdint>

class ChunkIdRegistry;
class LocalScene;

namespace chunkmesh {

// Resolves the opaque id to a live node and fills `out`'s identity/version
// fields with an empty vertex/index stream. Returns true when the id names a
// live chunk node; false when the id is unknown/stale.
bool buildChunkMesh(ChunkIdRegistry& ids, LocalScene& scene, uint64_t id,
                    chunkproto::MeshData& out);

} // namespace chunkmesh
