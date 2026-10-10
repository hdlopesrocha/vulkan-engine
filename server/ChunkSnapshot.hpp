#pragma once

// Snapshot helpers: convert live octree nodes into ChunkRecords and collect
// every chunk currently in a LocalScene (both layers).
//
// Chunk ids come from a ChunkIdRegistry (opaque monotonic numbers) — raw
// node pointers must never reach the wire. Pass the scene's registry into
// every function here.

#include "ChunkProtocol.hpp"
#include <vector>

class ChunkIdRegistry;
class LocalScene;
struct OctreeNodeData;

namespace chunksnap {

// Build one record from an octree callback payload. `layer` is 0/1.
chunkproto::ChunkRecord makeRecord(ChunkIdRegistry& ids, const OctreeNodeData& nd, uint8_t layer);

std::vector<chunkproto::ChunkRecord> collectAll(ChunkIdRegistry& ids, LocalScene& scene);

// Scene-level metadata (root lattice origin + ladder depth per layer) for
// the LoD rung gate. Thread-safe like collectAll.
std::vector<chunkproto::SceneMetaLayer> collectMeta(LocalScene& scene);

} // namespace chunksnap
