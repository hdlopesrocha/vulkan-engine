#pragma once

#include <cstdint>

#include "LodMesh.hpp"
#include "../utils/Scene.hpp"

// One queued chunk publish. Originally nested in SceneRenderer; kept as a
// standalone record so publishers and streaming targets can name it without
// pulling in the full renderer.
struct PendingMeshData {
    Layer::Id layer;
    NodeID nid;
    LoDMesh lodMesh;
    OctreeNodeData nodeData;   // world cube of the source node (stable band center)
    // Pad to a multiple of 8 bytes. An odd-sized value type makes the
    // std::unordered_map hash node's in-place value slot 1 byte smaller than
    // sizeof(PendingMeshData) used by the (compiler-generated) move/copy
    // constructor, so moving a PendingMeshData out of the map reads 1 byte
    // past the node and corrupts the adjacent heap chunk header (app aborts
    // on free with "corrupted size vs. prev_size").
    uint8_t _pad[7] = {};
};
