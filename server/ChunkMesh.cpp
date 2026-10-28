#include "ChunkMesh.hpp"
#include "ChunkIds.hpp"
#include "../utils/LocalScene.hpp"
#include "../space/OctreeNode.hpp"
#include "../space/OctreeNodeData.hpp"
#include <atomic>
#include <iostream>
#include <shared_mutex>
#include <vector>

namespace chunkmesh {
namespace {

// Finds the node in one tree by pointer. This is a structural walk only (cube
// child indices); it never evaluates an SDF or builds a surface vertex.
bool containsNode(const Octree& tree, OctreeNode* target) {
    if (!tree.root || !target) return false;
    std::shared_lock<std::shared_mutex> lock(tree.treeMutex);
    std::vector<OctreeNode*> stack;
    stack.reserve(4096);
    stack.push_back(tree.root);
    while (!stack.empty()) {
        OctreeNode* node = stack.back();
        stack.pop_back();
        if (!node) continue;
        if (node == target) return true;
        OctreeNode* children[8] = {};
        node->getChildren(*tree.allocator, children);
        for (int i = 0; i < 8; ++i)
            if (children[i]) stack.push_back(children[i]);
    }
    return false;
}

} // namespace

// The wire is SDF-only now: chunk records carry the cube and eight corner
// samples, and the receiving GPU runs the topology walk. The mesh message is
// kept for protocol compatibility but always carries an empty vertex/index
// stream; the host never computes Cartesian surface vertices.
bool buildChunkMesh(ChunkIdRegistry& ids, LocalScene& scene, uint64_t id,
                    chunkproto::MeshData& out) {
    out = chunkproto::MeshData();
    out.id = id;
    OctreeNode* target = ids.lookup(id);
    if (!target) return false;

    static std::atomic<bool> logged{false};
    bool expected = false;
    if (logged.compare_exchange_strong(expected, true)) {
        std::cerr << "[chunkmesh] mesh streaming is SDF-only: MESH_DATA carries no geometry; "
                     "clients build vertices on the GPU from the chunk records\n";
    }

    bool found = false;
    const size_t n = scene.layerCount();
    for (size_t i = 0; i < n && !found; ++i) {
        Layer* l = scene.layerAt(i);
        Octree* tree = l ? l->octree() : nullptr;
        if (!tree) continue;
        if (containsNode(*tree, target)) {
            found = true;
            out.layer = static_cast<uint8_t>(l->id());
            out.version = target->version;
        }
    }
    if (!found) return false;
    // Re-verify AFTER the walk: a concurrent delete + allocator reuse could
    // have swapped the occupant. On mismatch the reply is discarded.
    if (!ids.verify(id, target)) {
        out = chunkproto::MeshData();
        out.id = id;
        return false;
    }
    return true;
}

} // namespace chunkmesh
