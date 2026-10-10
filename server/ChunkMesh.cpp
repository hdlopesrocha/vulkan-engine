#include "ChunkMesh.hpp"
#include "ChunkIds.hpp"
#include "../utils/LocalScene.hpp"
#include "../space/OctreeNode.hpp"
#include "../space/OctreeNodeData.hpp"
#include "../space/Tesselator.hpp"
#include "../space/ThreadContext.hpp"
#include <shared_mutex>
#include <vector>

namespace chunkmesh {
namespace {

// Tessellates one already-resolved live node. Holds the tree shared lock
// across the whole tessellation — the same nesting pattern
// LocalScene::requestModel3D uses (outer shared lock, then iterateTriangles
// takes its own shared guard).
void tessellateNode(const Octree& tree, OctreeNode* node, const BoundingCube& cube,
                    uint32_t level, uint8_t layer, uint64_t id, chunkproto::MeshData& out) {
    out.id = id;
    out.layer = layer;
    out.version = node->version;
    const uint8_t chunkLod = node->getChunkLod();
    // Only surface chunks at a tessellation rung carry geometry.
    if (chunkLod == 0 || node->getType() != SpaceType::Surface)
        return;
    long triCount = 0;
    Tesselator tess(&triCount);
    ThreadContext ctx;
    tree.iterateTriangles(node, cube, level, tess, &ctx, chunkLod);
    const Geometry& g = tess.geometry;
    out.positions.reserve(g.vertices.size() * 3);
    for (const Vertex& v : g.vertices) {
        out.positions.push_back(v.position.x);
        out.positions.push_back(v.position.y);
        out.positions.push_back(v.position.z);
    }
    out.indices.assign(g.indices.begin(), g.indices.end());
}

// Finds the node in one tree by POINTER (the registry already mapped the
// opaque id to it) and tessellates it. The pointer walk only confirms the
// node is still part of the live tree; identity comes from the registry.
bool tessellateInTree(const Octree& tree, OctreeNode* target, uint8_t layer,
                      uint64_t id, chunkproto::MeshData& out) {
    std::shared_lock<std::shared_mutex> lock(tree.treeMutex);
    if (!tree.root || !target) return false;

    struct Item { OctreeNode* node; BoundingCube cube; uint32_t level; };
    std::vector<Item> stack;
    stack.reserve(4096);
    stack.push_back({tree.root, static_cast<const BoundingCube&>(tree), 0});

    while (!stack.empty()) {
        Item cur = stack.back();
        stack.pop_back();
        OctreeNode* node = cur.node;
        if (!node) continue;
        if (node == target) {
            tessellateNode(tree, node, cur.cube, cur.level, layer, id, out);
            return true;
        }
        OctreeNode* children[8] = {};
        node->getChildren(*tree.allocator, children);
        for (int i = 7; i >= 0; --i) {
            if (children[i])
                stack.push_back({children[i], cur.cube.getChild(i), cur.level + 1});
        }
    }
    return false;
}

} // namespace

bool buildChunkMesh(ChunkIdRegistry& ids, LocalScene& scene, uint64_t id,
                    chunkproto::MeshData& out) {
    out = chunkproto::MeshData();
    out.id = id;
    // Resolve the opaque id BEFORE touching the tree. Unknown/stale ids
    // (including 0 and recycled-pointer aliases) stop here — no raw id is
    // ever cast back to a pointer.
    OctreeNode* target = ids.lookup(id);
    if (!target) return false;
    bool found = false;
    const size_t n = scene.layerCount();
    for (size_t i = 0; i < n && !found; ++i) {
        Octree* tree = scene.getLayerOctree(static_cast<Layer>(i));
        if (tree) found = tessellateInTree(*tree, target, static_cast<uint8_t>(i), id, out);
    }
    if (!found) return false;
    // Re-verify AFTER tessellation: a concurrent delete + allocator reuse
    // could have swapped the occupant mid-walk. On mismatch the reply is
    // discarded by the caller (empty mesh) instead of mislabeled geometry.
    if (!ids.verify(id, target)) {
        out = chunkproto::MeshData();
        out.id = id;
        return false;
    }
    return true;
}

} // namespace chunkmesh
