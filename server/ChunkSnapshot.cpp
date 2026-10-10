#include "ChunkSnapshot.hpp"
#include "ChunkIds.hpp"
#include "../utils/LocalScene.hpp"
#include "../space/OctreeNode.hpp"
#include "../space/OctreeNodeData.hpp"
#include <shared_mutex>

namespace chunksnap {

chunkproto::ChunkRecord makeRecord(ChunkIdRegistry& ids, const OctreeNodeData& nd, uint8_t layer) {
    chunkproto::ChunkRecord r;
    const OctreeNode* n = nd.node;
    // Opaque registry id — never the node pointer (see ChunkIds.hpp).
    r.id = ids.getOrAssign(const_cast<OctreeNode*>(n));
    r.minX = nd.cube.getMin().x;
    r.minY = nd.cube.getMin().y;
    r.minZ = nd.cube.getMin().z;
    r.size = nd.cube.getLengthX();
    r.level = nd.level;
    r.layer = layer;
    if (n) {
        r.lod = n->getLod();
        r.chunkLod = n->getChunkLod();
        r.isLeaf = n->isLeaf() ? 1 : 0;
        r.spaceType = static_cast<uint8_t>(n->getType());
        r.version = n->version;
        r.brushIndex = n->getBrush();
        for (int i = 0; i < 8; ++i) r.sdf[i] = n->sdf[i];
    }
    return r;
}

static void walkTree(const Octree& tree, uint8_t layer, ChunkIdRegistry& ids,
                     std::vector<chunkproto::ChunkRecord>& out) {
    if (!tree.root) return;
    std::shared_lock<std::shared_mutex> lock(tree.treeMutex);
    struct Item { OctreeNode* node; BoundingCube cube; uint32_t level; };
    std::vector<Item> stack;
    stack.reserve(4096);
    stack.push_back({tree.root, static_cast<const BoundingCube&>(tree), 0});
    while (!stack.empty()) {
        Item cur = stack.back();
        stack.pop_back();
        OctreeNode* node = cur.node;
        if (!node) continue;
        if (node->isChunk()) {
            OctreeNodeData nd(cur.level, node, cur.cube, nullptr);
            out.push_back(makeRecord(ids, nd, layer));
            continue; // chunks are frontier: do not descend (mirrors notifyChunkNodes)
        }
        OctreeNode* children[8] = {};
        node->getChildren(*tree.allocator, children);
        for (int i = 7; i >= 0; --i) {
            if (children[i])
                stack.push_back({children[i], cur.cube.getChild(i), cur.level + 1});
        }
    }
}

std::vector<chunkproto::ChunkRecord> collectAll(ChunkIdRegistry& ids, LocalScene& scene) {
    std::vector<chunkproto::ChunkRecord> out;
    out.reserve(8192);
    const size_t n = scene.layerCount();
    for (size_t i = 0; i < n; ++i) {
        Octree* tree = scene.getLayerOctree(static_cast<Layer>(i));
        if (tree) walkTree(*tree, static_cast<uint8_t>(i), ids, out);
    }
    return out;
}

static void metaForTree(const Octree& tree, uint8_t layer,
                        std::vector<chunkproto::SceneMetaLayer>& out) {
    std::shared_lock<std::shared_mutex> lock(tree.treeMutex);
    chunkproto::SceneMetaLayer m;
    m.layer = layer;
    m.rootMinX = tree.getMin().x;
    m.rootMinY = tree.getMin().y;
    m.rootMinZ = tree.getMin().z;
    m.rootLen = tree.getLengthX();
    m.chunkSize = tree.chunkSize;
    m.rootChunkLod = tree.root ? tree.root->getChunkLod() : 0;
    out.push_back(m);
}

std::vector<chunkproto::SceneMetaLayer> collectMeta(LocalScene& scene) {
    std::vector<chunkproto::SceneMetaLayer> out;
    const size_t n = scene.layerCount();
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        Octree* tree = scene.getLayerOctree(static_cast<Layer>(i));
        if (tree) metaForTree(*tree, static_cast<uint8_t>(i), out);
    }
    return out;
}

} // namespace chunksnap
