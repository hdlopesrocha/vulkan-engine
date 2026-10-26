#include "LocalScene.hpp"
#include "../space/OctreeFile.hpp"
#include "../space/OctreeNode.hpp"
#include "../space/OctreeAllocator.hpp"
#include "../services/OctreeTopologyTypes.hpp"
#include "../services/OctreeTopologyCapacity.hpp"
#include "../space/IteratorHandler.hpp"
#include "../sdf/SDF.hpp"
#include "../math/Math.hpp"
#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstring>
#include <shared_mutex>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <unordered_map>
#include <utility>
#include <vector>
#include <glm/glm.hpp>
#include <filesystem>

namespace {
struct SceneBundleHeader {
    char magic[8];
    uint32_t version;
    uint32_t hasSettings;
};

constexpr const char kSceneBundleMagic[8] = {'S', 'C', 'N', 'B', 'N', 'D', 'L', '1'};
constexpr uint32_t kSceneBundleVersionV1 = 1;
constexpr uint32_t kSceneBundleVersionV2 = 2;
constexpr uint32_t kSceneBundleVersion = kSceneBundleVersionV2;

bool cubeIntersectsCube(const glm::vec3& aMin, const glm::vec3& aMax, const glm::vec3& bMin, const glm::vec3& bMax) {
    for (int i = 0; i < 3; ++i) {
        if (aMax[i] < bMin[i] || aMin[i] > bMax[i]) return false;
    }
    return true;
}

// Surface-cell discovery/compaction: a scan cell can emit a segment only when
// at least one of its 12 cube edges has corners on both sides of the surface.
// The cube edge graph is connected, so that is exactly "at least one negative
// corner and at least one non-negative corner". INFINITY is a non-negative
// sentinel: SDF::eval skips it when classifying the node, so the stored type
// alone is not a safe filter (a negative+INFINITY node is typed Solid yet the
// shader still emits its crossing edges). Dropping the other cells from the
// scan list is therefore exact and shrinks the walk dispatch.
//
// The same pass counts the crossing edges because that count, not the cell
// count, is what bounds the GPU index stream: every crossing edge produces at
// most one segment per dyadic interval (see OctreeTopologyCapacity.hpp).
static uint32_t countCrossEdges(const OctreeNode* n) {
    uint32_t count = 0;
    for (const glm::ivec2& edge : SDF_EDGES) {
        if ((n->sdf[edge.x] < 0.0f) != (n->sdf[edge.y] < 0.0f)) ++count;
    }
    return count;
}

// Flattens the octree neighborhood the GPU topology walker needs for one
// chunk: the anchor's ancestor path (root-consistent cubes), the node table
// reachable by the neighbour descent (nodes coarser than targetLod inside the
// chunk + one cell, plus their children, which are the descent's stopping
// cells) and the scan cells at targetLod (walkLadder order).
struct GpuTopologyBuilder {
    const Octree& tree;
    OctreeTopologyInput& out;
    int targetLod;
    glm::vec3 domainMin;
    glm::vec3 domainMax;
    // Largest cell at targetLod in the subtree, INCLUDING cells filtered out
    // of the scan list. The search-domain margin must cover the largest
    // possible neighbour cell (collectBreaks samples a non-surface neighbour's
    // cube faces too), so deriving it from the compacted list would be wrong.
    float maxScanCellSize = 0.0f;
    // Cross-edge counts aligned with out.cells. Kept here (not in the input)
    // because they are only needed until the capacity hint is computed.
    std::vector<uint32_t> cellCrossEdges;
    uint32_t crossEdgeCount = 0;
    std::unordered_map<OctreeNode*, uint32_t> ids;

    GpuTopologyBuilder(const Octree& t, OctreeTopologyInput& o, int lod)
        : tree(t), out(o), targetLod(lod) {}

    uint32_t addNode(OctreeNode* n, const BoundingCube& cube) {
        auto it = ids.find(n);
        if (it != ids.end()) return it->second;
        OctreeGpuNode rec{};
        for (int i = 0; i < 8; ++i) rec.sdf[i] = n->sdf[i];
        rec.cubeMin = cube.getMin();
        rec.cubeSize = cube.getLengthX();
        rec.lod = static_cast<int32_t>(n->getLod());
        rec.type = static_cast<int32_t>(n->getType());
        rec.brushIndex = n->brushIndex;
        rec.textureType = n->textureType;
        rec.data = n->data;
        rec.flags = n->isLeaf() ? 1 : 0;
        rec.hsv = n->hsv;
        for (int i = 0; i < 8; ++i) rec.children[i] = -1;
        const uint32_t id = static_cast<uint32_t>(out.nodes.size());
        out.nodes.push_back(rec);
        ids[n] = id;
        return id;
    }

    // Mirrors the findCellAt descent: descend while the node is coarser than
    // targetLod and keep every child whose cube can be sampled.
    void descend(uint32_t id, OctreeNode* n, const BoundingCube& cube) {
        if (n->isLeaf() || static_cast<int>(n->getLod()) <= targetLod) return;
        OctreeNode* children[8] = {};
        n->getChildren(*tree.allocator, children);
        for (int i = 0; i < 8; ++i) {
            if (children[i] == nullptr) continue;
            BoundingCube childCube = cube.getChild(i);
            if (!cubeIntersectsCube(childCube.getMin(), childCube.getMax(), domainMin, domainMax)) continue;
            const uint32_t childId = addNode(children[i], childCube);
            out.nodes[id].children[i] = static_cast<int32_t>(childId);
            descend(childId, children[i], childCube);
        }
    }

    // Mirrors walkLadder: cells exactly at targetLod, depth-first in child order.
    void collectCells(OctreeNode* n, const BoundingCube& cube) {
        if (n == nullptr) return;
        const int lod = static_cast<int>(n->getLod());
        if (lod == targetLod) {
            // Discovery/compaction: only cells with a zero crossing can emit.
            maxScanCellSize = std::max(maxScanCellSize, cube.getLengthX());
            const uint32_t crossings = countCrossEdges(n);
            if (crossings != 0) {
                out.cells.push_back(addNode(n, cube));
                cellCrossEdges.push_back(crossings);
                crossEdgeCount += crossings;
            }
        } else if (lod > targetLod) {
            OctreeNode* children[8] = {};
            n->getChildren(*tree.allocator, children);
            for (int i = 0; i < 8; ++i) {
                if (children[i] != nullptr) collectCells(children[i], cube.getChild(i));
            }
        }
    }
};

// Builds the flattening for one chunk. Returns false when the chunk node
// cannot be reached from the world root (stale parameters) so the caller can
// fall back to the CPU walker.
bool buildGpuTopologyInput(const Octree& tree, OctreeNode* chunk, const BoundingCube& chunkCubeHint,
                           int targetLod, OctreeTopologyInput& out) {
    if (tree.root == nullptr || chunk == nullptr) return false;

    // Root-consistent ancestor chain: replay the child indices from the world
    // root so every cube matches the shader's own descent (BoundingCube drift
    // from cached traversal cubes could otherwise flip getChildIndex).
    std::vector<std::pair<OctreeNode*, BoundingCube>> chain;
    OctreeNode* node = tree.root;
    BoundingCube cube = static_cast<const BoundingCube&>(tree);
    chain.push_back({node, cube});
    const glm::vec3 center = chunkCubeHint.getCenter();
    while (node != nullptr && node != chunk) {
        if (node->isLeaf() || chain.size() > 200) return false;
        OctreeNode* children[8] = {};
        node->getChildren(*tree.allocator, children);
        const int childIndex = cube.getChildIndex(center);
        if (children[childIndex] == nullptr) return false;
        cube = cube.getChild(childIndex);
        node = children[childIndex];
        chain.push_back({node, cube});
    }
    if (node != chunk) return false;
    const BoundingCube chunkCube = chain.back().second;

    GpuTopologyBuilder builder(tree, out, targetLod);

    // Table entry 0 MUST be the world root: the shader's findCellAt starts its
    // descent there (mirroring the CPU walker). Claim it before any other node.
    const uint32_t rootId = builder.addNode(chain[0].first, chain[0].second);
    builder.collectCells(chunk, chunkCube);

    out.targetLod = targetLod;
    // Descent anchor: the root descent stops at the first node on this path
    // whose stored lod <= targetLod. Starting there is bit-identical for every
    // sample inside its cube; if no ancestor stops earlier, the chunk itself is
    // the safe start (all its ancestors are strictly above the stop level).
    // Starting below that node would skip the stop and return a finer cell.
    OctreeNode* startNode = chunk;
    BoundingCube startCube = chunkCube;
    for (const auto& link : chain) {
        if (link.first->getLod() <= targetLod) {
            startNode = link.first;
            startCube = link.second;
            break;
        }
    }
    out.fromId = builder.addNode(startNode, startCube);
    out.fromBrush = chunk->brushIndex;
    out.rootMin = tree.getMin();
    out.rootSize = tree.getLengthX();
    out.crossEdgeCount = builder.crossEdgeCount;

    // No surface cell at this rung: this is a valid empty chunk, not a
    // flattening failure. computeResident() reports ok with zero geometry and
    // the caller settles the revision as empty.
    if (out.cells.empty()) return true;

    // Sample points are cell corners nudged outward by one ulp, so one cell of
    // margin covers everything collectBreaks can query. Use the largest cell at
    // targetLod, including the non-surface cells compacted out of `cells`: they
    // are still sampled as neighbours and their cube faces add breaks.
    const float cellSize = builder.maxScanCellSize;
    const glm::vec3 chunkMin = chunkCube.getMin();
    const float chunkLength = chunkCube.getLengthX();
    builder.domainMin = chunkMin - glm::vec3(cellSize);
    builder.domainMax = chunkMin + glm::vec3(chunkLength + cellSize);

    // Start the table's descent at the world root, exactly like the CPU walker:
    // sample points nudged across the chunk boundary follow the sibling's path,
    // so the table must contain the root-to-cell path for the whole search
    // domain. (The shader anchors individual lookups at the chunk node when the
    // sample is inside it and falls back to this root otherwise.)
    builder.descend(rootId, chain[0].first, chain[0].second);

    // Cross-edge-derived index capacity. `descend` has now added every node
    // the shader's findCellAt can return, so the smallest cube in the table is
    // the lattice on which collectBreaks() can place a break. Per scan cell,
    // an edge of size S contributes at most S/smallest intervals; summing that
    // over the cell's crossing edges (instead of using the chunk's largest
    // scan cell for all of them) keeps the bound tight without under-sizing.
    float smallestCellSize = chunkLength;
    for (const OctreeGpuNode& n : out.nodes) {
        if (n.cubeSize > 0.0f) smallestCellSize = std::min(smallestCellSize, n.cubeSize);
    }
    uint64_t segmentCount = 0;
    uint32_t maxSegments = 1;
    for (size_t i = 0; i < out.cells.size(); ++i) {
        const OctreeGpuNode& cell = out.nodes[out.cells[i]];
        const uint32_t segments =
            octree_topology::segmentsForCellSize(cell.cubeSize, smallestCellSize);
        maxSegments = std::max(maxSegments, segments);
        segmentCount += static_cast<uint64_t>(segments) * builder.cellCrossEdges[i];
    }
    out.maxSegmentsPerCrossEdge = maxSegments;
    out.indexCapacityHint = octree_topology::indexCapacityFromSegments(segmentCount);

    // Vertex generation: assign one dense slot to every surface node in the
    // flattened table (the only nodes that can become triangle corners). The
    // slot is packed into flags bits 1..31 (bit 0 stays the leaf flag) and
    // decoded by the shader as (flags >> 1) - 1. Table order makes the pool's
    // vertex order deterministic; the count is the dense slot address space.
    // The shader writes only the slots a triangle actually references, so the
    // published extent (highest referenced slot + 1) can be smaller.
    uint32_t vertexSlot = 0;
    for (OctreeGpuNode& n : out.nodes) {
        if (n.type == static_cast<int32_t>(SpaceType::Surface)) {
            n.flags |= static_cast<int32_t>((vertexSlot + 1u) << 1);
            ++vertexSlot;
        }
    }
    out.activeVertexCount = vertexSlot;
    return true;
}
}

LocalScene::LocalScene()
    : threadPool(std::thread::hardware_concurrency()) {
    layers_.reserve(3);
    SceneLayer opaque;
    opaque.name = "Opaque";
    opaque.renderer = LayerRendererType::Solid;
    opaque.enabled = true;
    opaque.octree = std::make_shared<Octree>(BoundingCube(glm::vec3(0.0f), 30.0f), glm::pow(2, 9));
    layers_.push_back(std::move(opaque));
    SceneLayer transp;
    transp.name = "Transparent";
    transp.renderer = LayerRendererType::Water;
    transp.enabled = true;
    transp.octree = std::make_shared<Octree>(BoundingCube(glm::vec3(0.0f), 30.0f), glm::pow(2, 9));
    layers_.push_back(std::move(transp));
    // Video layer: geometry using MaterialKind::Video materials. It is drawn
    // after solid and water so per-material blend modes can composite against
    // the finished scene color.
    SceneLayer video;
    video.name = "Video";
    video.renderer = LayerRendererType::Video;
    video.enabled = true;
    video.octree = std::make_shared<Octree>(BoundingCube(glm::vec3(0.0f), 30.0f), glm::pow(2, 9));
    layers_.push_back(std::move(video));
}

LocalScene::~LocalScene() = default;

void LocalScene::stopPools() {
    threadPool.stop();
    std::lock_guard<std::mutex> lock(layersMutex_);
    for (auto& l : layers_) {
        if (l.octree) l.octree->threadPool.stop();
    }
}

size_t LocalScene::layerCount() const {
    std::lock_guard<std::mutex> lock(layersMutex_);
    return layers_.size();
}

std::string LocalScene::layerName(Layer layer) const {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return {};
    return layers_[static_cast<size_t>(layer)].name;
}

void LocalScene::setLayerName(Layer layer, const std::string& name) {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return;
    layers_[static_cast<size_t>(layer)].name = name;
}

LayerRendererType LocalScene::layerRenderer(Layer layer) const {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size())
        return (layer == LAYER_TRANSPARENT) ? LayerRendererType::Water : LayerRendererType::Solid;
    return layers_[static_cast<size_t>(layer)].renderer;
}

void LocalScene::setLayerRenderer(Layer layer, LayerRendererType renderer) {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return;
    layers_[static_cast<size_t>(layer)].renderer = renderer;
}

bool LocalScene::layerEnabled(Layer layer) const {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return false;
    return layers_[static_cast<size_t>(layer)].enabled;
}

void LocalScene::setLayerEnabled(Layer layer, bool enabled) {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return;
    layers_[static_cast<size_t>(layer)].enabled = enabled;
}

Layer LocalScene::addLayer(const std::string& name, LayerRendererType renderer) {
    std::lock_guard<std::mutex> lock(layersMutex_);
    SceneLayer l;
    l.name = name.empty() ? ("Layer " + std::to_string(layers_.size())) : name;
    l.renderer = renderer;
    l.enabled = true;
    // New layers share the reference root lattice so the GPU rung gate stays
    // aligned across layers.
    glm::vec3 rootMin(0.0f);
    float rootLen = 30.0f;
    float chunkSize = glm::pow(2, 9);
    if (!layers_.empty() && layers_[0].octree) {
        rootMin = layers_[0].octree->getMin();
        rootLen = layers_[0].octree->getLengthX();
        chunkSize = layers_[0].octree->chunkSize;
    }
    l.octree = std::make_shared<Octree>(BoundingCube(rootMin, rootLen), chunkSize);
    layers_.push_back(std::move(l));
    return static_cast<Layer>(layers_.size() - 1);
}

bool LocalScene::removeLayer(Layer layer) {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return false;
    if (layers_.size() <= 1) return false; // keep at least one layer
    layers_.erase(layers_.begin() + layer);
    return true;
}

Octree* LocalScene::getLayerOctree(Layer layer) {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return nullptr;
    return layers_[static_cast<size_t>(layer)].octree.get();
}

const Octree* LocalScene::getLayerOctree(Layer layer) const {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return nullptr;
    return layers_[static_cast<size_t>(layer)].octree.get();
}

Octree* LocalScene::layerOctreeLocked(Layer layer) const {
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return nullptr;
    return layers_[static_cast<size_t>(layer)].octree.get();
}

void LocalScene::resetAllLayers() {
    std::lock_guard<std::mutex> lock(layersMutex_);
    for (auto& l : layers_) {
        if (l.octree) l.octree->reset();
    }
    std::lock_guard<std::mutex> elock(emittedMutex_);
    emittedVersion_.clear();
}


// ── GPU-resident chunk topology ─────────────────────────────────────────────
// Flattens this chunk's SDF neighborhood for the compute pass. Returns
// UpToDate when the node's current revision is already live on the GPU (the
// walk would produce the same mesh), Unavailable when the chunk cannot be
// flattened (no octree/stale parameters) so the caller falls back to the CPU
// walker.
Scene::ChunkTopologyStatus LocalScene::buildChunkTopology(Layer layer, OctreeNodeData& data,
                                                          uint8_t chunkLod, uint32_t version,
                                                          OctreeTopologyInput& out) {
    std::shared_ptr<Octree> tree;
    {
        std::lock_guard<std::mutex> lock(layersMutex_);
        if (layer < 0 || static_cast<size_t>(layer) >= layers_.size())
            return ChunkTopologyStatus::Unavailable;
        tree = layers_[static_cast<size_t>(layer)].octree;
    }
    if (!tree || data.node == nullptr || chunkLod < 1) return ChunkTopologyStatus::Unavailable;

    {
        std::lock_guard<std::mutex> lock(emittedMutex_);
        auto it = emittedVersion_.find(reinterpret_cast<uintptr_t>(data.node));
        if (it != emittedVersion_.end() && it->second == version) return ChunkTopologyStatus::UpToDate;
    }

    // Root-consistent flattening reads the tree structure (ancestor chain,
    // children blocks), so it runs under the shared lock; the GPU pass itself
    // only touches the flattened copy.
    std::shared_lock<std::shared_mutex> treeLock(tree->treeMutex);
    if (!buildGpuTopologyInput(*tree, data.node, data.cube, static_cast<int>(chunkLod), out))
        return ChunkTopologyStatus::Unavailable;
    return ChunkTopologyStatus::Ready;
}

float LocalScene::lodRootSize(Layer layer) const {
    std::lock_guard<std::mutex> lock(layersMutex_);
    const Octree* tree = layerOctreeLocked(layer);
    if (!tree) return 0.0f;
    return tree->getLengthX();
}

void LocalScene::commitChunkTopology(Layer layer, uintptr_t emittingNodeId, uint32_t version) {
    (void)layer;
    std::lock_guard<std::mutex> lock(emittedMutex_);
    emittedVersion_[emittingNodeId] = version;
}

// World-SDF query table for the GPU droplet/physics path. Walks the subtree(s)
// intersecting the region with the same root-consistent cube replay the
// topology flattening uses (cube.getChild(i) from the tree root), so a query
// interpolates the identical corner values the mesh walker triangulates.
bool LocalScene::buildSdfRegionTable(Layer layer, const glm::vec3& min, const glm::vec3& max,
                                     std::vector<OctreeGpuNode>& out, size_t maxNodes) {
    std::shared_ptr<Octree> tree;
    {
        std::lock_guard<std::mutex> lock(layersMutex_);
        if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return false;
        tree = layers_[static_cast<size_t>(layer)].octree;
    }
    if (!tree || tree->root == nullptr || !tree->allocator) return false;

    out.clear();
    out.reserve(8192);

    std::shared_lock<std::shared_mutex> treeLock(tree->treeMutex);

    bool truncated = false;
    // Entry 0 is the world root (the shader's descent anchor); a child's index
    // is patched into its parent record after the child has been appended
    // (out may reallocate during the recursion, so the parent is re-indexed by
    // id instead of holding a reference).
    std::function<int32_t(OctreeNode*, const BoundingCube&)> addNode =
        [&](OctreeNode* node, const BoundingCube& cube) -> int32_t {
        if (node == nullptr) return -1;
        const glm::vec3 cMin = cube.getMin();
        const glm::vec3 cMax = cube.getMax();
        if (!cubeIntersectsCube(cMin, cMax, min, max)) return -1;
        if (out.size() >= maxNodes) {
            truncated = true;
            return -1;
        }

        const int32_t id = static_cast<int32_t>(out.size());
        OctreeGpuNode rec{};
        for (int i = 0; i < 8; ++i) rec.sdf[i] = node->sdf[i];
        rec.cubeMin = cMin;
        rec.cubeSize = cube.getLengthX();
        rec.lod = static_cast<int32_t>(node->getLod());
        rec.type = static_cast<int32_t>(node->getType());
        rec.brushIndex = node->brushIndex;
        rec.flags = node->isLeaf() ? 1 : 0;
        rec.hsv = node->hsv;
        for (int i = 0; i < 8; ++i) rec.children[i] = -1;
        out.push_back(rec);

        OctreeNode* children[8] = {};
        node->getChildren(*tree->allocator, children);
        for (int i = 0; i < 8; ++i) {
            if (children[i] == nullptr) continue;
            const int32_t childId = addNode(children[i], cube.getChild(i));
            out[static_cast<size_t>(id)].children[i] = childId;
        }
        return id;
    };

    addNode(tree->root, static_cast<const BoundingCube&>(*tree));

    if (truncated) {
        std::cerr << "[LocalScene] SDF region table truncated (maxNodes=" << maxNodes
                  << ", region " << min.x << "," << min.y << "," << min.z << " .. "
                  << max.x << "," << max.y << "," << max.z << ")\n";
        out.clear();
        return false;
    }
    if (out.empty()) return false;
    std::cout << "[LocalScene] SDF region table: " << out.size() << " nodes (region "
              << (max.x - min.x) << " x " << (max.y - min.y) << " x " << (max.z - min.z) << " m)\n";
    return true;
}

void LocalScene::requestSDFCubes(Layer layer, OctreeNodeData &data, const SdfCubeCallback& callback, ThreadPool* poolOverride) {
    std::shared_ptr<Octree> tree;
    {
        std::lock_guard<std::mutex> lock(layersMutex_);
        if (layer >= 0 && static_cast<size_t>(layer) < layers_.size())
            tree = layers_[static_cast<size_t>(layer)].octree;
    }
    if (!tree) return;
    OctreeThreadContext context;

    // Walk starts AT the chunk node (not the whole tree): only this chunk's
    // subtree is visited, so the whole-tree inSubtree filter is no longer needed.
    {
        std::shared_lock<std::shared_mutex> lock(tree->treeMutex);
        IteratorHandler handler;
        ThreadPool& pool = (poolOverride != nullptr) ? *poolOverride : tree->threadPool;
        handler.iterateMultiThreaded(*tree, data, pool,
        [tree, &data, &context, &callback](const Octree &treeRef, OctreeNodeData &params) {
            if (params.node->getType() != SpaceType::Surface) {
                return true;  // descend through non-surface cells toward the chunk
            }

            // SDF debug cubes live at lod==1 (the same rung the solid ladder uses).
            // Emit every lod==1 surface node; the lod==1 filter above selects exactly
            // the right cells within this chunk's subtree.
            if (params.node->getLod() == 1u) {
                std::array<float, 8> sdf;
                for (size_t i = 0; i < 8; ++i) sdf[i] = params.node->sdf[i];
                callback(params.cube, sdf, 1u, params.node->version,
                         reinterpret_cast<uintptr_t>(params.node),
                         static_cast<uint32_t>(params.node->brushIndex));
            }
            // Keep descending so finer SDF nodes (also at lod==1 in deeper chunks)
            // are visited; the lod==1 filter above selects exactly the right cells.
            return params.node->getLod() > 1u;
        },
        [](const Octree &treeRef, OctreeNodeData &params, uint8_t order[8]) {
            for (int i = 0; i < 8; ++i) order[i] = i;
        },
        [tree, &data, &context](const Octree &treeRef, OctreeNodeData &params) {
            return params.node ? params.node->chunkLod > 0 : false;
        }
        );
    }
}

void LocalScene::requestBoundingBoxes(Layer layer, OctreeNodeData &data, const BBoxCallback& callback, ThreadPool* poolOverride) {
    std::shared_ptr<Octree> tree;
    {
        std::lock_guard<std::mutex> lock(layersMutex_);
        if (layer >= 0 && static_cast<size_t>(layer) < layers_.size())
            tree = layers_[static_cast<size_t>(layer)].octree;
    }
    if (!tree) return;
    OctreeThreadContext context;

    // Walk starts AT the chunk node (not the whole tree): only this chunk's
    // subtree is visited, so the whole-tree inSubtree filter is no longer needed.
    {
        std::shared_lock<std::shared_mutex> lock(tree->treeMutex);
        IteratorHandler handler;
        ThreadPool& pool = (poolOverride != nullptr) ? *poolOverride : tree->threadPool;
        handler.iterateMultiThreaded(*tree, data, pool,
        [tree, &data, &context, &callback](const Octree &treeRef, OctreeNodeData &params) {
            if (params.node->getType() != SpaceType::Surface) {
                return true;  // descend through non-surface cells toward the chunk
            }

            // Emit every surface node whose ladder level matches the CHUNK's LoD
            // (node.lod == chunk.chunkLod), so the overlay shows all node boxes at
            // the chunk's current resolution instead of a single chunk-sized box.
            if (params.node->getLod() == data.node->getChunkLod()) {
                callback(params.cube);
            }
            // Keep descending so finer nodes (also at lod==chunkLod deeper in) are
            // visited; the equality filter above selects exactly the right cells.
            return true;
        },
        [](const Octree &treeRef, OctreeNodeData &params, uint8_t order[8]) {
            for (int i = 0; i < 8; ++i) order[i] = i;
        },
        [tree, &data, &context](const Octree &treeRef, OctreeNodeData &params) {
            return params.node ? params.node->chunkLod > 0 : false;
        }
        );
    }
}

bool LocalScene::isNodeUpToDate(Layer layer, OctreeNodeData &data, uint version) {
    (void)layer;
    return data.node->version >= version;
}

void LocalScene::noteDeletedNode(uintptr_t nodeId) {
    std::lock_guard<std::mutex> lock(emittedMutex_);
    emittedVersion_.erase(nodeId);
}

int LocalScene::maxChunkLod(Layer layer, float minSize) const {    // The number of LoD levels a chunk can hold above its tessellation
    // frontier before reaching the chunk-size boundary, clamped to the
    // ladder the tree actually provides: the root carries the highest
    // chunkLod, and its mesh is the far-distance fallback, so levels beyond
    // it are never drawn. The root's chunkLod is stored +1 shifted (uint8,
    // 0 = unset), so decode it back to the 0-based level count here.
    std::lock_guard<std::mutex> lock(layersMutex_);
    const Octree* tree = layerOctreeLocked(layer);
    if (!tree) return 0;
    int rootChunkLod = -1;
    if (tree->root) {
        const int stored = tree->root->getChunkLod();
        rootChunkLod = stored > 0 ? stored - 1 : -1;
    }
    return std::max(0, std::min(tree->heightRootToChunk(0, minSize), rootChunkLod));
}

glm::vec3 LocalScene::lodRootMin(Layer layer) const {
    std::lock_guard<std::mutex> lock(layersMutex_);
    const Octree* tree = layerOctreeLocked(layer);
    if (!tree) return glm::vec3(0.0f);
    return tree->getMin();
}

void LocalScene::loadScene(SceneLoaderCallback& callback, std::vector<Octree::OctreeNodeDataHandler> updateHandlers, std::vector<Octree::OctreeNodeDataHandler> deleteHandlers) {
    std::cout << "LocalScene::loadScene() layers=" << layerCount() << std::endl;
    auto startTime = std::chrono::steady_clock::now();
    std::vector<Octree*> octrees;
    {
        std::lock_guard<std::mutex> lock(layersMutex_);
        octrees.reserve(layers_.size());
        for (auto& l : layers_) octrees.push_back(l.octree.get());
    }
    // Pad handler vectors to the layer count so the loader can index freely.
    if (updateHandlers.size() < octrees.size()) updateHandlers.resize(octrees.size());
    if (deleteHandlers.size() < octrees.size()) deleteHandlers.resize(octrees.size());
    callback.loadScene(octrees, updateHandlers, deleteHandlers);
    auto endTime = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(endTime - startTime).count();
    std::cout << "LocalScene::loadScene Ok! " << std::to_string(elapsed) << "s"  << std::endl;
}

void LocalScene::save(const std::string& filePath, const Settings* settings) {
    std::filesystem::path outPath(filePath);
    if (outPath.has_parent_path()) {
        std::filesystem::create_directories(outPath.parent_path());
    }

    std::ofstream file(filePath, std::ios::binary);
    if (!file) {
        std::cerr << "LocalScene::save() Error opening file: " << filePath << std::endl;
        return;
    }

    std::ostringstream raw;
    SceneBundleHeader header = {};
    std::memcpy(header.magic, kSceneBundleMagic, sizeof(header.magic));
    header.version = kSceneBundleVersion;
    header.hasSettings = settings ? 1u : 0u;
    raw.write(reinterpret_cast<const char*>(&header), sizeof(header));

    uint32_t layerCount = 0;
    std::vector<std::string> names;
    std::vector<uint8_t> renderers;
    std::vector<uint8_t> enabled;
    std::vector<Octree*> trees;
    {
        std::lock_guard<std::mutex> lock(layersMutex_);
        layerCount = static_cast<uint32_t>(layers_.size());
        for (auto& l : layers_) {
            names.push_back(l.name);
            renderers.push_back(static_cast<uint8_t>(l.renderer));
            enabled.push_back(l.enabled ? 1u : 0u);
            trees.push_back(l.octree.get());
        }
    }
    raw.write(reinterpret_cast<const char*>(&layerCount), sizeof(layerCount));
    for (uint32_t i = 0; i < layerCount; ++i) {
        uint32_t nlen = static_cast<uint32_t>(names[i].size());
        raw.write(reinterpret_cast<const char*>(&nlen), sizeof(nlen));
        if (nlen > 0) raw.write(names[i].data(), nlen);
        raw.write(reinterpret_cast<const char*>(&renderers[i]), sizeof(uint8_t));
        raw.write(reinterpret_cast<const char*>(&enabled[i]), sizeof(uint8_t));
        OctreeFile saver(trees[i], names[i].empty() ? ("layer" + std::to_string(i)) : names[i]);
        saver.writeToStream(raw);
    }

    if (settings) {
        raw.write(reinterpret_cast<const char*>(settings), sizeof(Settings));
    }

    std::istringstream input(raw.str());
    gzipCompressToOfstream(input, file);
    file.close();

    std::cout << "LocalScene::save('" << filePath << "') Ok! layers=" << layerCount << std::endl;
}

void LocalScene::load(const std::string& filePath, Settings* settings) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        std::cerr << "LocalScene::load() Error opening file: " << filePath << std::endl;
        return;
    }

    std::stringstream raw = gzipDecompressFromIfstream(file);

    SceneBundleHeader header = {};
    raw.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!raw || std::memcmp(header.magic, kSceneBundleMagic, sizeof(header.magic)) != 0) {
        std::cerr << "LocalScene::load() Invalid scene bundle: " << filePath << std::endl;
        return;
    }
    if (header.version != kSceneBundleVersionV1 && header.version != kSceneBundleVersionV2) {
        std::cerr << "LocalScene::load() Unsupported bundle version " << header.version << " in " << filePath << std::endl;
        return;
    }

    if (header.version == kSceneBundleVersionV1) {
        // Legacy two-layer bundle: opaque + transparent, no layer metadata.
        std::lock_guard<std::mutex> lock(layersMutex_);
        while (layers_.size() < 2) {
            SceneLayer l;
            l.name = "Layer " + std::to_string(layers_.size());
            l.renderer = LayerRendererType::Solid;
            l.octree = std::make_shared<Octree>(BoundingCube(glm::vec3(0.0f), 30.0f), glm::pow(2, 9));
            layers_.push_back(std::move(l));
        }
        OctreeFile opaqueLoader(layers_[0].octree.get(), "opaque");
        OctreeFile transparentLoader(layers_[1].octree.get(), "transparent");
        opaqueLoader.readFromStream(raw);
        transparentLoader.readFromStream(raw);
    } else {
        uint32_t layerCount = 0;
        raw.read(reinterpret_cast<char*>(&layerCount), sizeof(layerCount));
        if (!raw || layerCount == 0 || layerCount > 64) {
            std::cerr << "LocalScene::load() Bad layer count " << layerCount << std::endl;
            return;
        }
        std::lock_guard<std::mutex> lock(layersMutex_);
        layers_.clear();
        for (uint32_t i = 0; i < layerCount; ++i) {
            uint32_t nlen = 0;
            raw.read(reinterpret_cast<char*>(&nlen), sizeof(nlen));
            std::string name;
            if (nlen > 0 && nlen < 256) {
                name.resize(nlen);
                raw.read(name.data(), nlen);
            }
            uint8_t rend = 0, en = 1;
            raw.read(reinterpret_cast<char*>(&rend), sizeof(uint8_t));
            raw.read(reinterpret_cast<char*>(&en), sizeof(uint8_t));
            SceneLayer l;
            l.name = name.empty() ? ("Layer " + std::to_string(i)) : name;
            l.renderer = (rend == 1) ? LayerRendererType::Water : LayerRendererType::Solid;
            l.enabled = (en != 0);
            l.octree = std::make_shared<Octree>();
            OctreeFile loader(l.octree.get(), l.name);
            loader.readFromStream(raw);
            layers_.push_back(std::move(l));
        }
    }

    if (header.hasSettings != 0u) {
        Settings loadedSettings = {};
        raw.read(reinterpret_cast<char*>(&loadedSettings), sizeof(Settings));
        if (raw && settings) {
            *settings = loadedSettings;
        }
    }

    file.close();
    std::cout << "LocalScene::load('" << filePath << "') Ok! layers=" << layerCount() << std::endl;
}

static void notifyChunkNodes(OctreeNode* node, const BoundingCube& cube, uint level,
                             OctreeAllocator& allocator, Octree::OctreeNodeDataHandler updateHandler, Octree::OctreeNodeDataHandler deleteHandler) {
    if (!node) return;
    if (node->isChunk()) {
        updateHandler(OctreeNodeData(level, node, cube, nullptr));
        return;
    }
    OctreeNode* children[8] = {};
    node->getChildren(allocator, children);
    for (int i = 0; i < 8; ++i) {
        if (children[i])
            notifyChunkNodes(children[i], cube.getChild(i), level + 1, allocator, updateHandler, deleteHandler);
    }
    (void)deleteHandler;
}

void LocalScene::load(const std::string& filePath, std::vector<Octree::OctreeNodeDataHandler> updateHandlers, std::vector<Octree::OctreeNodeDataHandler> deleteHandlers, Settings* settings) {
    load(filePath, settings);
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (updateHandlers.size() < layers_.size()) updateHandlers.resize(layers_.size());
    if (deleteHandlers.size() < layers_.size()) deleteHandlers.resize(layers_.size());
    for (size_t i = 0; i < layers_.size(); ++i) {
        Octree* tree = layers_[i].octree.get();
        if (tree && tree->root && tree->allocator && updateHandlers[i])
            notifyChunkNodes(tree->root, *tree, 0, *tree->allocator, updateHandlers[i], deleteHandlers[i]);
    }
}

void LocalScene::reemitLayer(Layer layer, const Octree::OctreeNodeDataHandler& updateHandler) {
    if (!updateHandler) return;
    // Hold the octree alive for the whole walk so a concurrent removeLayer
    // cannot free it mid-traversal; take the tree's shared lock so a
    // concurrent brush apply (unique lock) cannot mutate nodes under us.
    std::shared_ptr<Octree> tree;
    {
        std::lock_guard<std::mutex> lock(layersMutex_);
        if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return;
        tree = layers_[static_cast<size_t>(layer)].octree;
    }
    if (!tree || !tree->root || !tree->allocator) return;
    Octree::OctreeNodeDataHandler noop;
    {
        std::shared_lock<std::shared_mutex> treeLock(tree->treeMutex);
        notifyChunkNodes(tree->root, *tree, 0, *tree->allocator, updateHandler, noop);
    }
}

void LocalScene::invalidateTessellationCache(Layer layer) {
    (void)layer;
    // buildChunkTopology skips nodes whose version matches this cache. After the
    // GPU slots for a layer are freed (disable / renderer switch), a re-emit
    // must re-tessellate even though octree versions are unchanged. The map
    // is keyed by node address only, so clear it wholesale; the next dispatch
    // of the re-enabled layer rebuilds exactly its own entries.
    std::lock_guard<std::mutex> lock(emittedMutex_);
    emittedVersion_.clear();
}
