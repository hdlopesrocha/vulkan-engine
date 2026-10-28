#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <array>
#include <functional>
#include <string>
#include <cstdint>
#include "../math/geometry/Geometry.hpp"
#include "../space/Octree.hpp"
#include "../space/OctreeNodeData.hpp"
#include "../services/OctreeTopologyTypes.hpp"

// Dynamic scene layers: each Scene owns a list of Layer objects, and each
// Layer owns its octree + renderer selection (utils/Layer.hpp). Layer ids are
// stable keys for the wire protocol / GPU state arrays; layer POSITIONS are
// the list order (layerAt).
#include "Layer.hpp"

constexpr Layer::Id LAYER_OPAQUE = 0;
constexpr Layer::Id LAYER_TRANSPARENT = 1;

#include "SceneLoaderCallback.hpp"

// Forward declaration: optional pools may be supplied to the debug/SDF walks
// so a caller (e.g. brush editing) does not serialize on the scene pool.
class ThreadPool;

class Scene {

public:
    Scene();
    virtual ~Scene();

    // ── Dynamic layer list ──
    // Positional access (0..layerCount()-1). Null when out of range, so a UI
    // loop can iterate without racing a concurrent remove.
    virtual Layer* layerAt(size_t index) = 0;
    virtual const Layer* layerAt(size_t index) const = 0;
    // Stable-id access (wire protocol, settings, renderer state).
    virtual Layer* findLayer(Layer::Id id) = 0;
    virtual const Layer* findLayer(Layer::Id id) const = 0;
    virtual size_t layerCount() const = 0;
    virtual Layer* addLayer(const std::string& name, LayerRendererType renderer) = 0;
    virtual bool removeLayer(Layer& layer) = 0;
    bool validLayer(Layer::Id id) const;

    virtual void loadScene(SceneLoaderCallback& callback, std::vector<Octree::OctreeNodeDataHandler> updateHandlers, std::vector<Octree::OctreeNodeDataHandler> deleteHandlers) = 0;
    // Re-emit every chunk of one layer into updateHandler (used to repopulate
    // GPU meshes after a layer is re-enabled or its renderer is switched).
    // Only the update side is needed: the GPU slots were freed, so every
    // live chunk must be re-queued for tessellation.
    virtual void reemitLayer(Layer& layer, const Octree::OctreeNodeDataHandler& updateHandler) = 0;
    // Drop tessellation dedup state so a re-emitted layer actually
    // re-tessellates instead of being skipped as up-to-date. Default no-op.
    virtual void invalidateTessellationCache(Layer& layer);

    // ── GPU-resident chunk topology (the only chunk mesh path) ───────────────
    // The host never tessellates: it flattens the octree neighborhood the GPU
    // topology walker needs for one chunk (services/OctreeTopologyCompute) and
    // the compute pass writes the chunk's deduped Cartesian vertices and index
    // stream straight into the renderer's packed pools. Scenes without an
    // octree to flatten cannot render geometry.
    enum class ChunkTopologyStatus {
        Ready,      // `out` holds this revision's neighborhood: run the GPU pass
        UpToDate,   // this revision is already on the GPU: skip the chunk
        Unavailable // this scene cannot flatten its nodes: skip the chunk
    };
    // `chunkLod` is the node's stored (1-based) chunk level: the walk target is
    // exactly this value and the published mesh rung is chunkLod - 1. `version`
    // is the node revision the caller captured (commit passes the same value
    // back, so a concurrent brush edit re-emits instead of being swallowed).
    virtual ChunkTopologyStatus buildChunkTopology(Layer& layer, OctreeNodeData& data,
                                                   uint8_t chunkLod, uint32_t version,
                                                   OctreeTopologyInput& out);
    // Records that `version` of this chunk is live on the GPU, so a later walk
    // of the same revision is skipped (UpToDate).
    virtual void commitChunkTopology(Layer& layer, uintptr_t emittingNodeId, uint32_t version);
    // Flattens every octree node whose cube intersects [min, max] into the same
    // node layout the GPU topology pass consumes (entry 0 = world root, corner
    // samples with INFINITY sentinels, children indices). This is the world-SDF
    // query table for consumers that need distances outside the per-chunk
    // topology path (GPU particle/physics collision); it reuses the exact same
    // node representation, so a query interpolates the identical SDF the mesh
    // walker triangulates. Returns false when the layer has no octree.
    // `maxNodes` caps the table; a region that does not fit is reported false.
    virtual bool buildSdfRegionTable(Layer& layer, const glm::vec3& min, const glm::vec3& max,
                                     std::vector<OctreeGpuNode>& out, size_t maxNodes);
    // Collect SDF debug cubes: the walk emits one callback per node that
    // carries a drawable SDF face, with the node's own cube (band box) so the
    // caller can place a debug cube. Boxes are emitted for nodes at lod==1
    // (the same level the solid ladder uses).
    using SdfCubeCallback = std::function<void(
        const BoundingCube& cube,
        const std::array<float, 8>& sdf,
        uint8_t lod,
        uint version,
        uintptr_t emittingNodeId,
        uint32_t brushIndex
    )>;
    virtual void requestSDFCubes(Layer& layer, OctreeNodeData &data, const SdfCubeCallback& callback, ThreadPool* poolOverride = nullptr);
    // Collect mesh bounding-box debug cubes: walk the octree (subtree rooted at
    // `data`) and emit one callback per surface node whose ladder level equals
    // its chunk LoD (node.lod == node.chunkLod), so the debug overlay shows every
    // node box at the chunk's current resolution instead of a single chunk-sized box.
    using BBoxCallback = std::function<void(const BoundingCube& cube)>;
    virtual void requestBoundingBoxes(Layer& layer, OctreeNodeData &data, const BBoxCallback& callback, ThreadPool* poolOverride = nullptr);
    virtual bool isNodeUpToDate(Layer& layer, OctreeNodeData &data, uint version) = 0;

    // Maximum LoD level a chunk can publish for the given layer (>= 0). The
    // chunk's ladder maxLevel is clamped to this so HeightRootToChunk(N) >= 0
    // always holds — coarse levels never exceed the chunk's own size band.
    virtual int maxChunkLod(const Layer& layer, float minSize) const = 0;

    // Root-lattice origin for the GPU rung gate (must be bit-identical to the
    // tree root min the chunks were tessellated against; per-chunk values
    // mis-align parent cells and cull most rungs).
    virtual glm::vec3 lodRootMin(const Layer& layer) const;
    // Edge length of the layer's octree root cube (0 when there is no tree).
    virtual float lodRootSize(const Layer& layer) const;
};
