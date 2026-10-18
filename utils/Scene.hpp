#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <array>
#include <functional>
#include <string>
#include <cstdint>
#include "../math/Geometry.hpp"
#include "../space/Octree.hpp"
#include "../space/OctreeNodeData.hpp"
#include "../services/OctreeTopologyTypes.hpp"

// Dynamic scene layers: the Scene owns a list of layers (octrees) instead of
// two fixed members. Layer is the index into that list.
using Layer = int;
constexpr Layer LAYER_OPAQUE = 0;
constexpr Layer LAYER_TRANSPARENT = 1;

// Per-layer renderer selection (edited in the LayersWidget). Solid layers
// feed the SolidRenderer's IndirectRenderer, Water layers feed the
// WaterRenderer's one. Both passes still render once per frame; the mapping
// only decides which GPU pipeline a layer's chunks are published into.
enum class LayerRendererType : uint8_t {
    Solid = 0,
    Water = 1
};

inline const char* layerRendererName(LayerRendererType r) {
    return (r == LayerRendererType::Water) ? "Water" : "Solid";
}


// Forward declaration: an optional thread pool may be supplied to
// requestModel3D so a caller (e.g. brush editing) can tessellate on a
// dedicated pool instead of the scene's shared generation pool.
class ThreadPool;

// One tessellation walk returns the chunk's whole LoD ladder: lods[i] is the
// level-i mesh (0 = full-detail frontier, up to the chunk root's coarse cell).

class SceneLoaderCallback {
public:
    SceneLoaderCallback() = default;
    ~SceneLoaderCallback() = default;

    // Dynamic layers: octrees[i] is scene layer i, with its update/delete
    // handlers at the same index. Implementations should tolerate any size
    // (>= 1): the default procedural loader populates layer 0 (opaque) and
    // layer 1 (transparent) when present and leaves extra layers empty.
    virtual void loadScene(
        std::vector<Octree*>& octrees,
        std::vector<Octree::OctreeNodeDataHandler>& updateHandlers,
        std::vector<Octree::OctreeNodeDataHandler>& deleteHandlers
    ) = 0;

   
};

class Scene {

public:
    Scene() = default;
    virtual ~Scene() = default;

    // ── Dynamic layer list ──
    virtual size_t layerCount() const = 0;
    virtual std::string layerName(Layer layer) const = 0;
    virtual void setLayerName(Layer layer, const std::string& name) = 0;
    virtual LayerRendererType layerRenderer(Layer layer) const = 0;
    virtual void setLayerRenderer(Layer layer, LayerRendererType renderer) = 0;
    virtual bool layerEnabled(Layer layer) const = 0;
    virtual void setLayerEnabled(Layer layer, bool enabled) = 0;
    virtual Layer addLayer(const std::string& name, LayerRendererType renderer) = 0;
    virtual bool removeLayer(Layer layer) = 0;
    virtual Octree* getLayerOctree(Layer layer) = 0;
    virtual const Octree* getLayerOctree(Layer layer) const = 0;
    bool validLayer(Layer layer) const {
        return layer >= 0 && static_cast<size_t>(layer) < layerCount();
    }

    virtual void loadScene(SceneLoaderCallback& callback, std::vector<Octree::OctreeNodeDataHandler> updateHandlers, std::vector<Octree::OctreeNodeDataHandler> deleteHandlers) = 0;
    // Re-emit every chunk of one layer into updateHandler (used to repopulate
    // GPU meshes after a layer is re-enabled or its renderer is switched).
    // Only the update side is needed: the GPU slots were freed, so every
    // live chunk must be re-queued for tessellation.
    virtual void reemitLayer(Layer layer, const Octree::OctreeNodeDataHandler& updateHandler) = 0;
    // Drop tessellation dedup state so a re-emitted layer actually
    // re-tessellates instead of being skipped as up-to-date. Default no-op.
    virtual void invalidateTessellationCache(Layer layer) { (void)layer; }
    virtual void requestModel3D(Layer layer, OctreeNodeData &data, const GeometryLodCallback& callback, ThreadPool* poolOverride = nullptr) = 0;

    // ── GPU-resident chunk topology (app path) ───────────────────────────────
    // The app does not tessellate on the CPU: it flattens the octree
    // neighborhood the GPU topology walker needs for one chunk
    // (services/OctreeTopologyCompute) and the compute pass writes the chunk's
    // deduped vertices and index stream straight into the renderer's packed
    // pools. requestModel3D stays the CPU/server path.
    enum class ChunkTopologyStatus {
        Ready,      // `out` holds this revision's neighborhood: run the GPU pass
        UpToDate,   // this revision is already on the GPU: skip the chunk
        Unavailable // this scene cannot flatten its nodes: use the CPU path
    };
    // `chunkLod` is the node's stored (1-based) chunk level: the walk target is
    // exactly this value and the published mesh rung is chunkLod - 1, matching
    // the CPU path's callback. `version` is the node revision the caller
    // captured (commit passes the same value back, so a concurrent brush edit
    // re-emits instead of being swallowed). LocalScene implements this;
    // RemoteScene has no octree to flatten (its chunks arrive as meshes from
    // the server).
    virtual ChunkTopologyStatus buildChunkTopology(Layer layer, OctreeNodeData& data,
                                                   uint8_t chunkLod, uint32_t version,
                                                   OctreeTopologyInput& out) {
        (void)layer; (void)data; (void)chunkLod; (void)version; (void)out;
        return ChunkTopologyStatus::Unavailable;
    }
    // Records that `version` of this chunk is live on the GPU, so a later walk
    // of the same revision is skipped (UpToDate).
    virtual void commitChunkTopology(Layer layer, uintptr_t emittingNodeId, uint32_t version) {
        (void)layer; (void)emittingNodeId; (void)version;
    }
    // Collect SDF debug cubes the same way requestModel3D collects meshes: the
    // walk emits one callback per node that carries a drawable SDF face, with the
    // node's own cube (band box) so the caller can place a debug cube. Boxes are
    // emitted for nodes at lod==1 (the same level the solid ladder uses).
    using SdfCubeCallback = std::function<void(
        const BoundingCube& cube,
        const std::array<float, 8>& sdf,
        uint8_t lod,
        uint version,
        uintptr_t emittingNodeId,
        uint32_t brushIndex
    )>;
    virtual void requestSDFCubes(Layer layer, OctreeNodeData &data, const SdfCubeCallback& callback, ThreadPool* poolOverride = nullptr) {}
    // Collect mesh bounding-box debug cubes: walk the octree (subtree rooted at
    // `data`) and emit one callback per surface node whose ladder level equals
    // its chunk LoD (node.lod == node.chunkLod), so the debug overlay shows every
    // node box at the chunk's current resolution instead of a single chunk-sized box.
    using BBoxCallback = std::function<void(const BoundingCube& cube)>;
    virtual void requestBoundingBoxes(Layer layer, OctreeNodeData &data, const BBoxCallback& callback, ThreadPool* poolOverride = nullptr) {}
    virtual bool isNodeUpToDate(Layer layer, OctreeNodeData &data, uint version) = 0;

    // Maximum LoD level a chunk can publish for the given layer (>= 0). The
    // chunk's ladder maxLevel is clamped to this so HeightRootToChunk(N) >= 0
    // always holds — coarse levels never exceed the chunk's own size band.
    virtual int maxChunkLod(Layer layer, float minSize) const = 0;

    // Root-lattice origin for the GPU rung gate (must be bit-identical to the
    // tree root min the chunks were tessellated against; per-chunk values
    // mis-align parent cells and cull most rungs).
    virtual glm::vec3 lodRootMin(Layer layer) const { return glm::vec3(0.0f); }
};