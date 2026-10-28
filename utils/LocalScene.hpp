#pragma once

#include "Scene.hpp"
#include "../space/Octree.hpp"
#include "types/Settings.hpp"
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

class LocalScene : public Scene {
public:
    ThreadPool threadPool;

public:
    LocalScene();
    ~LocalScene();

    // ── Scene dynamic-layer interface ──
    // Layers are heap-allocated and shared: a worker task keeps its octree
    // alive after the layer is removed from the list.
    Layer* layerAt(size_t index) override;
    const Layer* layerAt(size_t index) const override;
    Layer* findLayer(Layer::Id id) override;
    const Layer* findLayer(Layer::Id id) const override;
    size_t layerCount() const override;
    Layer* addLayer(const std::string& name, LayerRendererType renderer) override;
    bool removeLayer(Layer& layer) override;

    // Explicitly stop all thread pools (LocalScene + every layer Octree).
    // Must be called before any objects captured by enqueued tasks are destroyed.
    void stopPools();

    ChunkTopologyStatus buildChunkTopology(Layer& layer, OctreeNodeData& data, uint8_t chunkLod,
                                           uint32_t version, OctreeTopologyInput& out) override;
    void commitChunkTopology(Layer& layer, uintptr_t emittingNodeId, uint32_t version) override;
    bool buildSdfRegionTable(Layer& layer, const glm::vec3& min, const glm::vec3& max,
                             std::vector<OctreeGpuNode>& out, size_t maxNodes) override;
    void requestSDFCubes(Layer& layer, OctreeNodeData &data, const SdfCubeCallback& callback, ThreadPool* poolOverride = nullptr) override;
    void requestBoundingBoxes(Layer& layer, OctreeNodeData &data, const BBoxCallback& callback, ThreadPool* poolOverride = nullptr) override;
    bool isNodeUpToDate(Layer& layer, OctreeNodeData &data, uint version) override;
    int maxChunkLod(const Layer& layer, float minSize) const override;
    glm::vec3 lodRootMin(const Layer& layer) const override;
    float lodRootSize(const Layer& layer) const override;
    void loadScene(SceneLoaderCallback& callback, std::vector<Octree::OctreeNodeDataHandler> updateHandlers, std::vector<Octree::OctreeNodeDataHandler> deleteHandlers) override;
    void reemitLayer(Layer& layer, const Octree::OctreeNodeDataHandler& updateHandler) override;
    void invalidateTessellationCache(Layer& layer) override;
    void save(const std::string& filePath, const Settings* settings = nullptr);
    void load(const std::string& filePath, Settings* settings = nullptr);
    void load(const std::string& filePath, std::vector<Octree::OctreeNodeDataHandler> updateHandlers, std::vector<Octree::OctreeNodeDataHandler> deleteHandlers, Settings* settings = nullptr);

    // Reset every layer octree (used by MyApp::resetSceneState).
    void resetAllLayers();

    // Forget a node's tessellation record when it is deleted (the node memory
    // may be reused; the stale entry could otherwise suppress a re-tessellation
    // of the new occupant).
    void noteDeletedNode(uintptr_t nodeId);

private:
    std::vector<std::shared_ptr<Layer>> layers_;
    // Stable ids, never reused (the wire protocol and GPU layer-state arrays
    // key on them, and a removed layer's pending work must stay distinct from
    // a new layer appended at the same list position).
    Layer::Id nextLayerId_ = 0;
    // Removed layers stay alive until the next controlled reset: a worker
    // thread may still hold the Layer* it resolved for an in-flight chunk
    // build, and destroying the object under it would be a use-after-free.
    // MyApp joins its tessellation threads before resetAllLayers().
    std::vector<std::shared_ptr<Layer>> retiredLayers_;
    mutable std::mutex layersMutex_;
    // Last-tessellated version per emitting node. buildChunkTopology skips a
    // revision that is already resident on the GPU; node versions only bump in
    // the change walk (edits), so a matching version means the mesh is current.
    std::mutex emittedMutex_;
    std::unordered_map<uintptr_t, uint32_t> emittedVersion_;
};
