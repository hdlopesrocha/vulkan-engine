#pragma once

#include "Scene.hpp"
#include "../space/Octree.hpp"
#include "types/Settings.hpp"
#include <unordered_map>
#include <mutex>
#include <memory>

class LocalScene : public Scene {
public:
    // One dynamic scene layer: a named octree plus its renderer selection.
    // Octrees are heap-allocated (shared) so layer add/remove never destroys
    // an Octree while a worker thread still holds it (request* copies the
    // shared_ptr under lock and walks without holding the layer mutex).
    struct SceneLayer {
        std::string name = "Layer";
        LayerRendererType renderer = LayerRendererType::Solid;
        bool enabled = true;
        std::shared_ptr<Octree> octree;
    };

    ThreadPool threadPool;

public:
    LocalScene();
    ~LocalScene();

    // ── Scene dynamic-layer interface ──
    size_t layerCount() const override;
    std::string layerName(Layer layer) const override;
    void setLayerName(Layer layer, const std::string& name) override;
    LayerRendererType layerRenderer(Layer layer) const override;
    void setLayerRenderer(Layer layer, LayerRendererType renderer) override;
    bool layerEnabled(Layer layer) const override;
    void setLayerEnabled(Layer layer, bool enabled) override;
    Layer addLayer(const std::string& name, LayerRendererType renderer) override;
    bool removeLayer(Layer layer) override;
    Octree* getLayerOctree(Layer layer) override;
    const Octree* getLayerOctree(Layer layer) const override;

    // Explicitly stop all thread pools (LocalScene + every layer Octree).
    // Must be called before any objects captured by enqueued tasks are destroyed.
    void stopPools();

    ChunkTopologyStatus buildChunkTopology(Layer layer, OctreeNodeData& data, uint8_t chunkLod,
                                           uint32_t version, OctreeTopologyInput& out) override;
    void commitChunkTopology(Layer layer, uintptr_t emittingNodeId, uint32_t version) override;
    bool buildSdfRegionTable(Layer layer, const glm::vec3& min, const glm::vec3& max,
                             std::vector<OctreeGpuNode>& out, size_t maxNodes) override;
    void requestSDFCubes(Layer layer, OctreeNodeData &data, const SdfCubeCallback& callback, ThreadPool* poolOverride = nullptr) override;
    void requestBoundingBoxes(Layer layer, OctreeNodeData &data, const BBoxCallback& callback, ThreadPool* poolOverride = nullptr) override;
    bool isNodeUpToDate(Layer layer, OctreeNodeData &data, uint version) override;
    int maxChunkLod(Layer layer, float minSize) const override;
    glm::vec3 lodRootMin(Layer layer) const override;
    float lodRootSize(Layer layer) const override;
    void loadScene(SceneLoaderCallback& callback, std::vector<Octree::OctreeNodeDataHandler> updateHandlers, std::vector<Octree::OctreeNodeDataHandler> deleteHandlers) override;
    void reemitLayer(Layer layer, const Octree::OctreeNodeDataHandler& updateHandler) override;
    void invalidateTessellationCache(Layer layer) override;
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
    Octree* layerOctreeLocked(Layer layer) const;
    std::vector<SceneLayer> layers_;
    mutable std::mutex layersMutex_;
    // Last-tessellated version per emitting node. buildChunkTopology skips a
    // revision that is already resident on the GPU; node versions only bump in
    // the change walk (edits), so a matching version means the mesh is current.
    std::mutex emittedMutex_;
    std::unordered_map<uintptr_t, uint32_t> emittedVersion_;
};
