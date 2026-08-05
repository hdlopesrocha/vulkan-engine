#pragma once

#include "Scene.hpp"
#include "../space/Octree.hpp"
#include "../space/Tesselator.hpp"
#include "../space/InstanceData.hpp"
#include "../utils/Settings.hpp"
#include <unordered_map>
#include "OctreeLayer.tpp"

class LocalScene : public Scene {
    Octree opaqueOctree;
public:
    Octree& getOpaqueOctree();
    const Octree& getOpaqueOctree() const;
    Octree transparentOctree;
    ThreadPool threadPool;

public:
    // Instance/visibility layers and change handlers (owned by LocalScene)
    OctreeLayer<InstanceData> opaqueLayerInfo;
    OctreeLayer<InstanceData> transparentLayerInfo;
    
    LocalScene();
    ~LocalScene();

    // Explicitly stop all thread pools (LocalScene + both Octrees).
    // Must be called before any objects captured by enqueued tasks are destroyed.
    void stopPools();

    void requestModel3D(Layer layer, OctreeNodeData &data, const GeometryLodCallback& callback, ThreadPool* poolOverride = nullptr) override;
    bool isNodeUpToDate(Layer layer, OctreeNodeData &data, uint version) override;
    int maxChunkLod(Layer layer, float minSize) const override;
    void action(SceneLoaderCallback& callback, Octree::OctreeNodeDataHandler opaqueUpdateHandler, Octree::OctreeNodeDataHandler opaqueDeleteHandler, Octree::OctreeNodeDataHandler transparentUpdateHandler, Octree::OctreeNodeDataHandler transparentDeleteHandler) override;
    void loadScene(SceneLoaderCallback& callback, Octree::OctreeNodeDataHandler opaqueUpdateHandler, Octree::OctreeNodeDataHandler opaqueDeleteHandler, Octree::OctreeNodeDataHandler transparentUpdateHandler, Octree::OctreeNodeDataHandler transparentDeleteHandler) override;
    void save(const std::string& filePath, const Settings* settings = nullptr);
    void load(const std::string& filePath, Settings* settings = nullptr);
    void load(const std::string& filePath, Octree::OctreeNodeDataHandler opaqueUpdateHandler, Octree::OctreeNodeDataHandler opaqueDeleteHandler, Octree::OctreeNodeDataHandler transparentUpdateHandler, Octree::OctreeNodeDataHandler transparentDeleteHandler, Settings* settings = nullptr);
};