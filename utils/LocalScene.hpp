#pragma once

#include "Scene.hpp"
#include "../space/Octree.hpp"
#include "../space/OctreeVisibilityChecker.hpp"
#include "../space/Tesselator.hpp"
#include "../space/Processor.hpp"
#include "../utils/Settings.hpp"
#include <unordered_map>
#include "LiquidSpaceChangeHandler.hpp"
#include "SolidSpaceChangeHandler.hpp"

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

    void requestModel3D(Layer layer, OctreeNodeData &data, const LadderCallback& callback, ThreadPool* poolOverride = nullptr) override;
    bool isNodeUpToDate(Layer layer, OctreeNodeData &data, uint version) override;
    int maxChunkLod(Layer layer, float minSize) const override;
    void action(SceneLoaderCallback& callback, const OctreeChangeHandler &opaqueLayerChangeHandler, const OctreeChangeHandler &transparentLayerChangeHandler) override;
    void loadScene(SceneLoaderCallback& callback, const OctreeChangeHandler &opaqueLayerChangeHandler, const OctreeChangeHandler &transparentLayerChangeHandler) override;
    void save(const std::string& filePath, const Settings* settings = nullptr);
    void load(const std::string& filePath, Settings* settings = nullptr);
    void load(const std::string& filePath, const OctreeChangeHandler& opaqueHandler, const OctreeChangeHandler& transparentHandler, Settings* settings = nullptr);
};