#pragma once

#include "Scene.hpp"
#include "../space/Octree.hpp"
#include "../space/OctreeVisibilityChecker.hpp"
#include "../space/Tesselator.hpp"
#include "../space/Processor.hpp"
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

    void requestModel3D(Layer layer, OctreeNodeData &data, const GeometryCallback& callback) override;
    bool isNodeUpToDate(Layer layer, OctreeNodeData &data, uint version) override;
    void loadScene(SceneLoaderCallback& callback, const OctreeChangeHandler &opaqueLayerChangeHandler, const OctreeChangeHandler &transparentLayerChangeHandler) override;
    void save(const std::string& folderPath);
    void load(const std::string& folderPath);
    void load(const std::string& folderPath, const OctreeChangeHandler& opaqueHandler, const OctreeChangeHandler& transparentHandler);
};