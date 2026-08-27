#pragma once

#include "Scene.hpp"
#include "../space/Octree.hpp"
#include "../space/Tesselator.hpp"
#include "../space/InstanceData.hpp"
#include "../utils/Settings.hpp"
#include <unordered_map>
#include <mutex>
#include "OctreeLayer.tpp"

class LocalScene : public Scene {
public:

    Octree opaqueOctree;
    Octree transparentOctree;
    ThreadPool threadPool;

    Octree& getOpaqueOctree();
    const Octree& getOpaqueOctree() const;
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
    void requestSDFCubes(Layer layer, OctreeNodeData &data, const SdfCubeCallback& callback, ThreadPool* poolOverride = nullptr) override;
    void requestBoundingBoxes(Layer layer, OctreeNodeData &data, const BBoxCallback& callback, ThreadPool* poolOverride = nullptr) override;
    bool isNodeUpToDate(Layer layer, OctreeNodeData &data, uint version) override;
    int maxChunkLod(Layer layer, float minSize) const override;
    void action(SceneLoaderCallback& callback, Octree::OctreeNodeDataHandler opaqueUpdateHandler, Octree::OctreeNodeDataHandler opaqueDeleteHandler, Octree::OctreeNodeDataHandler transparentUpdateHandler, Octree::OctreeNodeDataHandler transparentDeleteHandler) override;
    void loadScene(SceneLoaderCallback& callback, Octree::OctreeNodeDataHandler opaqueUpdateHandler, Octree::OctreeNodeDataHandler opaqueDeleteHandler, Octree::OctreeNodeDataHandler transparentUpdateHandler, Octree::OctreeNodeDataHandler transparentDeleteHandler) override;
    void save(const std::string& filePath, const Settings* settings = nullptr);
    void load(const std::string& filePath, Settings* settings = nullptr);
    void load(const std::string& filePath, Octree::OctreeNodeDataHandler opaqueUpdateHandler, Octree::OctreeNodeDataHandler opaqueDeleteHandler, Octree::OctreeNodeDataHandler transparentUpdateHandler, Octree::OctreeNodeDataHandler transparentDeleteHandler, Settings* settings = nullptr);

    // Forget a node's tessellation record when it is deleted (the node memory
    // may be reused; the stale entry could otherwise suppress a re-tessellation
    // of the new occupant).
    void noteDeletedNode(uintptr_t nodeId);

private:
    // Last-tessellated version per emitting node. requestModel3D's walk emits
    // every cell on the root path for every added node; without this cache
    // each cell is re-tessellated (and re-uploaded) once per added descendant
    // during load. Node versions only bump in the change walk (edits), so a
    // matching version means the mesh is still current.
    std::mutex emittedMutex_;
    std::unordered_map<uintptr_t, uint32_t> emittedVersion_;
};