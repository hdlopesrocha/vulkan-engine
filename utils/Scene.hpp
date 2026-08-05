#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <functional>
#include "../math/Geometry.hpp"
#include "../space/Octree.hpp"
#include "../space/OctreeNodeData.hpp"

enum Layer {
    LAYER_OPAQUE = 0,
    LAYER_TRANSPARENT = 1,
    LAYER_UI = 2,
    LAYER_COUNT = 3
};


// Forward declaration: an optional thread pool may be supplied to
// requestModel3D so a caller (e.g. brush editing) can tessellate on a
// dedicated pool instead of the scene's shared generation pool.
class ThreadPool;

// Visible nodes are reported via a callback lambda taking a NodeID and its version
using VisibleNodeCallback = std::function<void(std::vector<OctreeNodeData>&)>;
// One tessellation walk returns the chunk's whole LoD ladder: lods[i] is the
// level-i mesh (0 = full-detail frontier, up to the chunk root's coarse cell).

class SceneLoaderCallback {
public:
    SceneLoaderCallback() = default;
    ~SceneLoaderCallback() = default;


    virtual void action(
        Octree &opaqueLayer, 
        const Octree::OctreeNodeDataHandler& opaqueUpdateHandler, 
        const Octree::OctreeNodeDataHandler& opaqueDeleteHandler, 
        Octree &transparentLayer, 
        const Octree::OctreeNodeDataHandler& transparentUpdateHandler, 
        const Octree::OctreeNodeDataHandler& transparentDeleteHandler
    ) = 0;

    virtual void loadScene(
        Octree &opaqueLayer, 
        Octree::OctreeNodeDataHandler &opaqueUpdateHandler,
        Octree::OctreeNodeDataHandler &opaqueDeleteHandler,
        Octree &transparentLayer,
        Octree::OctreeNodeDataHandler &transparentUpdateHandler,
        Octree::OctreeNodeDataHandler &transparentDeleteHandler
    ) = 0;

   
};

class Scene {

public:
    Scene() = default;
    ~Scene() = default;
    virtual void action(SceneLoaderCallback& callback, const Octree::OctreeNodeDataHandler opaqueUpdateHandler, const Octree::OctreeNodeDataHandler opaqueDeleteHandler, const Octree::OctreeNodeDataHandler transparentUpdateHandler, const Octree::OctreeNodeDataHandler transparentDeleteHandler) = 0;
    virtual void loadScene(SceneLoaderCallback& callback, const Octree::OctreeNodeDataHandler opaqueUpdateHandler, const Octree::OctreeNodeDataHandler opaqueDeleteHandler, const Octree::OctreeNodeDataHandler transparentUpdateHandler, const Octree::OctreeNodeDataHandler transparentDeleteHandler) = 0;
    virtual void requestModel3D(Layer layer, OctreeNodeData &data, const GeometryLodCallback& callback, ThreadPool* poolOverride = nullptr) = 0;
    virtual bool isNodeUpToDate(Layer layer, OctreeNodeData &data, uint version) = 0;

    // Maximum LoD level a chunk can publish for the given layer (>= 0). The
    // chunk's ladder maxLevel is clamped to this so HeightRootToChunk(N) >= 0
    // always holds — coarse levels never exceed the chunk's own size band.
    virtual int maxChunkLod(Layer layer, float minSize) const = 0;
};