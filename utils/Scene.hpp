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
using GeometryCallback = std::function<void(const Geometry&)>;

class SceneLoaderCallback {
public:
    SceneLoaderCallback() = default;
    ~SceneLoaderCallback() = default;

    virtual void loadScene(Octree &opaqueLayer, const OctreeChangeHandler& opaqueHandler, Octree &transparentLayer, const OctreeChangeHandler& transparentHandler) = 0;
    virtual void action(Octree &opaqueLayer, const OctreeChangeHandler& opaqueHandler, Octree &transparentLayer, const OctreeChangeHandler& transparentHandler) = 0;
};

class Scene {

public:
    Scene() = default;
    ~Scene() = default;
    virtual void action(SceneLoaderCallback& callback, const OctreeChangeHandler &opaqueLayerChangeHandler, const OctreeChangeHandler &transparentLayerChangeHandler) = 0;
    virtual void loadScene(SceneLoaderCallback& callback, const OctreeChangeHandler &opaqueLayerChangeHandler, const OctreeChangeHandler &transparentLayerChangeHandler) = 0;
    virtual void requestModel3D(Layer layer, OctreeNodeData &data, const GeometryCallback& callback, ThreadPool* poolOverride = nullptr, int lod = -1, float* outCellSize = nullptr) = 0;
    virtual bool isNodeUpToDate(Layer layer, OctreeNodeData &data, uint version) = 0;

    // Maximum LoD level a chunk can publish for the given layer (>= 0). The
    // chunk's ladder maxLevel is clamped to this so HeightRootToChunk(N) >= 0
    // always holds — coarse levels never exceed the chunk's own size band.
    virtual int maxChunkLod(Layer layer, float minSize) const = 0;
};