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

typedef uintptr_t NodeID;

// Visible nodes are reported via a callback lambda taking a NodeID and its version
using VisibleNodeCallback = std::function<void(std::vector<OctreeNodeData>&)>;
using GeometryCallback = std::function<void(const Geometry&)>;

class SceneLoaderCallback {
public:
    SceneLoaderCallback() = default;
    ~SceneLoaderCallback() = default;

    virtual void loadScene(Octree &opaqueLayer, OctreeChangeHandler& opaqueHandler, Octree &transparentLayer, OctreeChangeHandler& transparentHandler) = 0;
};

class Scene {

public:
    Scene() = default;
    ~Scene() = default;
    virtual void loadScene(SceneLoaderCallback& callback) = 0;
    virtual void requestVisibleNodes(Layer layer, glm::mat4 viewMatrix, const VisibleNodeCallback& callback) = 0;
    virtual void requestModel3D(Layer layer, OctreeNodeData &data, const GeometryCallback& callback) = 0;
    virtual bool isNodeUpToDate(Layer layer, OctreeNodeData &data, uint version) = 0;
};