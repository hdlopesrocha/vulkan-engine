#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <functional>
#include "../math/Mesh3D.hpp"
#include "../space/Octree.hpp"
#include "../space/OctreeNodeData.hpp"

enum Layer {
    LAYER_OPAQUE = 0,
    LAYER_TRANSPARENT = 1,
    LAYER_UI = 2,
    LAYER_COUNT = 3
};

// Visible nodes are reported via a callback lambda taking an OctreeNodeData reference
using VisibleNodeCallback = std::function<void(const OctreeNodeData&)>;

// Mesh3D results are delivered via a callback lambda taking a Model3D reference
using Model3DCallback = std::function<void(Mesh3D&)>;

class SceneLoaderCallback {
public:
    SceneLoaderCallback() = default;
    ~SceneLoaderCallback() = default;

    virtual void loadScene(Octree &opaqueLayer, Octree &transparentLayer) = 0;
};

class Scene {

public:
    Scene() = default;
    ~Scene() = default;
    virtual void loadScene(SceneLoaderCallback& callback) = 0;
    virtual void requestVisibleNodes(Layer layer, glm::mat4 viewMatrix, const VisibleNodeCallback& callback) = 0;
    virtual void requestModel3D(Layer layer, OctreeNodeData &data, const Model3DCallback& callback) = 0;
    virtual bool isNodeUpToDate(Layer layer, OctreeNodeData &data, uint version) = 0;
};