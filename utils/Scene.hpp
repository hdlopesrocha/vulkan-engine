#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include "Model3D.hpp"
#include "../space/Octree.hpp"

enum Layer {
    LAYER_OPAQUE = 0,
    LAYER_TRANSPARENT = 1,
    LAYER_UI = 2,
    LAYER_COUNT = 3
};

class VisibleNodeCallback {
public:
    VisibleNodeCallback() = default;
    ~VisibleNodeCallback() = default;

    virtual void onVisibleNode(long nodeId, uint version) = 0;
};

class Model3DCallback {
public:
    Model3DCallback() = default;
    ~Model3DCallback() = default;

    virtual void onModel3DLoaded(Model3D &model) = 0;
};

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
    virtual void requestVisibleNodes(Layer layer, glm::mat4 viewMatrix, VisibleNodeCallback& callback) = 0;
    virtual void requestModel3D(Layer layer, long nodeId, Model3DCallback& callback) = 0;
    virtual bool isNodeUpToDate(Layer layer, long nodeId, uint version) = 0;
};