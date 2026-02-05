#include "LocalScene.hpp"
#include <iostream>
#include <chrono>

LocalScene::LocalScene(const OctreeChangeHandler &opaqueLayerChangeHandler, const OctreeChangeHandler &transparentLayerChangeHandler)
    : opaqueOctree(BoundingCube(glm::vec3(0.0f), 30.0f), glm::pow(2, 9)),
      transparentOctree(BoundingCube(glm::vec3(0.0f), 30.0f), glm::pow(2, 9)),
      threadPool(std::thread::hardware_concurrency()),
      opaqueLayerInfo(),
      transparentLayerInfo(),
      opaqueLayerChangeHandler(opaqueLayerChangeHandler),
      transparentLayerChangeHandler(transparentLayerChangeHandler) {}

LocalScene::~LocalScene() = default;

Octree& LocalScene::getOpaqueOctree() { return opaqueOctree; }
const Octree& LocalScene::getOpaqueOctree() const { return opaqueOctree; }

void LocalScene::requestVisibleNodes(Layer layer, glm::mat4 viewMatrix, const VisibleNodeCallback& callback) {
    Octree* tree = layer == LAYER_OPAQUE ? &opaqueOctree : &transparentOctree;
    OctreeVisibilityChecker checker;
    checker.update(viewMatrix);
    tree->iterate(checker);
    callback(checker.visibleNodes);
}

void LocalScene::requestModel3D(Layer layer, OctreeNodeData &data, const GeometryCallback& callback) {
    long tessCount = 0;
    Octree* tree = layer == LAYER_OPAQUE ? &opaqueOctree : &transparentOctree;
    long trianglesCount = 0;
    ThreadContext context = ThreadContext(data.cube);
    Tesselator tesselator(&trianglesCount);
    std::vector<OctreeNodeTriangleHandler*> handlers;
    handlers.emplace_back(&tesselator);
    Processor processor(&tessCount, threadPool, &context, &handlers);
    tree->iterateFlat(processor, data);
    if(!tesselator.geometry.indices.empty()) {
        callback(tesselator.geometry);
    }
}

bool LocalScene::isNodeUpToDate(Layer layer, OctreeNodeData &data, uint version) {
    return data.node->version >= version;
}

void LocalScene::loadScene(SceneLoaderCallback& callback) {
    std::cout << "LocalScene::loadScene() " << std::endl;
    auto startTime = std::chrono::steady_clock::now();
    callback.loadScene(opaqueOctree, opaqueLayerChangeHandler, transparentOctree, transparentLayerChangeHandler);
    auto endTime = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(endTime - startTime).count();
    std::cout << "LocalScene::loadScene Ok! " << std::to_string(elapsed) << "s"  << std::endl;
}
