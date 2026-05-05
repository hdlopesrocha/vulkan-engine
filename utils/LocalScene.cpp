#include "LocalScene.hpp"
#include "../space/OctreeFile.hpp"
#include "../space/OctreeNode.hpp"
#include "../space/OctreeAllocator.hpp"
#include <iostream>
#include <chrono>

LocalScene::LocalScene()
    : opaqueOctree(BoundingCube(glm::vec3(0.0f), 30.0f), glm::pow(2, 9)),
      transparentOctree(BoundingCube(glm::vec3(0.0f), 30.0f), glm::pow(2, 9)),
      threadPool(std::thread::hardware_concurrency()),
      opaqueLayerInfo(),
      transparentLayerInfo() {}

LocalScene::~LocalScene() = default;

Octree& LocalScene::getOpaqueOctree() { return opaqueOctree; }
const Octree& LocalScene::getOpaqueOctree() const { return opaqueOctree; }


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
    // Log geometry stats for every chunk
    //printf("[requestModel3D] Node %p, Layer %d, Cube Min=(%.2f,%.2f,%.2f), Max=(%.2f,%.2f,%.2f), Verts=%zu, Indices=%zu\n",
    //    data.node, (int)layer,
    //    data.cube.getMin().x, data.cube.getMin().y, data.cube.getMin().z,
    //    data.cube.getMax().x, data.cube.getMax().y, data.cube.getMax().z,
    //    tesselator.geometry.vertices.size(), tesselator.geometry.indices.size());
    if (tesselator.geometry.indices.empty()) {
        //printf("[requestModel3D] EMPTY geometry for node %p, Layer %d\n", data.node, (int)layer);
    }
    if(!tesselator.geometry.indices.empty()) {
        callback(tesselator.geometry);
    }
}

bool LocalScene::isNodeUpToDate(Layer layer, OctreeNodeData &data, uint version) {
    return data.node->version >= version;
}

void LocalScene::loadScene(SceneLoaderCallback& callback, const OctreeChangeHandler &opaqueLayerChangeHandler, const OctreeChangeHandler &transparentLayerChangeHandler) {
    std::cout << "LocalScene::loadScene() " << std::endl;
    auto startTime = std::chrono::steady_clock::now();
    callback.loadScene(opaqueOctree, opaqueLayerChangeHandler, transparentOctree, transparentLayerChangeHandler);
    auto endTime = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(endTime - startTime).count();
    std::cout << "LocalScene::loadScene Ok! " << std::to_string(elapsed) << "s"  << std::endl;
}

void LocalScene::save(const std::string& folderPath) {
    OctreeFile opaqueSaver(&opaqueOctree, "opaque");
    OctreeFile transparentSaver(&transparentOctree, "transparent");
    opaqueSaver.save(folderPath, 4096);
    transparentSaver.save(folderPath, 4096);
}

void LocalScene::load(const std::string& folderPath) {
    OctreeFile opaqueLoader(&opaqueOctree, "opaque");
    OctreeFile transparentLoader(&transparentOctree, "transparent");
    opaqueLoader.load(folderPath, 4096);
    transparentLoader.load(folderPath, 4096);
}

static void notifyChunkNodes(OctreeNode* node, const BoundingCube& cube, uint level,
                             OctreeAllocator& allocator, const OctreeChangeHandler& handler) {
    if (!node) return;
    if (node->isChunk()) {
        handler.onNodeAdded(OctreeNodeData(level, node, cube, ContainmentType::Intersects, nullptr));
        return;
    }
    OctreeNode* children[8] = {};
    node->getChildren(allocator, children);
    for (int i = 0; i < 8; ++i) {
        if (children[i])
            notifyChunkNodes(children[i], cube.getChild(i), level + 1, allocator, handler);
    }
}

void LocalScene::load(const std::string& folderPath, const OctreeChangeHandler& opaqueHandler, const OctreeChangeHandler& transparentHandler) {
    load(folderPath);
    if (opaqueOctree.root)
        notifyChunkNodes(opaqueOctree.root, opaqueOctree, 0, *opaqueOctree.allocator, opaqueHandler);
    if (transparentOctree.root)
        notifyChunkNodes(transparentOctree.root, transparentOctree, 0, *transparentOctree.allocator, transparentHandler);
}
