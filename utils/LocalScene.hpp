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
    Octree& getOpaqueOctree() { return opaqueOctree; }
    const Octree& getOpaqueOctree() const { return opaqueOctree; }
    Octree transparentOctree;
    ThreadPool threadPool;

public:
    // Instance/visibility layers and change handlers (owned by LocalScene)
    OctreeLayer<InstanceData> opaqueLayerInfo;
    OctreeLayer<InstanceData> transparentLayerInfo;
    SolidSpaceChangeHandler opaqueLayerChangeHandler;
    LiquidSpaceChangeHandler transparentLayerChangeHandler;
    LocalScene() : 
        opaqueOctree(BoundingCube(glm::vec3(0.0f), 30.0f), glm::pow(2, 9)), 
        transparentOctree(BoundingCube(glm::vec3(0.0f), 30.0f), glm::pow(2, 9)),
        threadPool(std::thread::hardware_concurrency()),
        opaqueLayerInfo(),
        transparentLayerInfo(),
        opaqueLayerChangeHandler(&opaqueLayerInfo),
        transparentLayerChangeHandler(&transparentLayerInfo)
        {
    };
    ~LocalScene() = default;

    void requestVisibleNodes(Layer layer, glm::mat4 viewMatrix, const VisibleNodeCallback& callback) override {
        Octree* tree = layer == LAYER_OPAQUE ? &opaqueOctree : &transparentOctree;

        // nodeDataMap.clear();  // Don't clear, keep data for async requests

        OctreeVisibilityChecker checker;
        checker.update(viewMatrix);
        tree->iterate(checker);
        callback(checker.visibleNodes);
    }

    void requestModel3D(Layer layer, OctreeNodeData &data, const GeometryCallback& callback) override {
        long tessCount = 0;
        Octree* tree = layer == LAYER_OPAQUE ? &opaqueOctree : &transparentOctree;
        long trianglesCount = 0;
        ThreadContext context = ThreadContext(data.cube);
        Tesselator tesselator(&trianglesCount, &context);
        std::vector<OctreeNodeTriangleHandler*> handlers;
        handlers.emplace_back(&tesselator);
        Processor processor(&trianglesCount, threadPool, &context, &handlers);
        processor.iterateFlatIn(*tree, data);
        
        if(!tesselator.geometry.indices.empty()) {
            callback(tesselator.geometry);
        }
    }

    bool isNodeUpToDate(Layer layer, OctreeNodeData &data, uint version) override {
        return data.node->version >= version;
    }

    void loadScene(SceneLoaderCallback& callback) override {

        std::cout << "LocalScene::loadScene() " << std::endl;
        auto startTime = std::chrono::steady_clock::now();

        callback.loadScene(opaqueOctree,opaqueLayerChangeHandler, transparentOctree, transparentLayerChangeHandler);

        auto endTime = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(endTime - startTime).count();
        std::cout << "LocalScene::loadScene Ok! " << std::to_string(elapsed) << "s"  << std::endl;
    }

    // Iterate all chunk nodes in the scene and call the callback for each
    // This is used to load all meshes after scene loading is complete
    void forEachChunkNode(Layer layer, const VisibleNodeCallback& callback) {
        Octree* tree = layer == LAYER_OPAQUE ? &opaqueOctree : &transparentOctree;
        
        // Create a simple iterator that collects all chunk nodes
        class ChunkCollector : public IteratorHandler {
        public:
            std::vector<OctreeNodeData> chunkNodes;
            
            void before(const Octree &tree, OctreeNodeData &params) override {}
            void after(const Octree &tree, OctreeNodeData &params) override {
                if (params.node->isChunk() && params.node->getType() == SpaceType::Surface) {
                    chunkNodes.push_back(params);
                }
            }
            bool test(const Octree &tree, OctreeNodeData &params) override {
                return true; // Visit all nodes
            }
            void getOrder(const Octree &tree, OctreeNodeData &params, uint8_t order[8]) override {
                for (uint8_t i = 0; i < 8; ++i) order[i] = i;
            }
        };
        
        ChunkCollector collector;
        tree->iterate(collector);
        
        if (!collector.chunkNodes.empty()) {
            callback(collector.chunkNodes);
        }
    }
};