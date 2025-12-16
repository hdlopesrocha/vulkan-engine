#include "Scene.hpp"
#include "../space/Octree.hpp"
#include "../space/OctreeVisibilityChecker.hpp"
#include "../space/Tesselator.hpp"
#include "../space/Processor.hpp"

class LocalScene : public Scene {

    Octree opaqueOctree;
    Octree transparentOctree;
	ThreadPool threadPool;

public:
    LocalScene() : 
        opaqueOctree(BoundingCube(glm::vec3(0.0f), 30.0f), glm::pow(2, 9)), 
        transparentOctree(BoundingCube(glm::vec3(0.0f), 30.0f), glm::pow(2, 9)),
        threadPool(std::thread::hardware_concurrency()) {
    };
    ~LocalScene() = default;

    void requestVisibleNodes(Layer layer, glm::mat4 viewMatrix, const VisibleNodeCallback& callback) override {
        Octree* tree = layer == LAYER_OPAQUE ? &opaqueOctree : &transparentOctree;

        OctreeVisibilityChecker checker;
        checker.update(viewMatrix);
        tree->iterate(checker);
        for(const auto& nodeData : checker.visibleNodes) {
            callback(nodeData);
        }
    }

    void requestModel3D(Layer layer, OctreeNodeData &data, const Model3DCallback& callback) override {
        long tessCount = 0;
        Octree* tree = layer == LAYER_OPAQUE ? &opaqueOctree : &transparentOctree;
        long trianglesCount = 0;
        ThreadContext context = ThreadContext(data.cube);
        Tesselator tesselator(&trianglesCount, &context);
        std::vector<OctreeNodeTriangleHandler*> handlers;
        handlers.emplace_back(&tesselator);
        Processor processor(&trianglesCount, threadPool, &context, &handlers);
        processor.iterateFlatIn(*tree, data);

        if(tesselator.geometry && !tesselator.geometry->indices.empty()) {
            Model3D model;
            // copy vertices
            const std::vector<Vertex> &verts = tesselator.geometry->vertices;
            // convert indices from uint to uint16_t (clamp if necessary)
            std::vector<uint16_t> indices;
            indices.reserve(tesselator.geometry->indices.size());
            for(uint idx : tesselator.geometry->indices) {
                indices.push_back(static_cast<uint16_t>(idx));
            }
            model.setGeometry(verts, indices);
            model.computeNormals();
            model.computeTangents();
            callback(model);
        }
    }

    bool isNodeUpToDate(Layer layer, OctreeNodeData &data, uint version) override {
        return data.node->version >= version;
    }

    void loadScene(SceneLoaderCallback& callback) override {

        std::cout << "LocalScene::loadScene() " << std::endl;
        auto startTime = std::chrono::steady_clock::now();

        callback.loadScene(opaqueOctree, transparentOctree);

        auto endTime = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(endTime - startTime).count();
        std::cout << "LocalScene::loadScene Ok! " << std::to_string(elapsed) << "s"  << std::endl;
    }
};