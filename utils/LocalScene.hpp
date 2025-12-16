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

    void requestVisibleNodes(Layer layer, glm::mat4 viewMatrix, VisibleNodeCallback& callback) override {
        Octree* tree = layer == LAYER_OPAQUE ? &opaqueOctree : &transparentOctree;

        OctreeVisibilityChecker checker;
        checker.update(viewMatrix);
        tree->iterate(checker);
        for(const auto& nodeData : checker.visibleNodes) {
            callback.onVisibleNode(nodeData);
        }
    }

    void requestModel3D(Layer layer, OctreeNodeData &data, Model3DCallback& callback) override {
        long tessCount = 0;
        Octree* tree = layer == LAYER_OPAQUE ? &opaqueOctree : &transparentOctree;
        long trianglesCount = 0;
        ThreadContext context = ThreadContext(data.cube);
        Tesselator tesselator(&trianglesCount, &context);
        std::vector<OctreeNodeTriangleHandler*> handlers;
        handlers.emplace_back(&tesselator);
        Processor processor(&trianglesCount, threadPool, &context, &handlers);
        processor.iterateFlatIn(*tree, data);

        if(tesselator.geometry->indices.size() > 0) {
            InstanceGeometry<InstanceData> * pre = new InstanceGeometry<InstanceData>(tesselator.geometry);
            pre->instances.emplace_back(InstanceData(0, glm::mat4(1.0), 0.0f));
            Model3D model = Model3D();
            model.setGeometry(pre->geometry->vertices, std::vector<uint16_t>(pre->geometry->indices.begin(), pre->geometry->indices.end()));
            callback.onModel3DLoaded(model);
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