#include "Scene.hpp"
#include "../space/Octree.hpp"
#include "../space/OctreeVisibilityChecker.hpp"
#include "../space/Tesselator.hpp"

class LocalScene : public Scene {

    Octree * opaqueOctree = NULL;
    Octree * transparentOctree = NULL;

public:
    LocalScene() {
        opaqueOctree = new Octree(BoundingCube(glm::vec3(0.0f), 1.0f), 32.0f);
        transparentOctree = new Octree(BoundingCube(glm::vec3(0.0f), 1.0f), 32.0f);
    };
    ~LocalScene() = default;

    void requestVisibleNodes(Layer layer, glm::mat4 viewMatrix, VisibleNodeCallback& callback) override {
        OctreeVisibilityChecker checker;
        checker.update(viewMatrix);

        if(layer == LAYER_OPAQUE) {
            opaqueOctree->iterate(checker);
        } else if(layer == LAYER_TRANSPARENT) {
            transparentOctree->iterate(checker);
        }
        for(const auto& nodeData : checker.visibleNodes) {
            callback.onVisibleNode((long)nodeData.node, nodeData.node->version);
        }
    }

    void requestModel3D(Layer layer, long nodeId, Model3DCallback& callback) override {
        long tessCount = 0;
        Tesselator tesselator(&tessCount, nullptr);
        OctreeNode* node = (OctreeNode*)nodeId;
        
        if(layer == LAYER_OPAQUE) {
            //TODO: tesselate opaque octree node
        } else if(layer == LAYER_TRANSPARENT) {
            //TODO: tesselate transparent octree node
        }
        // Local scene does not support dynamic loading; no models to provide
    }

    bool isNodeUpToDate(Layer layer, long nodeId, uint version) override {
        OctreeNode* node = (OctreeNode*)nodeId;       
        if(layer == LAYER_OPAQUE) {
            //TODO: check opaque octree node version
        } else if(layer == LAYER_TRANSPARENT) {
            //TODO: check transparent octree node version  
        }
        
        // Local scene does not support dynamic updates; all nodes are considered up to date
        //TODO: add version to octree node and compare
        return true;
    }

    void loadScene(SceneLoaderCallback& callback) override {
        callback.loadScene(*opaqueOctree, *transparentOctree);
    }
};