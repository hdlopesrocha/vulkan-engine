#include <iostream>
#include "utils/LocalScene.hpp"
#include "utils/MainSceneLoader.hpp"
#include "space/UniqueChangeCollector.hpp"


int main(int argc, char** argv) {

    Octree::OctreeNodeDataHandler liquidNodeEventCallback = [](const OctreeNodeData& nd) {
        std::cout << "Transparent node updated" << std::endl;
    };
    
    Octree::OctreeNodeDataHandler liquidNodeEraseCallback = [](const OctreeNodeData& nd) {
        std::cout << "Transparent node erased" << std::endl;
    };

    Octree::OctreeNodeDataHandler solidNodeEventCallback = [](const OctreeNodeData& nd) {
        std::cout << "Opaque node updated" << std::endl;
    };

    Octree::OctreeNodeDataHandler solidNodeEraseCallback = [](const OctreeNodeData& nd) {
        std::cout << "Opaque node erased" << std::endl;
    };

    LocalScene mainScene;

    MainSceneLoader mainSceneLoader = MainSceneLoader();
    mainScene.loadScene(mainSceneLoader,
        solidNodeEventCallback,
        solidNodeEraseCallback,
        liquidNodeEventCallback,
        liquidNodeEraseCallback);

    std::cout << "server: started" << std::endl;
    return 0;
}