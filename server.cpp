#include <iostream>
#include "utils/LocalScene.hpp"
#include "utils/MainSceneLoader.hpp"


int main(int argc, char** argv) {

    NodeDataCallback liquidNodeEventCallback = [](const OctreeNodeData& nd) {
        NodeID nid = reinterpret_cast<NodeID>(nd.node);
        std::cout << "Transparent node updated" << std::endl;
    };
    
    NodeDataCallback liquidNodeEraseCallback = [](const OctreeNodeData& nd) {
        std::cout << "Transparent node erased" << std::endl;
    };

    NodeDataCallback solidNodeEventCallback = [](const OctreeNodeData& nd) {
        NodeID nid = reinterpret_cast<NodeID>(nd.node);
        std::cout << "Opaque node updated" << std::endl;
    };

    NodeDataCallback solidNodeEraseCallback = [](const OctreeNodeData& nd) {
        std::cout << "Opaque node erased" << std::endl;
    };

    // Initialize and load the main scene so rendering has valid scene data
    UniqueOctreeChangeHandler solidHandler(
        SolidSpaceChangeHandler(solidNodeEventCallback, solidNodeEraseCallback)
    );

    UniqueOctreeChangeHandler liquidHandler(
        LiquidSpaceChangeHandler(liquidNodeEventCallback, liquidNodeEraseCallback)  
    );
        

    LocalScene mainScene;

    MainSceneLoader mainSceneLoader = MainSceneLoader();
    mainScene.loadScene(mainSceneLoader, solidHandler, liquidHandler);

    std::cout << "server: started" << std::endl;
    return 0;
}
