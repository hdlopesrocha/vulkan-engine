#include <iostream>
#include "utils/LocalScene.hpp"
#include "utils/MainSceneLoader.hpp"


int main(int argc, char** argv) {

    // Initialize and load the main scene so rendering has valid scene data
    SolidSpaceChangeHandler solidHandler(
        [](const OctreeNodeData& nd){ 
            std::cout << "Opaque node updated" << std::endl;
        },
        [](const OctreeNodeData& nd){
            std::cout << "Opaque node erased" << std::endl;
        
        }
    );

    LiquidSpaceChangeHandler liquidHandler(
        [](const OctreeNodeData& nd){ 
            std::cout << "Transparent node updated" << std::endl            ;
        },
        [](const OctreeNodeData& nd){ 
            std::cout << "Transparent node erased" << std::endl;
        }
    );
        

    LocalScene mainScene = LocalScene( solidHandler, liquidHandler );

    MainSceneLoader mainSceneLoader = MainSceneLoader();
    mainScene.loadScene(mainSceneLoader);

    std::cout << "server: started" << std::endl;
    return 0;
}
