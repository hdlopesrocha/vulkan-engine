#include <iostream>
#include "utils/LocalScene.hpp"
#include "utils/MainSceneLoader.hpp"


int main(int argc, char** argv) {

    LocalScene mainScene = LocalScene();

    MainSceneLoader mainSceneLoader = MainSceneLoader(&mainScene.transparentLayerChangeHandler, &mainScene.opaqueLayerChangeHandler);
    mainScene.loadScene(mainSceneLoader);

    std::cout << "server: started" << std::endl;
    return 0;
}
