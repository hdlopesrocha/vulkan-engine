#include <iostream>
#include "utils/LocalScene.hpp"
#include "utils/MainSceneLoader.hpp"


LocalScene * mainScene;

int main(int argc, char** argv) {

    mainScene = new LocalScene();

    MainSceneLoader mainSceneLoader = MainSceneLoader();
    mainScene->loadScene(mainSceneLoader);

    std::cout << "server: started" << std::endl;
    return 0;
}
