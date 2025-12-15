#pragma once

#include "Widget.hpp"
#include "../utils/Camera.hpp"
#include "../vulkan/TextureManager.hpp"
#include <imgui.h>
#include <cstddef>

class DebugWidget : public Widget {
public:
    DebugWidget(TextureManager* textureManager, Camera* camera, size_t* cubeCount);
    
    void render() override;
    
private:
    TextureManager* textureManager;
    Camera* camera;
    size_t* cubeCount;
};
