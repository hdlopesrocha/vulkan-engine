#pragma once

#include "Widget.hpp"
#include "../math/Camera.hpp"
#include <vector>
#include "../utils/MaterialProperties.hpp"
#include <imgui.h>
#include <cstddef>

class DebugWidget : public Widget {
public:
    DebugWidget(std::vector<MaterialProperties>* materials, Camera* camera, size_t* cubeCount);
    
    void render() override;
    
private:
    std::vector<MaterialProperties>* materials;
    Camera* camera;
    size_t* cubeCount;
};
