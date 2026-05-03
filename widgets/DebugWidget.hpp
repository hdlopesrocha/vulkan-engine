#pragma once

#include "Widget.hpp"
#include "../math/Camera.hpp"
#include "../vulkan/renderer/VegetationRenderer.hpp"
#include <vector>
#include "../utils/MaterialProperties.hpp"
#include <imgui.h>
#include <cstddef>

class DebugWidget : public Widget {
public:
    DebugWidget(std::vector<MaterialProperties>* materials, Camera* camera, size_t* cubeCount, VegetationRenderer* vegetationRenderer);
    
    void render() override;
    bool getShowVegetationDensityDebug() const { return showVegetationDensityDebug; }
    
private:
    std::vector<MaterialProperties>* materials;
    Camera* camera;
    size_t* cubeCount;
    VegetationRenderer* vegetationRenderer;
    bool showVegetationDensityDebug = false;
};
