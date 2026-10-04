#pragma once

#include "Widget.hpp"
#include "../vulkan/renderer/WaterRenderer.hpp"
#include "types/WaterParams.hpp"
#include <vector>

class WaterWidget : public Widget {
public:
    WaterWidget(WaterRenderer* renderer_, std::vector<WaterParams>* params_);
    void render() override;

private:
    WaterRenderer* renderer;
    std::vector<WaterParams>* params;
    int currentLayer = 0;
};
