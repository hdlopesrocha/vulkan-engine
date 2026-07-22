#pragma once

#include "Widget.hpp"
#include "../vulkan/renderer/WaterRenderer.hpp"
#include "../utils/WaterParams.hpp"
#include <vector>

class WaterWidget : public Widget {
public:
    WaterWidget(WaterRenderer* renderer_, std::vector<WaterParams>* params_);
    void render() override;
    int getCurrentLayer() const { return currentLayer; }

private:
    WaterRenderer* renderer;
    std::vector<WaterParams>* params;
    int currentLayer = 0;
};
