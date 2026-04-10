#pragma once

#include "Widget.hpp"
#include "../vulkan/WaterRenderer.hpp"
#include "../utils/WaterParams.hpp"
#include <vector>

class WaterWidget : public Widget {
public:
    WaterWidget(WaterRenderer* renderer, std::vector<WaterParams>* params);
    void render() override;
    int getCurrentLayer() const { return currentLayer; }

private:
    WaterRenderer* renderer;
    std::vector<WaterParams>* params;
    int currentLayer = 0;
};
