#pragma once

#include "Widget.hpp"
#include "../vulkan/renderer/WaterRenderer.hpp"
#include "types/WaterSettings.hpp"
#include <vector>

class WaterWidget : public Widget {
public:
    WaterWidget(WaterRenderer* renderer_, std::vector<WaterSettings>* params_);
    void render() override;

private:
    WaterRenderer* renderer;
    std::vector<WaterSettings>* params;
    int currentLayer = 0;
};
