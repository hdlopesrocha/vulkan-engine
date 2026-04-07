#pragma once

#include "Widget.hpp"
#include "../vulkan/WaterRenderer.hpp"

class WaterWidget : public Widget {
public:
    explicit WaterWidget(WaterRenderer* renderer);
    void render() override;

private:
    WaterRenderer* renderer;
};
