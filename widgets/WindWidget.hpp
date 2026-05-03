#pragma once

#include "Widget.hpp"
#include "../vulkan/renderer/VegetationRenderer.hpp"

class WindWidget : public Widget {
public:
    explicit WindWidget(VegetationRenderer* vegetationRenderer);
    void render() override;

private:
    VegetationRenderer* vegetationRenderer;
};
