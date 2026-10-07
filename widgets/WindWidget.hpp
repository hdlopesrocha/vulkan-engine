#pragma once

#include "Widget.hpp"
#include "../vulkan/renderer/vegetation/VegetationRenderer.hpp"

class WindWidget : public Widget {
public:
    explicit WindWidget(VegetationRenderer* vegetationRenderer_);
    void render() override;

private:
    VegetationRenderer* vegetationRenderer;
    // Index of the tornado slot selected in the "Tornadoes" section. Drives
    // which slot's sliders are drawn; clamped every frame to [0, 4).
    int selectedTornado = 0;
};
