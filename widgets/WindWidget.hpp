#pragma once

#include "Widget.hpp"
#include "../vulkan/renderer/vegetation/VegetationRenderer.hpp"

class Settings;

class WindWidget : public Widget {
public:
    explicit WindWidget(VegetationRenderer* vegetationRenderer_, Settings* settings_ = nullptr);
    void render() override;

private:
    VegetationRenderer* vegetationRenderer;
    // For the raymarched wind-surface mode (enable + target speed). Nullable:
    // the widget still edits the shared wind field without it.
    Settings* settings;
    // Index of the tornado slot selected in the "Tornadoes" section. Drives
    // which slot's sliders are drawn; clamped every frame to [0, 4).
    int selectedTornado = 0;
};
