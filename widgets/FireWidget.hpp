#pragma once

#include "Widget.hpp"
#include "../vulkan/renderer/VegetationRenderer.hpp"
#include <functional>

class FireWidget : public Widget {
public:
    explicit FireWidget(VegetationRenderer* vegetationRenderer_);
    void setOnRecaptureFire(std::function<void()> cb) { onRecaptureFire = std::move(cb); }
    void render() override;

private:
    VegetationRenderer* vegetationRenderer;
    std::function<void()> onRecaptureFire;
};
