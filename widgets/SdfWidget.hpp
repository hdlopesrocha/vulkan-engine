#pragma once

#include "Widget.hpp"
#include "../vulkan/renderer/SdfRenderer.hpp"

// Generic SDF renderer controls. Fire is the first volumetric consumer
// (flames anchor to brush-4 lava terrain as chunks stream in); the lava
// tuning lives in SdfEffectConfig owned by the renderer and the sliders
// below bind its fields directly, so renderer and widget share one copy.
class SdfWidget : public Widget {
public:
    explicit SdfWidget(SdfRenderer* sdf);
    void render() override;

    float timeScale = 1.0f;

private:
    SdfRenderer* sdfRenderer = nullptr;
};
