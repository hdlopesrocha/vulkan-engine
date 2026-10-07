#pragma once

#include "Widget.hpp"
#include "../vulkan/renderer/sdf/SdfRenderer.hpp"
#include "../vulkan/renderer/vegetation/VegetationRenderer.hpp"

class Camera;

// Ray-marching controls for the generic GPU SDF renderer (surface/volume/
// emissive/transparent). Fire (lava-anchored flames) and smoke (a sphere
// shape with bullet interaction) are just shapes inside the renderer; all
// of their tuning lives in SdfEffectConfig owned by the renderer and the
// sections below bind its fields directly, so renderer and widget share one
// copy. Replaces the former SdfWidget (generic part) + SmokeBulletWidget
// (effect part), merged so all SDF logic lives in one place.
class RaymarchWidget : public Widget {
public:
    RaymarchWidget(SdfRenderer* sdf, Camera* cam, VegetationRenderer* veg);
    void render() override;

    float timeScale = 1.0f;

private:
    SdfRenderer* sdfRenderer = nullptr;
    Camera* camera = nullptr;
    VegetationRenderer* vegetationRenderer = nullptr;
};
