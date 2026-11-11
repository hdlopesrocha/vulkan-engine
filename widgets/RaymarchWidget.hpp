#pragma once

#include "Widget.hpp"
#include "../vulkan/renderer/sdf/SdfRenderer.hpp"
#include "../vulkan/renderer/vegetation/VegetationRenderer.hpp"

class Camera;
class Settings;

// Ray-marching controls for the generic GPU SDF renderer (surface/volume/
// emissive/transparent). Fire (lava-anchored flames) and smoke (a sphere
// shape with bullet interaction) are just shapes inside the renderer; all
// of their tuning lives in SdfEffectConfig owned by the renderer and the
// sections below bind its fields directly, so renderer and widget share one
// copy. Replaces the former SdfWidget (generic part) + SmokeBulletWidget
// (effect part), merged so all SDF logic lives in one place.
//
// Also hosts the terrain solid-path selector (Settings::octreeRaymarchEnabled):
// the octree SDF raymarch is the full-screen equivalent of the tessellated
// raster chunks, including tessellation displacement folded into the SDF.
class RaymarchWidget : public Widget {
public:
    RaymarchWidget(SdfRenderer* sdf, Camera* cam, VegetationRenderer* veg,
                   Settings* settings = nullptr);
    void render() override;

    float timeScale = 1.0f;

private:
    SdfRenderer* sdfRenderer = nullptr;
    Camera* camera = nullptr;
    VegetationRenderer* vegetationRenderer = nullptr;
    Settings* settings = nullptr;
};
