#pragma once

#include "Widget.hpp"
#include "../vulkan/renderer/SdfRenderer.hpp"

// Generic SDF renderer controls. SDF fire is always on (volumetric
// material with flame anchors from brush-4 lava chunks); the widget
// drives the generic renderer (mode, marching, debug) plus fire tuning.
class SdfWidget : public Widget {
public:
    SdfWidget(SdfRenderer* sdf);
    void render() override;

    float timeScale = 1.0f;
    // Lava source controls (write through to the SdfRenderer collector;
    // apply to newly streamed chunks, like the vegetation density control).
    float lavaAreaPerFlame = 10000.0f; // m² of lava surface per flame
    float lavaScale = 32.0f;    // flame anchor scale multiplier
    float lavaSpikiness = 0.35f; // spike amplitude (0 = smooth rounded capsule)
    float lavaTipRadius = 0.25f; // tip radius, local x scale (= m at scale 32)
    float lavaBaseRadius = 1.0f; // base radius, local x scale (= m at scale 32)
    float lavaHeight = 3.2f;     // base-to-tip height, local units
    float lavaSpikeFreq = 2.0f;  // tongue count around the flame axis

private:
    SdfRenderer* sdfRenderer = nullptr;
};
