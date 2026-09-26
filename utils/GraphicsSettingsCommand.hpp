#pragma once

#include "GraphicsQuality.hpp"
#include "Settings.hpp"

// Command that applies a graphics-quality preset to the runtime settings:
//   Maximum — every secondary-visibility ray path on (RT solid/water
//             reflections, water refraction, thickness, ray-traced depth),
//             global water blur on, geometry/wave tessellation on, shadows
//             on, full-rate dual-trace rays.
//   Minimal — all of the above off (blur and tessellation included), so the
//             renderer falls back to the cheapest raster paths.
//
// Global-only and renderer-agnostic: the preset edits Settings fields that
// gate the per-material water features (blurEnabled, rtWaterReflections,
// rtRefractions) without holding renderer types, so the headless server can
// still include it. The per-layer WaterParams are authored values and are
// never touched.
class GraphicsSettingsCommand {
public:
    explicit GraphicsSettingsCommand(GraphicsQuality quality) : quality_(quality) {}

    GraphicsQuality quality() const { return quality_; }

    // Global Settings only; the authored per-layer WaterParams stay untouched.
    void execute(Settings& settings) const {
        switch (quality_) {
        case GraphicsQuality::Maximum:
            applyMaximum(settings);
            break;
        case GraphicsQuality::Minimal:
            applyMinimal(settings);
            break;
        }
    }

private:
    static void applyMaximum(Settings& settings) {
        settings.rtReflections = true;
        settings.rtWaterReflections = true;
        settings.rtRefractions = true;
        settings.rtThickness = true;
        settings.rtWaterDepth = true;
        // Ray quality: Maximum means the full-rate, dual-trace reference. The
        // shipped defaults are the budgeted shortcuts - rtRayScale = 1
        // (checkerboard half-rate inline rays) and rtSingleRay = true (water
        // traces reflection XOR refraction as a stochastic pick). With the xor
        // on ~96% of pixels pick the reflection lobe (reflMixEst is Fresnel
        // weighted and reflectionStrength pushes it toward 1), so the water
        // reads as a pure mirror and refraction looks broken. Maximum traces
        // both lobes at full rate instead.
        settings.rtRayScale = 0;
        settings.rtSingleRay = false;
        settings.blurEnabled = true;
        settings.tessellationEnabled = true;
        settings.shadowTessellationEnabled = true;
        settings.enableShadows = true;
        // Full-resolution water targets (perf report 21 M10): the composite
        // blur is depth-guided, so Maximum keeps 1.0 for the sharpest body.
        settings.waterRenderScale = 1.0f;
    }

    static void applyMinimal(Settings& settings) {
        // Local contact shadows are disabled as well so SceneRenderer's RT
        // runtime gate (any ray path on) turns the RT pipeline off entirely.
        settings.rtReflections = false;
        settings.rtWaterReflections = false;
        settings.rtRefractions = false;
        settings.rtThickness = false;
        settings.rtWaterDepth = false;
        // Keep the budgeted ray shortcuts explicit on the low preset.
        settings.rtRayScale = 1;
        settings.rtSingleRay = true;
        settings.rtLocalShadows = false;
        settings.blurEnabled = false;
        // The global tessellation toggle drives both the solid and the water
        // TCS (tess level collapses to 1 when it is off).
        settings.tessellationEnabled = false;
        settings.shadowTessellationEnabled = false;
        // Shadow maps + cascade passes off (the solid pass then uses the
        // unshadowed direct-lighting path).
        settings.enableShadows = false;
        // Half-resolution water targets (perf report 21 M10): Minimal's water
        // is blur-tolerant, and the blur disc is authored in screen-space
        // units, so the look survives while fragment/bandwidth/target costs
        // quarter. Propagates via the existing per-frame waterRenderScale
        // check (device-idle target recreation); the slider still overrides.
        settings.waterRenderScale = 0.5f;
    }

    GraphicsQuality quality_;
};
