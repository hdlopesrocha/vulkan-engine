#pragma once

#include "GraphicsQuality.hpp"
#include "Settings.hpp"
#include "WaterParams.hpp"

#include <cstddef>
#include <functional>
#include <vector>

// Command that applies a graphics-quality preset to the runtime settings:
//   Maximum — every secondary-visibility ray path on (RT solid/water
//             reflections, water refraction, thickness, ray-traced depth),
//             global water blur on, geometry/wave tessellation on,
//             shadows on, and the Full water look tier: volumetric
//             scattering, foam, caustics and glitter restored to their
//             WaterParams{} defaults.
//   Minimal — all of the above off (blur and tessellation included), so the
//             renderer falls back to the cheapest raster paths, and the
//             Minimal water look tier: volumetric scattering, foam, caustics
//             and glitter disabled, which makes the per-pixel water shader
//             gates take their cheap paths.
//
// Global-only and renderer-agnostic: the preset edits Settings fields that
// gate the per-material water features (blurEnabled, rtWaterReflections,
// rtRefractions) without holding renderer types, so the headless server can
// still include it.
class GraphicsSettingsCommand {
public:
    // Called once per water layer the preset changed (index + updated
    // params), after the edit, so the caller can push it to the GPU.
    using WaterLayerUpload =
        std::function<void(std::size_t layer, const WaterParams& params)>;

    explicit GraphicsSettingsCommand(GraphicsQuality quality) : quality_(quality) {}

    GraphicsQuality quality() const { return quality_; }

    // Global Settings only; the authored per-layer WaterParams stay untouched.
    void execute(Settings& settings) const {
        switch (quality_) {
        case GraphicsQuality::Maximum:
            applyMaximum(settings, nullptr, {});
            break;
        case GraphicsQuality::Minimal:
            applyMinimal(settings, nullptr, {});
            break;
        }
    }

    // Global Settings plus the water look tier on every layer (Full for
    // Maximum, Minimal for Minimal).
    void execute(Settings& settings, std::vector<WaterParams>& waterLayers,
                 const WaterLayerUpload& onLayerChanged = {}) const {
        switch (quality_) {
        case GraphicsQuality::Maximum:
            applyMaximum(settings, &waterLayers, onLayerChanged);
            break;
        case GraphicsQuality::Minimal:
            applyMinimal(settings, &waterLayers, onLayerChanged);
            break;
        }
    }

private:
    static void applyMaximum(Settings& settings, std::vector<WaterParams>* waterLayers,
                             const WaterLayerUpload& onLayerChanged) {
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

    }

    static void applyMinimal(Settings& settings, std::vector<WaterParams>* waterLayers,
                             const WaterLayerUpload& onLayerChanged) {
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

    }

    GraphicsQuality quality_;
};
