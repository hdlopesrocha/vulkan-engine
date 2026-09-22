#pragma once

#include "GraphicsQuality.hpp"
#include "Settings.hpp"

// Command that applies a graphics-quality preset to the runtime settings:
//   Maximum — every secondary-visibility ray path on (RT solid/water
//             reflections, water refraction, thickness, ray-traced depth),
//             global water blur on, geometry/wave tessellation on,
//             shadows on.
//   Minimal — all of the above off (blur and tessellation included), so the
//             renderer falls back to the cheapest raster paths.
//
// Global-only and renderer-agnostic: the preset edits Settings fields that
// gate the per-material water features (blurEnabled, rtWaterReflections,
// rtRefractions), so the authored per-layer WaterParams are NEVER modified —
// the shaders combine the global gate with the material flag. Keeping the
// command free of renderer types also lets the headless server include it
// without any Vulkan linkage.
class GraphicsSettingsCommand {
public:
    explicit GraphicsSettingsCommand(GraphicsQuality quality) : quality_(quality) {}

    GraphicsQuality quality() const { return quality_; }

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
        settings.blurEnabled = true;
        settings.tessellationEnabled = true;
        settings.shadowTessellationEnabled = true;
        settings.enableShadows = true;
    }

    static void applyMinimal(Settings& settings) {
        // Local contact shadows are disabled as well so SceneRenderer's RT
        // runtime gate (any ray path on) turns the RT pipeline off entirely.
        settings.rtReflections = false;
        settings.rtWaterReflections = false;
        settings.rtRefractions = false;
        settings.rtThickness = false;
        settings.rtWaterDepth = false;
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
