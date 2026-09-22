#pragma once

#include "GraphicsQuality.hpp"
#include "Settings.hpp"
#include "WaterParams.hpp"

#include <cstdint>
#include <functional>
#include <vector>

// Command that applies a graphics-quality preset to the runtime settings:
//   Maximum — every secondary-visibility ray path on (RT solid/water
//             reflections, water refraction, thickness, ray-traced depth),
//             per-layer water blur on, geometry/wave tessellation on,
//             shadows on.
//   Minimal — all of the above off (blur and tessellation included), with
//             per-layer water reflection/refraction lobes off too, so the
//             renderer falls back to the cheapest raster paths.
//
// Header-only and renderer-agnostic: callers that own a WaterRenderer pass an
// `upload` callback so the touched per-layer params reach the GPU even when
// the Water Settings widget is hidden. Keeping the command free of renderer
// types also lets the headless server include it without any Vulkan linkage.
class GraphicsSettingsCommand {
public:
    using WaterUploadFn = std::function<void(uint32_t layer, const WaterParams& params)>;

    explicit GraphicsSettingsCommand(GraphicsQuality quality) : quality_(quality) {}

    GraphicsQuality quality() const { return quality_; }

    void execute(Settings& settings,
                 std::vector<WaterParams>& waterParams,
                 const WaterUploadFn& upload = {}) const {
        switch (quality_) {
        case GraphicsQuality::Maximum:
            applyMaximum(settings, waterParams);
            break;
        case GraphicsQuality::Minimal:
            applyMinimal(settings, waterParams);
            break;
        }

        if (upload) {
            for (uint32_t layer = 0; layer < waterParams.size(); ++layer) {
                upload(layer, waterParams[layer]);
            }
        }
    }

private:
    static void applyMaximum(Settings& settings, std::vector<WaterParams>& waterParams) {
        settings.rtReflections = true;
        settings.rtWaterReflections = true;
        settings.rtRefractions = true;
        settings.rtThickness = true;
        settings.rtWaterDepth = true;
        settings.tessellationEnabled = true;
        settings.shadowTessellationEnabled = true;
        settings.enableShadows = true;

        for (WaterParams& params : waterParams) {
            params.enableBlur = true;
            params.enableReflection = true;
            params.enableRefraction = true;
        }
    }

    static void applyMinimal(Settings& settings, std::vector<WaterParams>& waterParams) {
        // Local contact shadows are disabled as well so SceneRenderer's RT
        // runtime gate (any ray path on) turns the RT pipeline off entirely.
        settings.rtReflections = false;
        settings.rtWaterReflections = false;
        settings.rtRefractions = false;
        settings.rtThickness = false;
        settings.rtWaterDepth = false;
        settings.rtLocalShadows = false;
        // The global tessellation toggle drives both the solid and the water
        // TCS (tess level collapses to 1 when it is off).
        settings.tessellationEnabled = false;
        settings.shadowTessellationEnabled = false;
        // Shadow maps + cascade passes off (the solid pass then uses the
        // unshadowed direct-lighting path).
        settings.enableShadows = false;

        for (WaterParams& params : waterParams) {
            params.enableBlur = false;
            params.enableReflection = false;
            params.enableRefraction = false;
        }
    }

    GraphicsQuality quality_;
};
