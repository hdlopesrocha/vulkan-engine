#pragma once

#include <array>

#include "TornadoSettings.hpp"
#include "WindFieldConstants.hpp"

struct WindFieldSettings {
    std::array<TornadoSettings, kWindFieldMaxTornadoes> tornadoes{};
    // Wind debug mode shown by the composite overlay: 0 = off,
    // 1 = velocity heat volume, 2 = velocity isosurface,
    // 3 = air-compression heat volume, 4 = normalized direction volume.
    // Owned by the ray marching widget (not the wind widget: it visualizes,
    // it doesn't simulate).
    int windDebugMode = 0;
    // Isosurface speed for mode 2 (m/s): the surface where |wind| = iso.
    float windDebugIso = 8.0f;
};
