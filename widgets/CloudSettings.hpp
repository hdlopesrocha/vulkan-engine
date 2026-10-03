#pragma once

#include <glm/glm.hpp>

// Volumetric cloud settings: three altitude tiers + global wind/lighting.
// Altitudes follow the standard classification:
//   Low  ~2 km        (stratus, stratocumulus, cumulus; fog = ground-touching stratus)
//   Mid  2..7 km      (altocumulus, altostratus: water/ice mix)
//   High 5..13+ km    (cirrus, cirrocumulus, cirrostratus: ice crystals)
struct CloudSettings {
    // Master enable (also gated by Settings::cloudsEnabled). Minimal preset disables.
    bool enabled = true;

    // --- Low tier (~2 km): dense water-droplet layers ---
    bool lowEnabled = true;
    float lowBaseHeight = 2000.0f;   // m above origin
    float lowThickness = 400.0f;     // m slab thickness
    float lowCoverage = 0.45f;       // 0 = clear, 1 = overcast
    float lowDensity = 1.0f;         // extinction multiplier
    float lowScale = 0.00045f;       // horizontal feature scale (1/m)
    float lowWindSpeedMul = 1.0f;    // multiplier on global wind

    // --- Mid tier (2..7 km): altocumulus / altostratus ---
    bool midEnabled = true;
    float midBaseHeight = 4500.0f;
    float midThickness = 800.0f;
    float midCoverage = 0.35f;
    float midDensity = 0.8f;
    float midScale = 0.00028f;
    float midWindSpeedMul = 1.6f;

    // --- High tier (5..13 km): cirrus ice crystals, wispy streaks ---
    bool highEnabled = true;
    float highBaseHeight = 9000.0f;
    float highThickness = 1000.0f;
    float highCoverage = 0.30f;
    float highDensity = 0.5f;
    float highScale = 0.00016f;
    float highWindSpeedMul = 2.5f;

    // --- Global wind / animation ---
    float windSpeed = 12.0f;         // m/s horizontal drift
    float windAngleDeg = 45.0f;      // 0 = +X, 90 = +Z
    float timeScale = 0.02f;          // animation speed multiplier
    float detailStrength = 0.35f;    // erosion detail amount (0 = smooth)

    // --- Global look / lighting ---
    float densityScale = 1.0f;       // global extinction multiplier
    float shadowStrength = 0.75f;    // 0 = no ground shadows, 1 = full
    float ambientBoost = 0.35f;      // ambient lift inside clouds
    float silverLining = 0.6f;       // forward-scattering bright edge
    float sunForwardG = 0.55f;       // Henyey-Greenstein g (-0.9..0.9)
    float exposure = 1.0f;           // cloud brightness multiplier

    // --- Quality / performance ---
    // Raymarch steps for the fullscreen sky pass (equirect uses half, min 4).
    int raymarchSteps = 8;
    // Light-march steps toward the sun per raymarch sample (2..6).
    int lightSteps = 3;
};
