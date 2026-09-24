#pragma once

#include <cstddef>

// Canonical debug view IDs — single source of truth for the raster solid/water
// fragment paths and the SettingsWidget combo. Keep in sync with
// shaders/includes/debug_modes.glsl (same order, same values).
//
// Both surfaces dispatch on the same IDs. Views that have no meaning on a
// surface simply fall through to that surface's normal shading, so e.g.
// "Roughness" only tints terrain while water keeps rendering water, and
// "Water Noise" only tints water while terrain keeps rendering terrain.
enum class DebugMode : int {
    None = 0,             // normal render
    // Normals
    ShadingNormal,        // solid: material-perturbed world normal | water: displaced wave normal
    GeometricNormal,      // solid: interpolated base normal | water: undisplaced base normal
    FaceNormal,           // both: derivative face normal (rasterizer-visible facets)
    // Solid material
    Albedo,               // solid only: blended albedo actually shaded
    NormalMap,            // solid only: raw normal-map samples
    HeightMap,            // solid only: final height/bump sample (triplanar or UV)
    Roughness,            // solid only: roughness value * material factor
    AmbientOcclusion,     // solid only
    TriplanarWeights,     // solid only: RGB projection blend weights
    MaterialIndex,        // solid: brush/texture index palette | water: water layer id
    UV,                   // solid only: texture UVs
    // Lighting / shadow
    NdotL,                // both: max(dot(N, L), 0)
    LightVector,          // both: direction to the sun, packed to RGB
    Shadow,               // solid: R=CSM, G=RT local, B=combined | water: always 0 by design
    // Reflection
    ReflectionColor,      // solid: env reflection * Fresnel | water: RT/SSR/sky reflection used
    ReflectionVector,     // both: view reflection direction, packed to RGB
    Fresnel,              // solid: reflection mix factor | water: Schlick Fresnel term
    // Diagnostics (shared)
    RayMask,              // both: ray-budget coverage mask (must stay budgeted)
    TessHeat,             // solid only: tessellation-level heatmap
    // Water
    SkyReflection,        // water only: raw equirect sky sample
    RefractionColor,      // water only: refracted scene color (pre aerial fade)
    WaterNoise,           // water only: refraction noise
    Displacement,         // water only: wave displacement
    Thickness,            // water only: normalized water thickness
    Absorption,           // water only: Beer-Lambert transmittance
    Caustics,             // water only: final caustic mask
    DepthSource,          // water only: which branch produced the thickness (must stay budgeted)
    WaterCompose,         // water only: tint / mirror / thickness channels
    WaterColor,           // water only: the single water tint (one region)
    WaterDepthSources,    // water only: R=solid-scene depth drop, G=water back-face drop, B=final depth (all / zDeep)
    SceneDepth,           // both: scene depth (solid: own fragment depth; water: solid depth behind), linear/far grayscale
    ReflectionSource,     // water only: which branch resolved the reflection (solid/water/guard/SSR/sky/equirect)
    // Water wave-system components (one view per noise, so every term of the
    // wave field can be inspected while tuning its period/amplitude).
    WaterSlope,           // water only: swell along-shore slope (R signed, G = |slope|)
    WaterContact,         // water only: swell profile (R, signed) + shoreline contact foam (G)
    WaterSwell,           // water only: the sine swell profile (R) and its along-shore slope (G), signed
    WaterFoamMask,        // water only: whitewater (R) and shoreline contact (G) coverage
    WaterSpecularNoise,   // water only: highlight perturbation (R) + glitter (G) + sun-lobe gate (B) - shows
                          //            WHERE the lobe lets them run (gated since perf report 20 M10)
    ShoreDirection,       // water only: the shore direction (direction of DECREASING depth), which drives the
                          //            wave movement in every region, packed to RGB + a measured/fallback flag
    WaterGerstner,        // water only: the per-pixel Gerstner evaluation the shading normal is built from:
                          //            R = wave height (grey 0.5 = mean level), G = analytic slope magnitude,
                          //            B = horizontal pinch (the Gerstner crest sharpening)
    Count
};

inline constexpr const char* kDebugModeNames[] = {
    "Default Render",
    "Shading Normal",
    "Geometric Normal",
    "Face Normal",
    "Albedo",
    "Normal Map (raw)",
    "Height/Bump Map",
    "Roughness",
    "Ambient Occlusion",
    "Triplanar Weights",
    "Material/Brush Index",
    "UV Coordinates",
    "N·L (sun diffuse)",
    "Light Vector",
    "Shadow (CSM/RT/combined)",
    "Reflection Color",
    "Reflection Vector",
    "Fresnel",
    "Ray Mask",
    "Tessellation Level Heat",
    "Sky Reflection",
    "RT Refraction Color",
    "Water Noise",
    "Water Displacement",
    "Water Thickness",
    "Absorption (Beer-Lambert)",
    "Caustics",
    "Water Depth Source",
    "Water Compose",
    "Water Color",
    "Water Depth Sources",
    "Scene Depth",
    "Reflection Source",
    "Water Slope (swell)",
    "Water Contact Foam",
    "Water Swell",
    "Water Foam / Contact",
    "Water Specular/Glitter Noise",
    "Water Shore Direction",
    "Water Gerstner",
};

static_assert(sizeof(kDebugModeNames) / sizeof(kDebugModeNames[0])
    == static_cast<std::size_t>(DebugMode::Count));

constexpr DebugMode debugModeFromInt(int value) {
    return (value > 0 && value < static_cast<int>(DebugMode::Count))
        ? static_cast<DebugMode>(value)
        : DebugMode::None;
}

// Views whose displayed value is produced by traced reflection/refraction
// rays must force the full-quality reference path (full-rate + dual-trace) or
// the ray-budget cuts would distort the diagnostics. Ray mask / depth source
// must NOT force it: they visualize the live budgeted behavior itself.
constexpr bool debugModeForcesRtReference(DebugMode mode) {
    switch (mode) {
        case DebugMode::ReflectionColor:
        case DebugMode::RefractionColor:
        case DebugMode::Thickness:
        case DebugMode::ReflectionSource:
        case DebugMode::Shadow:
            return true;
        default:
            return false;
    }
}
