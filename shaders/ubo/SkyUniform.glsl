#ifndef SKY_UNIFORM_GLSL
#define SKY_UNIFORM_GLSL

// Canonical sky + volumetric-cloud UBO (std140, 224 B), set=0 binding=6.
// Single definition shared with the CPU: vulkan/ubo/SkyUniform.hpp — same
// names, fields and offsets.
// Offset table (verified against the C++ static_asserts):
//   horizonColor 0, warmth 12, zenithColor 16, exponent 28,
//   nightHorizonColor 32, sunFlare 44, nightZenithColor 48, nightIntensity 60,
//   starIntensity 64, skyMode 68, cloudsEnabled 72, lowEnabled 76,
//   midEnabled 80, highEnabled 84, densityScale 88, windResponse 92,
//   windTime 96, detailStrength 100, cloudTime 104, shadowStrength 108,
//   raymarchSteps 112, lightSteps 116, lowCoverage 120, lowDensity 124,
//   lowScale 128, lowWindSpeedMul 132, lowBaseHeight 136, lowThickness 140,
//   midCoverage 144, midDensity 148, midScale 152, midWindSpeedMul 156,
//   midBaseHeight 160, midThickness 164, highCoverage 168, highDensity 172,
//   highScale 176, highWindSpeedMul 180, highBaseHeight 184, highThickness 188,
//   silverLining 192, ambientBoost 196, sunForwardG 200, exposure 204,
//   timeScale 208, cloudRaycastPixelSize 212, invScreenSize 216.
// 224 is the std140 block size (multiple of 16).
// Flags are uint (compare with == 0u / != 0u).
struct SkyUniform {
    vec3 horizonColor;      // offset   0
    float warmth;           // offset  12
    vec3 zenithColor;       // offset  16
    float exponent;         // offset  28
    vec3 nightHorizonColor; // offset  32
    float sunFlare;         // offset  44
    vec3 nightZenithColor;  // offset  48
    float nightIntensity;   // offset  60
    float starIntensity;    // offset  64
    uint skyMode;           // offset  68  0 gradient, 1 grid
    uint cloudsEnabled;     // offset  72
    uint lowEnabled;        // offset  76
    uint midEnabled;        // offset  80
    uint highEnabled;       // offset  84
    float densityScale;     // offset  88
    float windResponse;     // offset  92  cloud gain on the shared wind field
    float windTime;         // offset  96  real shared-wind clock (seconds)
    float detailStrength;   // offset 100
    float cloudTime;        // offset 104  advanced on CPU
    float shadowStrength;   // offset 108
    float raymarchSteps;    // offset 112
    float lightSteps;       // offset 116
    float lowCoverage;      // offset 120
    float lowDensity;       // offset 124
    float lowScale;         // offset 128
    float lowWindSpeedMul;  // offset 132
    float lowBaseHeight;    // offset 136
    float lowThickness;     // offset 140
    float midCoverage;      // offset 144
    float midDensity;       // offset 148
    float midScale;         // offset 152
    float midWindSpeedMul;  // offset 156
    float midBaseHeight;    // offset 160
    float midThickness;     // offset 164
    float highCoverage;     // offset 168
    float highDensity;      // offset 172
    float highScale;        // offset 176
    float highWindSpeedMul; // offset 180
    float highBaseHeight;   // offset 184
    float highThickness;    // offset 188
    float silverLining;     // offset 192
    float ambientBoost;     // offset 196
    float sunForwardG;      // offset 200
    float exposure;         // offset 204
    float timeScale;        // offset 208
    float cloudRaycastPixelSize; // offset 212  one cloud ray per NxN px block
    vec2 invScreenSize;     // offset 216  swapchain inverse size (ray UV)
};

#endif // SKY_UNIFORM_GLSL
