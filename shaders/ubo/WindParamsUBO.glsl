#ifndef WIND_PARAMS_U_B_O_GLSL
#define WIND_PARAMS_U_B_O_GLSL

// Canonical vegetation/impostor wind parameters (std140, 80 B). Single
// definition shared with the CPU: vulkan/ubo/WindParamsUBO.hpp — same names,
// fields and offsets.
// Offset table (verified against the C++ static_asserts):
//   windDirection 0, windStrength 8, windBaseFrequency 12, windSpeed 16,
//   gustFrequency 20, gustStrength 24, skewAmount 28, trunkStiffness 32,
//   noiseScale 36, verticalFlutter 40, turbulence 44, densityEnabled 48,
//   nearDistance 52, farDistance 56, minFactor 60, cameraPosition 64,
//   densityFalloff 76.
// Booleans are uint flags (compare with == 0u / != 0u).
struct WindParamsUBO {
    vec2 windDirection;    // offset  0  normalized wind dir (XZ)
    float windStrength;    // offset  8
    float windBaseFrequency;// offset 12
    float windSpeed;       // offset 16
    float gustFrequency;   // offset 20
    float gustStrength;    // offset 24
    float skewAmount;      // offset 28
    float trunkStiffness;  // offset 32
    float noiseScale;      // offset 36
    float verticalFlutter; // offset 40
    float turbulence;      // offset 44
    uint densityEnabled;   // offset 48  1 = distance thinning on
    float nearDistance;    // offset 52
    float farDistance;     // offset 56
    float minFactor;       // offset 60
    vec3 cameraPosition;   // offset 64
    float densityFalloff;  // offset 76
};

#endif // WIND_PARAMS_U_B_O_GLSL
