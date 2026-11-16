#ifndef WIND_PARAMS_U_B_O_GLSL
#define WIND_PARAMS_U_B_O_GLSL

// Canonical vegetation/impostor wind parameters (std140, 80 B). Single
// definition shared with the CPU: vulkan/ubo/WindParamsUBO.hpp — same names,
// fields and offsets.
// Offset table (verified against the C++ static_asserts):
//   windStrength 0, windBaseFrequency 4, windSpeed 8, gustFrequency 12,
//   gustStrength 16, skewAmount 20, trunkStiffness 24, noiseScale 28,
//   verticalFlutter 32, turbulence 36, densityEnabled 40, nearDistance 44,
//   farDistance 48, minFactor 52, windLeanSpeed 56, windSwirlVelocity 60,
//   cameraPosition 64, densityFalloff 76.
// There is no wind-direction lane: the shared wind velocity is the 3D fBm
// field and the sway direction is extracted from it at each blade.
// Booleans are uint flags (compare with == 0u / != 0u).
struct WindParamsUBO {
    float windStrength;    // offset  0  sway bend amount (m)
    float windBaseFrequency;// offset  4
    float windSpeed;       // offset  8  sway noise scroll
    float gustFrequency;   // offset 12
    float gustStrength;    // offset 16  0 = laminar
    float skewAmount;      // offset 20
    float trunkStiffness;  // offset 24
    float noiseScale;      // offset 28
    float verticalFlutter; // offset 32
    float turbulence;      // offset 36
    uint densityEnabled;   // offset 40  1 = distance thinning on
    float nearDistance;    // offset 44
    float farDistance;     // offset 48
    float minFactor;       // offset 52
    float windLeanSpeed;   // offset 56  wind speed for ~63% lean (m/s)
    float windSwirlVelocity;// offset 60  swirl threshold (m/s)
    vec3 cameraPosition;   // offset 64
    float densityFalloff;  // offset 76
};

#endif // WIND_PARAMS_U_B_O_GLSL
