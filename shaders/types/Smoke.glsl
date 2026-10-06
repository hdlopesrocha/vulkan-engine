#ifndef SMOKE_GLSL
#define SMOKE_GLSL

// Canonical smoke tuning (std430, 176 B). Single definition shared with the
// CPU (widget/renderer): sdf/types/Smoke.hpp — identical layout, names
// and offsets.
// Offset table (verified against the C++ static_asserts):
//   growthDuration 0, loopDuration 4, dissipation 8, noiseScale 12,
//   noiseStrength 16, noiseWarp 20, wind 24, densityScale 32,
//   tunnelStrength 36, tunnelFalloff 40, wakeStrength 44, wakeDissipation 48,
//   wakeRadius 52, wakeExpansion 56, wakeLength 60, pressureRadius 64,
//   pressureStrength 68, rippleAmp 72, rippleFreq 76, turbScale 80,
//   turbStrength 84, turbSpeed 88, shadowSamples 92, shadowStrength 96,
//   heatStrength 100, shape 104, smokeColor 112, goldDeep 128,
//   goldSpecPower 140, goldBright 144, goldPatternScale 156,
//   goldSpecStrength 160, goldFresnelBoost 164, goldWarmFloor 168,
//   goldNormalDistort 172.
// 108..112 and 124..128 are std430 padding (16-byte vector alignment).
// wind is in the smoke shape's LOCAL frame (generic SdfModel transform).
struct Smoke {
    float growthDuration;   // offset   0  expansion window (s)
    float loopDuration;     // offset   4  animation repeat period (s)
    float dissipation;      // offset   8  end-of-loop fade strength
    float noiseScale;       // offset  12  1/m
    float noiseStrength;    // offset  16
    float noiseWarp;        // offset  20
    vec2 wind;              // offset  24  local-frame wind velocity (m/s)
    float densityScale;     // offset  32  live density multiplier
    float tunnelStrength;   // offset  36
    float tunnelFalloff;    // offset  40  m
    float wakeStrength;     // offset  44
    float wakeDissipation;  // offset  48  1/s, doubles as tunnel refill rate
    float wakeRadius;       // offset  52  m
    float wakeExpansion;    // offset  56  wake entrainment coefficient
    float wakeLength;       // offset  60  m
    float pressureRadius;   // offset  64  m
    float pressureStrength; // offset  68  /4 = reference shock push
    float rippleAmp;        // offset  72
    float rippleFreq;       // offset  76
    float turbScale;        // offset  80  1/m
    float turbStrength;     // offset  84
    float turbSpeed;        // offset  88  1/s
    uint shadowSamples;     // offset  92
    float shadowStrength;   // offset  96
    float heatStrength;     // offset 100
    uint shape;             // offset 104  0 cloud, 1 sphere, 2 cube
    vec3 smokeColor;        // offset 112  lit albedo tint
    vec3 goldDeep;          // offset 128  tracer deep gold
    float goldSpecPower;    // offset 140
    vec3 goldBright;        // offset 144  tracer bright gold
    float goldPatternScale; // offset 156
    float goldSpecStrength; // offset 160
    float goldFresnelBoost; // offset 164
    float goldWarmFloor;    // offset 168
    float goldNormalDistort;// offset 172
};

#endif // SMOKE_GLSL
