#pragma once
#include <cstddef>
#include <cstdint>

#include <glm/glm.hpp>

// Canonical vegetation/impostor wind parameters UBO (std140, 80 B). Single
// definition shared with the GLSL twin shaders/ubo/WindParamsUBO.glsl — same
// names, fields and offsets; bound at set=2 binding=0 and uploaded verbatim
// (write-on-change memcpy) by VegetationRenderer.
//
// The shared-field direction was removed: the wind velocity is the 3D fBm
// field and the per-blade sway direction is extracted from that field at each
// blade (see VegetationRenderer.vert). This block now only carries the sway
// detail plus the distance-density ramp.
// Booleans are uint32 flags (GLSL bool is not a host-shareable block type).
// Member alignment is pinned with alignas to the std140 vector rules
// (vec3 -> 16) so the layout never depends on glm packing.
struct alignas(16) WindParamsUBO {
    float windStrength = 0.0f;                // offset  0  sway bend amount (m)
    float windBaseFrequency = 0.0f;           // offset  4
    float windSpeed = 0.0f;                   // offset  8  sway noise scroll
    float gustFrequency = 0.0f;               // offset 12
    float gustStrength = 0.0f;                // offset 16  0 = laminar
    float skewAmount = 0.0f;                  // offset 20
    float trunkStiffness = 0.0f;              // offset 24
    float noiseScale = 0.0f;                  // offset 28
    float verticalFlutter = 0.0f;             // offset 32
    float turbulence = 0.0f;                  // offset 36
    uint32_t densityEnabled = 0u;             // offset 40  1 = distance thinning on
    float nearDistance = 0.0f;                // offset 44  full-density distance (m)
    float farDistance = 0.0f;                 // offset 48  min-density distance (m)
    float minFactor = 0.0f;                   // offset 52  density floor [0,1]
    // Sway response (see VegetationRenderer.vert applyWindSkew):
    float windLeanSpeed = 6.0f;               // offset 56  wind speed for ~63% lean (m/s)
    float windSwirlVelocity = 12.0f;          // offset 60  swirl threshold (m/s)
    alignas(16) glm::vec3 cameraPosition{0.0f}; // offset 64  main camera world pos
    float densityFalloff = 0.0f;              // offset 76  per-metre falloff
    // offset 80 = struct end (multiple of 16; no tail padding needed).
};
static_assert(sizeof(WindParamsUBO) == 80, "WindParamsUBO must be 80 bytes");
static_assert(offsetof(WindParamsUBO, windStrength) == 0, "windStrength offset");
static_assert(offsetof(WindParamsUBO, windBaseFrequency) == 4, "windBaseFrequency offset");
static_assert(offsetof(WindParamsUBO, windSpeed) == 8, "windSpeed offset");
static_assert(offsetof(WindParamsUBO, gustFrequency) == 12, "gustFrequency offset");
static_assert(offsetof(WindParamsUBO, gustStrength) == 16, "gustStrength offset");
static_assert(offsetof(WindParamsUBO, skewAmount) == 20, "skewAmount offset");
static_assert(offsetof(WindParamsUBO, trunkStiffness) == 24, "trunkStiffness offset");
static_assert(offsetof(WindParamsUBO, noiseScale) == 28, "noiseScale offset");
static_assert(offsetof(WindParamsUBO, verticalFlutter) == 32, "verticalFlutter offset");
static_assert(offsetof(WindParamsUBO, turbulence) == 36, "turbulence offset");
static_assert(offsetof(WindParamsUBO, densityEnabled) == 40, "densityEnabled offset");
static_assert(offsetof(WindParamsUBO, nearDistance) == 44, "nearDistance offset");
static_assert(offsetof(WindParamsUBO, farDistance) == 48, "farDistance offset");
static_assert(offsetof(WindParamsUBO, minFactor) == 52, "minFactor offset");
static_assert(offsetof(WindParamsUBO, windLeanSpeed) == 56, "windLeanSpeed offset");
static_assert(offsetof(WindParamsUBO, windSwirlVelocity) == 60, "windSwirlVelocity offset");
static_assert(offsetof(WindParamsUBO, cameraPosition) == 64, "cameraPosition offset");
static_assert(offsetof(WindParamsUBO, densityFalloff) == 76, "densityFalloff offset");
