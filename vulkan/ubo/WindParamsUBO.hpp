#pragma once
#include <cstddef>
#include <cstdint>

// Canonical vegetation/impostor wind parameters UBO (std140, 80 B). Single
// definition shared with the GLSL twin shaders/ubo/WindParamsUBO.glsl — same
// names, fields and offsets; bound at set=2 binding=0 and uploaded verbatim
// (write-on-change memcpy) by VegetationRenderer.
// Booleans are uint32 flags (GLSL bool is not a host-shareable block type).
// Member alignment is pinned with alignas to the std140 vector rules
// (vec2 -> 8, vec3 -> 16) so the layout never depends on glm packing.
struct alignas(16) WindParamsUBO {
    alignas(8) glm::vec2 windDirection{0.0f}; // offset  0  normalized wind dir (XZ)
    float windStrength = 0.0f;                // offset  8  amplitude (m)
    float windBaseFrequency = 0.0f;           // offset 12
    float windSpeed = 0.0f;                   // offset 16  gust scroll speed
    float gustFrequency = 0.0f;               // offset 20
    float gustStrength = 0.0f;                // offset 24  0 = laminar
    float skewAmount = 0.0f;                  // offset 28
    float trunkStiffness = 0.0f;              // offset 32
    float noiseScale = 0.0f;                  // offset 36
    float verticalFlutter = 0.0f;             // offset 40
    float turbulence = 0.0f;                  // offset 44
    uint32_t densityEnabled = 0u;             // offset 48  1 = distance thinning on
    float nearDistance = 0.0f;                // offset 52  full-density distance (m)
    float farDistance = 0.0f;                 // offset 56  min-density distance (m)
    float minFactor = 0.0f;                   // offset 60  density floor [0,1]
    alignas(16) glm::vec3 cameraPosition{0.0f}; // offset 64  main camera world pos
    float densityFalloff = 0.0f;              // offset 76  per-metre falloff
    // offset 80 = struct end (multiple of 16; no tail padding needed).
};
static_assert(sizeof(WindParamsUBO) == 80, "WindParamsUBO must be 80 bytes");
static_assert(offsetof(WindParamsUBO, windDirection) == 0, "windDirection offset");
static_assert(offsetof(WindParamsUBO, windStrength) == 8, "windStrength offset");
static_assert(offsetof(WindParamsUBO, windBaseFrequency) == 12, "windBaseFrequency offset");
static_assert(offsetof(WindParamsUBO, windSpeed) == 16, "windSpeed offset");
static_assert(offsetof(WindParamsUBO, gustFrequency) == 20, "gustFrequency offset");
static_assert(offsetof(WindParamsUBO, gustStrength) == 24, "gustStrength offset");
static_assert(offsetof(WindParamsUBO, skewAmount) == 28, "skewAmount offset");
static_assert(offsetof(WindParamsUBO, trunkStiffness) == 32, "trunkStiffness offset");
static_assert(offsetof(WindParamsUBO, noiseScale) == 36, "noiseScale offset");
static_assert(offsetof(WindParamsUBO, verticalFlutter) == 40, "verticalFlutter offset");
static_assert(offsetof(WindParamsUBO, turbulence) == 44, "turbulence offset");
static_assert(offsetof(WindParamsUBO, densityEnabled) == 48, "densityEnabled offset");
static_assert(offsetof(WindParamsUBO, nearDistance) == 52, "nearDistance offset");
static_assert(offsetof(WindParamsUBO, farDistance) == 56, "farDistance offset");
static_assert(offsetof(WindParamsUBO, minFactor) == 60, "minFactor offset");
static_assert(offsetof(WindParamsUBO, cameraPosition) == 64, "cameraPosition offset");
static_assert(offsetof(WindParamsUBO, densityFalloff) == 76, "densityFalloff offset");
