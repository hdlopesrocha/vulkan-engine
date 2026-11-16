#pragma once
#include <cstddef>
#include <cstdint>

#include <glm/glm.hpp>

// Wind-trail particle parameters UBO (std140, 208 B). Single definition shared
// with the GLSL twin shaders/ubo/WindParticlesUBO.glsl — same names, fields and
// offsets; uploaded verbatim (memcpy) by WindRenderer once per frame.
//
// The block is read by BOTH the particle simulation
// (shaders/computation/wind/WindParticles.comp) and the trail draw
// (shaders/renderer/wind/WindTrail.vert/.frag), so the two can never diverge:
// the draw ages the trail samples with the same clock the simulation wrote.
//
// Member alignment is pinned with alignas to the std140 rules (mat4/vec4 -> 16)
// so the layout never depends on glm packing.
struct alignas(16) WindParticlesUBO {
    glm::mat4 viewProjection{1.0f};    // offset   0  world -> clip (frustum test)
    glm::mat4 invViewProjection{1.0f}; // offset  64  clip -> world (respawn inside)
    // x = shared wind clock (s, mainTime), y = frame dt (s), z = particle count,
    // w = spawn/frame seed (varies every dispatch so respawns are not identical).
    alignas(16) glm::vec4 timing{0.0f};
    // x = trail duration (s), y = fixed trail sample interval (s),
    // z = trail ring capacity (samples), w = trail width (px).
    alignas(16) glm::vec4 trail{0.0f};
    // x = speed at which the blue->green->yellow->red ramp reaches red (m/s),
    // y = peak opacity, zw reserved (0).
    alignas(16) glm::vec4 style{0.0f};
    // x = spawn near distance (m), y = spawn far plane (m, clamped to the
    // camera far plane on the host), z = viewport width (px), w = height (px).
    alignas(16) glm::vec4 spawn{0.0f};
    // xyz = camera world position (ray origin for respawns + far-plane test),
    // w = unused (0).
    alignas(16) glm::vec4 cameraPosition{0.0f};
    // x = frustum test margin (1 = exact, >1 keeps particles slightly outside),
    // y = fade-in seconds after a (re)spawn, z = wind enabled flag, w reserved (0).
    alignas(16) glm::vec4 extra{0.0f};
};
static_assert(sizeof(WindParticlesUBO) == 224, "WindParticlesUBO must be 224 bytes");
static_assert(sizeof(WindParticlesUBO) % 16 == 0, "WindParticlesUBO must be a multiple of 16");
static_assert(offsetof(WindParticlesUBO, viewProjection) == 0, "viewProjection offset");
static_assert(offsetof(WindParticlesUBO, invViewProjection) == 64, "invViewProjection offset");
static_assert(offsetof(WindParticlesUBO, timing) == 128, "timing offset");
static_assert(offsetof(WindParticlesUBO, trail) == 144, "trail offset");
static_assert(offsetof(WindParticlesUBO, style) == 160, "style offset");
static_assert(offsetof(WindParticlesUBO, spawn) == 176, "spawn offset");
static_assert(offsetof(WindParticlesUBO, cameraPosition) == 192, "cameraPosition offset");
static_assert(offsetof(WindParticlesUBO, extra) == 208, "extra offset");
