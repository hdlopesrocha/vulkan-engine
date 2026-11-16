#pragma once
#include <cstddef>

#include <glm/glm.hpp>

// WindRaymarchRenderer parameters (std140, 112 B). Single definition shared
// with the GLSL twin shaders/ubo/WindRaymarchUBO.glsl — same names, fields and
// offsets; uploaded verbatim (memcpy) once per frame by WindRaymarchRenderer.
//
// The fragment shader reconstructs the view ray from invViewProjection and the
// scene depth, then marches |windSVF(p, time)| looking for the target speed.
struct alignas(16) WindRaymarchUBO {
    glm::mat4 invViewProjection{1.0f}; // offset   0  clip -> world
    // x = target speed (m/s), y = max march distance (m), z = steps, w = opacity.
    alignas(16) glm::vec4 surface{0.0f};
    // x = viewport width (px), y = viewport height (px), z = shared wind clock (s),
    // w = unused (0).
    alignas(16) glm::vec4 viewport{0.0f};
    // xyz = camera world position (ray origin), w = unused (0).
    alignas(16) glm::vec4 cameraPosition{0.0f};
};
static_assert(sizeof(WindRaymarchUBO) == 112, "WindRaymarchUBO must be 112 bytes");
static_assert(sizeof(WindRaymarchUBO) % 16 == 0, "WindRaymarchUBO must be a multiple of 16");
static_assert(offsetof(WindRaymarchUBO, invViewProjection) == 0, "invViewProjection offset");
static_assert(offsetof(WindRaymarchUBO, surface) == 64, "surface offset");
static_assert(offsetof(WindRaymarchUBO, viewport) == 80, "viewport offset");
static_assert(offsetof(WindRaymarchUBO, cameraPosition) == 96, "cameraPosition offset");
