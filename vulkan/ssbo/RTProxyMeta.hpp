#pragma once
#include <glm/glm.hpp>

// One stable proxy entry per scene chunk (std430 SSBO element). Canonical
// shared layout, mirrors the GLSL RTProxyMetaGLSL in
// shaders/ssbo/RTProxyMetaGLSL.glsl (same name, field order and offsets).
// The BLAS stores the AABB; this metadata buffer (indexed by primitive ID in
// hit shaders / ray queries) carries the shading data the box alone cannot
// provide.
struct RTProxyMeta {
    glm::vec4 minAndMatId = glm::vec4(0.0f); // xyz = AABB min, w = material id (float)
    glm::vec4 maxAndFlags = glm::vec4(0.0f); // xyz = AABB max, w = flags (bit0 = water volume)
    glm::vec4 albedoRough = glm::vec4(0.5f, 0.5f, 0.5f, 0.9f); // rgb = avg albedo, a = roughness
    glm::vec4 extra = glm::vec4(0.0f); // reserved (metallic, emissive class, ...)
};
