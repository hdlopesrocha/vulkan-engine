#pragma once

#include <array>
#include <cstdint>

#include <glm/glm.hpp>

// Stable proxy source per main-scene chunk, recorded at publish time from the
// chunk geometry (world bounds + dominant material). Opaque chunks feed the
// solid BLAS; transparent (water) chunks feed the water BLAS so solid
// reflections see water.
struct SolidProxyData {
    glm::vec3 minp = glm::vec3(0.0f);
    glm::vec3 maxp = glm::vec3(0.0f);
    uint32_t materialId = 0;
    // 4x4 max-height grid over the XZ bounds (row-major, gz*4+gx). Proxy
    // boxes are emitted per grid cell so secondary rays follow the real
    // terrain shape instead of one flat box top per chunk; reflection
    // positions then match the surface to ~cell/4 instead of cell size.
    std::array<float, 16> hgrid = {};
    // 0-based band rung at publish (0 = frontier/finest). Coarse ancestors
    // (rung > 0) nest over the fine boxes and must be EXCLUDED from the
    // proxy set: their huge flat tops would otherwise dominate refraction
    // hits (terraces + giant rectangles) in the far field.
    uint32_t rung = 0;
};
