#pragma once
#include <glm/glm.hpp>

// GPU-side water render UBO
struct WaterRenderUBO {
    glm::vec4 timeParams; // x = waterTime, y = water refraction allowed,
                          // z = water reflection allowed, w = unused
};
