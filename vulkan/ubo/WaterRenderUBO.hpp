#pragma once
#include <glm/glm.hpp>

// GPU-side water render UBO
struct WaterRenderUBO {
    glm::vec4 timeParams; // x = waterTime, yzw = unused
};
