#pragma once

#include <glm/glm.hpp>

// CPU-side proxy source (chunk bounds already known by the scene graph).
struct RTProxyBox {
    glm::vec3 minp = glm::vec3(0.0f);
    glm::vec3 maxp = glm::vec3(0.0f);
    float materialId = 0.0f;
    float flags = 0.0f; // bit0 = water volume
    glm::vec3 albedo = glm::vec3(0.5f);
    float roughness = 0.9f;
};
