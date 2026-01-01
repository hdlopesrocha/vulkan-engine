#pragma once

#include <glm/glm.hpp>
#include "../math/Light.hpp"

struct ShadowParams {
    float orthoSize = 1024.0f;
    glm::mat4 lightSpaceMatrix = glm::mat4(1.0f);
    
    // Update shadow matrices based on camera position and light
    void update(const glm::vec3& camPos, const Light& light) {
        lightSpaceMatrix = light.computeLightSpaceMatrix(camPos, orthoSize);
    }
};
