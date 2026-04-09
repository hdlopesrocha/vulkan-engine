#pragma once

#include <glm/glm.hpp>
#include "../math/Light.hpp"
#include "../Uniforms.hpp"

struct ShadowParams {
    float orthoSize = 1024.0f;
    // Cascade multipliers: ortho0 = base, ortho1 = 4x, ortho2 = 16x
    static constexpr float cascadeMultipliers[SHADOW_CASCADE_COUNT] = { 1.0f, 4.0f, 16.0f };
    glm::mat4 lightSpaceMatrix[SHADOW_CASCADE_COUNT];
    
    // Update shadow matrices for all cascades based on camera position and light
    void update(const glm::vec3& camPos, Light& light) {
        for (int i = 0; i < SHADOW_CASCADE_COUNT; i++) {
            float cascadeOrtho = orthoSize * cascadeMultipliers[i];
            light.setTarget(camPos, cascadeOrtho);
            light.setProjection(light.computeLightProjectionMatrix(cascadeOrtho));
            lightSpaceMatrix[i] = light.getViewProjectionMatrix();
        }
    }
};
