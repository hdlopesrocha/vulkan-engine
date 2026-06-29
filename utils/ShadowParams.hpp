#pragma once

#include <glm/glm.hpp>
#include "../math/Light.hpp"
#include "../vulkan/ubo/UniformObject.hpp"

struct ShadowParams {
    float orthoSize = 1024.0f;
    static constexpr float cascadeMultipliers[SHADOW_CASCADE_COUNT] = { 1.0f, 4.0f, 16.0f };
    glm::mat4 lightSpaceMatrix[SHADOW_CASCADE_COUNT];
    
    void update(const glm::vec3& camPos, Light& light) {
        // Compute a single shared view matrix from the LARGEST cascade so that
        // UV coordinates are consistent across all cascades.  Only the
        // orthographic projection changes per cascade.
        float baseSize = orthoSize * cascadeMultipliers[SHADOW_CASCADE_COUNT - 1];
        glm::mat4 sharedView = light.computeLightViewMatrix(camPos, baseSize);
        light.setViewMatrix(sharedView);
        light.setTarget(camPos, baseSize);

        for (int i = 0; i < SHADOW_CASCADE_COUNT; i++) {
            float cascadeOrtho = orthoSize * cascadeMultipliers[i];
            light.setProjection(light.computeLightProjectionMatrix(cascadeOrtho));
            lightSpaceMatrix[i] = light.getViewProjectionMatrix();
        }
    }
};
