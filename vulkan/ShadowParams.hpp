#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct ShadowParams {
    float orthoSize = 1024.0f;
    glm::vec3 lightPos = glm::vec3(0.0f);
    glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::mat4 lightView = glm::mat4(1.0f);
    glm::mat4 lightProjection = glm::mat4(1.0f);
    glm::mat4 lightSpaceMatrix = glm::mat4(1.0f);
    
    // Update shadow matrices based on camera position and light direction
    void update(const glm::vec3& camPos, const glm::vec3& lightDirection) {
        glm::vec3 shadowLightDir = glm::normalize(lightDirection);
        lightPos = camPos - shadowLightDir * orthoSize * 0.5f;
        
        worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
        if (glm::abs(glm::dot(shadowLightDir, worldUp)) > 0.9f) {
            worldUp = glm::vec3(1.0f, 0.0f, 0.0f);
        }
        
        lightView = glm::lookAt(lightPos, camPos, worldUp);
        lightProjection = glm::ortho(-orthoSize, orthoSize, -orthoSize, orthoSize, 1.0f, orthoSize * 2.0f);
        lightSpaceMatrix = lightProjection * lightView;
    }
};
