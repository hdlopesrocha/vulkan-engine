#pragma once

// Must be defined before GLM to use Vulkan's [0,1] depth range in projection matrices
#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif

#include <glm/glm.hpp>

class Light {
public:
    Light(const glm::vec3 &dir, const glm::vec3 &col = glm::vec3(1.0f, 1.0f, 1.0f), float intensity_ = 1.0f);

    // Getters
    glm::vec3 getDirection() const { return glm::normalize(direction); }
    glm::vec3 getColor() const { return color; }
    float getIntensity() const { return intensity; }
    
    // Setters
    void setDirection(const glm::vec3 &dir);
    void setColor(const glm::vec3 &col) { color = col; }
    void setIntensity(float i) { intensity = i; }
    
    // Set direction from spherical coordinates (useful for UI control)
    void setFromSpherical(float azimuthDeg, float elevationDeg);
    
    // Get spherical coordinates from current direction
    void getSpherical(float &azimuthDeg, float &elevationDeg) const;
    
    // Projection and view-projection matrix (for shadow mapping)
    void setProjection(const glm::mat4 &proj) { projection = proj; }
    void setViewMatrix(const glm::mat4 &vm) { viewMatrix = vm; }
    glm::mat4 getViewMatrix() const { return viewMatrix; }
    glm::mat4 getViewProjectionMatrix() const { return projection * viewMatrix; }

private:
    glm::vec3 direction;
    glm::vec3 color;
    float intensity;
    glm::mat4 projection = glm::mat4(1.0f);
    glm::mat4 viewMatrix = glm::mat4(1.0f);
};
