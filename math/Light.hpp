#pragma once

#include <glm/glm.hpp>

class Light {
public:
    Light(const glm::vec3 &dir, const glm::vec3 &col = glm::vec3(1.0f, 1.0f, 1.0f), float intensity = 1.0f);

    // Getters
    glm::vec3 getDirection() const { return glm::normalize(direction); }
    glm::vec3 getColor() const { return color; }
    float getIntensity() const { return intensity; }
    
    // Setters
    void setDirection(const glm::vec3 &dir);
    void setColor(const glm::vec3 &col) { color = col; }
    void setIntensity(float i) { intensity = i; }
    
    // Rotate the light direction using spherical coordinates
    void rotateEuler(float yawDeg, float pitchDeg);
    
    // Set direction from spherical coordinates (useful for UI control)
    void setFromSpherical(float azimuthDeg, float elevationDeg);
    
    // Get spherical coordinates from current direction
    void getSpherical(float &azimuthDeg, float &elevationDeg) const;
    
    // Light matrix computation for shadow mapping
    glm::vec3 computeLightPosition(const glm::vec3& camPos, float distance) const;
    glm::mat4 computeLightViewMatrix(const glm::vec3& targetPos) const;
    glm::mat4 computeLightProjectionMatrix(float orthoSize) const;
    glm::mat4 computeLightSpaceMatrix(const glm::vec3& camPos, float orthoSize) const;
    
    // Projection and view-projection matrix (for shadow mapping)
    void setProjection(const glm::mat4 &proj) { projection = proj; }
    glm::mat4 getProjection() const { return projection; }
    void setTarget(const glm::vec3& target);
    glm::mat4 getViewMatrix() const { return viewMatrix; }
    glm::mat4 getViewProjectionMatrix() const { return projection * viewMatrix; }

private:
    glm::vec3 direction;
    glm::vec3 color;
    float intensity;
    glm::vec3 targetPosition = glm::vec3(0.0f);
    glm::mat4 projection = glm::mat4(1.0f);
    glm::mat4 viewMatrix = glm::mat4(1.0f);
};
