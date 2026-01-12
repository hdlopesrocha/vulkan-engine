#include "Light.hpp"
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

Light::Light(const glm::vec3 &dir, const glm::vec3 &col, float intensity)
    : direction(glm::normalize(dir)), color(col), intensity(intensity) {}

void Light::setDirection(const glm::vec3 &dir) {
    direction = glm::normalize(dir);
}

void Light::rotateEuler(float yawDeg, float pitchDeg) {
    // Get current spherical coordinates
    float azimuth, elevation;
    getSpherical(azimuth, elevation);
    
    // Apply rotation
    azimuth += yawDeg;
    elevation += pitchDeg;
    
    // Clamp elevation to avoid gimbal lock
    elevation = glm::clamp(elevation, -89.0f, 89.0f);
    
    // Set new direction
    setFromSpherical(azimuth, elevation);
}

void Light::setFromSpherical(float azimuthDeg, float elevationDeg) {
    float azimuth = glm::radians(azimuthDeg);
    float elevation = glm::radians(elevationDeg);
    
    // Convert spherical to Cartesian coordinates
    // Azimuth: rotation around Y axis (horizontal)
    // Elevation: rotation from horizontal plane (vertical)
    direction.x = std::cos(elevation) * std::sin(azimuth);
    direction.y = std::sin(elevation);
    direction.z = std::cos(elevation) * std::cos(azimuth);
    
    direction = glm::normalize(direction);
}

void Light::getSpherical(float &azimuthDeg, float &elevationDeg) const {
    // Convert Cartesian to spherical coordinates
    glm::vec3 dir = glm::normalize(direction);
    
    // Elevation: angle from horizontal plane
    elevationDeg = glm::degrees(std::asin(dir.y));
    
    // Azimuth: angle in horizontal plane
    azimuthDeg = glm::degrees(std::atan2(dir.x, dir.z));
}

glm::vec3 Light::computeLightPosition(const glm::vec3& camPos, float distance) const {
    glm::vec3 dir = glm::normalize(direction);
    return camPos - dir * distance;
}

glm::mat4 Light::computeLightViewMatrix(const glm::vec3& targetPos) const {
    glm::vec3 dir = glm::normalize(direction);
    glm::vec3 lightPos = targetPos - dir * 100.0f; // arbitrary distance for view matrix
    
    glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
    // Avoid gimbal lock when light is pointing straight up/down
    if (glm::abs(glm::dot(dir, worldUp)) > 0.9f) {
        worldUp = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    
    return glm::lookAt(lightPos, targetPos, worldUp);
}

glm::mat4 Light::computeLightProjectionMatrix(float orthoSize) const {
    return glm::ortho(-orthoSize, orthoSize, -orthoSize, orthoSize, 1.0f, orthoSize * 2.0f);
}

glm::mat4 Light::computeLightSpaceMatrix(const glm::vec3& camPos, float orthoSize) const {
    glm::vec3 dir = glm::normalize(direction);
    glm::vec3 lightPos = camPos - dir * orthoSize * 0.5f;
    
    glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
    if (glm::abs(glm::dot(dir, worldUp)) > 0.9f) {
        worldUp = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    
    glm::mat4 lightView = glm::lookAt(lightPos, camPos, worldUp);
    glm::mat4 lightProjection = glm::ortho(-orthoSize, orthoSize, -orthoSize, orthoSize, 1.0f, orthoSize * 2.0f);
    return lightProjection * lightView;
}

void Light::setTarget(const glm::vec3& target) {
    targetPosition = target;
    viewMatrix = computeLightViewMatrix(targetPosition);
}
