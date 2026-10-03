#include "Light.hpp"
#include <glm/gtc/constants.hpp>
#include <cmath>

Light::Light(const glm::vec3 &dir, const glm::vec3 &col, float intensity_)
    : direction(glm::normalize(dir)), color(col), intensity(intensity_) {}

void Light::setDirection(const glm::vec3 &dir) {
    direction = glm::normalize(dir);
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

