#pragma once

#include "Widget.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <imgui.h>
#include <iostream>

class LightWidget : public Widget {
private:
    glm::vec3* lightDirection;
    
    // Spherical coordinates in degrees
    float azimuth = 45.0f;   // Horizontal angle (0° = North/+Z, 90° = East/+X)
    float elevation = 45.0f; // Vertical angle (0° = horizon, 90° = zenith)
    
    // Convert spherical coordinates to direction vector
    void updateLightDirection();
    void calculateAnglesFromDirection();
    
public:
    LightWidget(glm::vec3* lightDir);

    void render() override;

    // Allow external access to angles if needed
    float getAzimuth() const;
    float getElevation() const;

    void setAzimuth(float a);
    void setElevation(float e);
};
