#pragma once

#include "Widget.hpp"
#include "../math/Light.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <imgui.h>
#include <iostream>

class LightWidget : public Widget {
private:
    Light* light;
    
    // Spherical coordinates in degrees
    float azimuth = 45.0f;   // Horizontal angle (0° = North/+Z, 90° = East/+X)
    float elevation = 45.0f; // Vertical angle (0° = horizon, 90° = zenith)
    
    // Light properties
    float color[3] = {1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
    
    // Convert spherical coordinates to direction vector
    void updateLight();
    void calculateAnglesFromDirection();
    
public:
    LightWidget(Light* light);

    void render() override;

    // Allow external access to angles if needed
    float getAzimuth() const;
    float getElevation() const;

    void setAzimuth(float a);
    void setElevation(float e);
};
