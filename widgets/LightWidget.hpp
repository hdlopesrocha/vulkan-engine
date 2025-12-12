#pragma once

#include "Widget.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <imgui.h>

class LightWidget : public Widget {
private:
    glm::vec3* lightDirection;
    
    // Spherical coordinates in degrees
    float azimuth = 45.0f;   // Horizontal angle (0° = North/+Z, 90° = East/+X)
    float elevation = 45.0f; // Vertical angle (0° = horizon, 90° = zenith)
    
    // Convert spherical coordinates to direction vector
    void updateLightDirection() {
        if (!lightDirection) return;
        
        // Convert degrees to radians
        float azimuthRad = glm::radians(azimuth);
        float elevationRad = glm::radians(elevation);
        
        // Calculate direction TO the light source (where light comes from)
        // Azimuth: 0° = +Z (North), 90° = +X (East), 180° = -Z (South), 270° = -X (West)
        // Elevation: 0° = horizon, 90° = zenith (straight up), -90° = nadir (straight down)
        glm::vec3 direction;
        direction.x = cos(elevationRad) * sin(azimuthRad);
        direction.y = sin(elevationRad);
        direction.z = cos(elevationRad) * cos(azimuthRad);
        
        *lightDirection = glm::normalize(direction);
    }
    
    // Calculate spherical coordinates from direction vector (for initialization)
    void calculateAnglesFromDirection() {
        if (!lightDirection) return;
        
        glm::vec3 dir = glm::normalize(*lightDirection);
        
        // Calculate elevation (angle from horizontal plane)
        elevation = glm::degrees(asin(dir.y));
        
        // Calculate azimuth (angle in horizontal plane)
        azimuth = glm::degrees(atan2(dir.x, dir.z));
    }
    
public:
    LightWidget(glm::vec3* lightDir) 
        : Widget("Light Control"), lightDirection(lightDir) {
        if (lightDirection) {
            calculateAnglesFromDirection();
        }
    }
    
    void render() override {
        if (!isOpen) return;
        
        if (ImGui::Begin(title.c_str(), &isOpen)) {
            // Show and allow editing the light direction vector (normalized when edited)
            float dir[3] = { lightDirection->x, lightDirection->y, lightDirection->z };
            if (ImGui::InputFloat3("Direction", dir, "%.3f")) {
                glm::vec3 newDir = glm::normalize(glm::vec3(dir[0], dir[1], dir[2]));
                *lightDirection = newDir;
                // Update sliders to reflect the new direction
                calculateAnglesFromDirection();
            }
            ImGui::Text("Normalized: (%.3f, %.3f, %.3f)", lightDirection->x, lightDirection->y, lightDirection->z);
            ImGui::Text("Azimuth: %.1f°, Elevation: %.1f°", azimuth, elevation);
            
            ImGui::Separator();
            
            if (ImGui::SliderFloat("Azimuth", &azimuth, -180.0f, 180.0f, "%.1f°")) {
                updateLightDirection();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Horizontal angle\n0° = North (+Z), 90° = East (+X), ±180° = South (-Z), -90° = West (-X)");
            }
            
            if (ImGui::SliderFloat("Elevation", &elevation, -89.0f, 89.0f, "%.1f°")) {
                updateLightDirection();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Vertical angle\n0° = Horizon, +90° = Zenith (straight up), -90° = Nadir (straight down)");
            }
            
            ImGui::Separator();
            ImGui::Text("Presets:");
            
            if (ImGui::Button("Top-Down")) {
                azimuth = 0.0f;
                elevation = 89.0f;
                updateLightDirection();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Light from directly above (zenith)");
            }
            
            ImGui::SameLine();
            if (ImGui::Button("Diagonal")) {
                azimuth = 45.0f;
                elevation = 45.0f;
                updateLightDirection();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("45° elevation from northeast");
            }
            
            ImGui::SameLine();
            if (ImGui::Button("Side")) {
                azimuth = 90.0f;
                elevation = 0.0f;
                updateLightDirection();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Light from the east (side, horizon)");
            }
            
            if (ImGui::Button("Front")) {
                azimuth = 0.0f;
                elevation = 0.0f;
                updateLightDirection();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Light from the north (front, horizon)");
            }
            
            ImGui::SameLine();
            if (ImGui::Button("Back")) {
                azimuth = 180.0f;
                elevation = 0.0f;
                updateLightDirection();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Light from the south (back, horizon)");
            }
            
            ImGui::SameLine();
            if (ImGui::Button("Low Angle")) {
                azimuth = 45.0f;
                elevation = 15.0f;
                updateLightDirection();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Low angle light for long shadows");
            }
        }
        ImGui::End();
    }
    
    // Allow external access to angles if needed
    float getAzimuth() const { return azimuth; }
    float getElevation() const { return elevation; }
    
    void setAzimuth(float a) { azimuth = a; updateLightDirection(); }
    void setElevation(float e) { elevation = e; updateLightDirection(); }
};
