#pragma once

#include "Widget.hpp"
#include "../utils/Camera.hpp"
#include <imgui.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>

class CameraWidget : public Widget {
public:
    CameraWidget(Camera* camera) : Widget("Camera"), camera(camera) {}
    
    void render() override {
        if (!ImGui::Begin(title.c_str(), &isOpen)) {
            ImGui::End();
            return;
        }
        
        ImGui::DragFloat("Move Speed", &camera->speed, 0.1f, 0.0f, 50.0f);
        float angDeg = glm::degrees(camera->angularSpeedRad);
        if (ImGui::SliderFloat("Angular Speed (deg/s)", &angDeg, 1.0f, 360.0f)) {
            camera->angularSpeedRad = glm::radians(angDeg);
        }
        glm::vec3 pos = camera->getPosition();
        ImGui::Text("Position: %.2f, %.2f, %.2f", pos.x, pos.y, pos.z);
        glm::quat orient = camera->getOrientation();
        glm::vec3 euler = glm::degrees(glm::eulerAngles(orient));
        ImGui::Text("Euler (deg): X=%.1f Y=%.1f Z=%.1f", euler.x, euler.y, euler.z);
        if (ImGui::Button("Reset Orientation")) {
            *camera = Camera(camera->getPosition());
        }
        
        ImGui::End();
    }
    
private:
    Camera* camera;
};
