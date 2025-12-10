#pragma once

#include "Widget.hpp"
#include "../vulkan/Camera.hpp"
#include "../vulkan/TextureManager.hpp"
#include <imgui.h>
#include <cstddef>

class DebugWidget : public Widget {
public:
    DebugWidget(TextureManager* textureManager, Camera* camera, size_t* cubeCount)
        : Widget("Debug"), textureManager(textureManager), camera(camera), cubeCount(cubeCount) {}
    
    void render() override {
        if (!ImGui::Begin(title.c_str(), &isOpen)) {
            ImGui::End();
            return;
        }
        
        ImGui::Text("Loaded texture triples: %zu", textureManager->count());
        ImGui::Text("Rendered cubes: %zu", *cubeCount);
        glm::vec3 camPos = camera->getPosition();
        ImGui::Text("Camera pos: %.2f %.2f %.2f", camPos.x, camPos.y, camPos.z);
        ImGui::Text("Cube grid spacing: %.1f", 2.5f);
        ImGui::Text("Grid layout: 4x3 cubes");
        
        ImGui::End();
    }
    
private:
    TextureManager* textureManager;
    Camera* camera;
    size_t* cubeCount;
};
