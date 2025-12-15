#include "DebugWidget.hpp"

DebugWidget::DebugWidget(TextureManager* textureManager, Camera* camera, size_t* cubeCount)
    : Widget("Debug"), textureManager(textureManager), camera(camera), cubeCount(cubeCount) {
}

void DebugWidget::render() {
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
#include "DebugWidget.hpp"

// Auto-generated implementation stub for DebugWidget

// Minimal no-op implementation to keep translation unit for build.
#include "DebugWidget.hpp"

// Minimal no-op implementation for DebugWidget
