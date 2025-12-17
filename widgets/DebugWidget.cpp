#include "DebugWidget.hpp"

DebugWidget::DebugWidget(std::vector<MaterialProperties>* materials, Camera* camera, size_t* cubeCount)
    : Widget("Debug"), materials(materials), camera(camera), cubeCount(cubeCount) {
}

void DebugWidget::render() {
    if (!ImGui::Begin(title.c_str(), &isOpen)) {
        ImGui::End();
        return;
    }

    size_t texCount = materials ? materials->size() : 0;
    ImGui::Text("Loaded texture triples: %zu", texCount);
    ImGui::Text("Rendered cubes: %zu", *cubeCount);
    glm::vec3 camPos = camera->getPosition();
    ImGui::Text("Camera pos: %.2f %.2f %.2f", camPos.x, camPos.y, camPos.z);
    ImGui::Text("Cube grid spacing: %.1f", 2.5f);
    ImGui::Text("Grid layout: 4x3 cubes");

    ImGui::End();
}
