#include "DebugWidget.hpp"
#include "components/ImGuiHelpers.hpp"

DebugWidget::DebugWidget(std::vector<MaterialProperties>* materials, Camera* camera, size_t* cubeCount, VegetationRenderer* vegetationRenderer)
    : Widget("Debug", u8"\uf188"), materials(materials), camera(camera), cubeCount(cubeCount), vegetationRenderer(vegetationRenderer) {
}

void DebugWidget::render() {
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;

    size_t texCount = materials ? materials->size() : 0;
    ImGui::Text("Loaded texture triples: %zu", texCount);
    ImGui::Text("Rendered cubes: %zu", *cubeCount);
    glm::vec3 camPos = camera->getPosition();
    ImGui::Text("Camera pos: %.2f %.2f %.2f", camPos.x, camPos.y, camPos.z);
    ImGui::Text("Cube grid spacing: %.1f", 2.5f);
    ImGui::Text("Grid layout: 4x3 cubes");

    if (vegetationRenderer) {
        ImGui::Separator();
        ImGui::Checkbox("Show Vegetation Density Debug", &showVegetationDensityDebug);
        ImGui::SetItemTooltip("Render colored debug cubes at vegetation chunk centers. Green = dense, red = sparse.");
        ImGui::Text("Vegetation chunks: %zu", vegetationRenderer->getChunkCount());
        ImGui::Text("Avg vegetation density: %.2f", vegetationRenderer->getAverageDensityFactor(camPos));
    }
}
