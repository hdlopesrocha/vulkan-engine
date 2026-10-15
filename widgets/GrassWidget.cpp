#include "GrassWidget.hpp"

#include <imgui.h>

#include "../vulkan/renderer/vegetation/VegetationRenderer.hpp"
#include "components/ImGuiHelpers.hpp"
#include "components/ColumnLayout.hpp"

GrassWidget::GrassWidget(VegetationRenderer* vegRenderer_)
    : Widget("Grass", u8"\uf4d8"), vegRenderer(vegRenderer_) {
    isOpen = false;
}

void GrassWidget::render() {
    if (!vegRenderer) return;
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;

    ImGui::TextWrapped("Bladed grass: each instance is a tuft of 8 triangle "
                       "blades (procedural root-to-tip gradient, wind-bent in "
                       "the vertex shader). Beyond the impostor distance the "
                       "tuft is replaced by a captured impostor quad.");
    ImGui::Spacing();
    ImGui::Text("chunks: %zu   instances: %zu",
                vegRenderer->getChunkCount(), vegRenderer->getInstanceTotal());
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Tufts", ImGuiTreeNodeFlags_DefaultOpen)) {
        float scale = vegRenderer->billboardScale;
        if (ImGuiComponents::SliderFloatField("Tuft scale", &scale, 1.0f, 30.0f, "%.1f",
                "World scale of a blade tuft (blade height/width/footprint). "
                "Re-captured impostors use the same scale.")) {
            vegRenderer->billboardScale = scale;
        }
    }

    if (ImGui::CollapsingHeader("Impostors", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Impostor hand-off lives in VegetationRenderer (also mirrored to
        // Settings by MyApp each frame); editing here previews the swap.
        // The widget reads the renderer's live value via a getter is not
        // available, so it edits Settings-level distance through the renderer
        // field directly is not possible — keep a local mirror is overkill.
        // Instead expose nothing structural here: the hand-off is
        // Settings > Impostor Distance. Wind is in the Wind widget.
        ImGui::TextDisabled("Far LOD: Settings > Impostor Distance. Wind: Wind widget.");
    }
}
