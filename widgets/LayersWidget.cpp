#include "LayersWidget.hpp"
#include "../vulkan/renderer/SceneRenderer.hpp"
#include <imgui.h>
#include "components/ImGuiHelpers.hpp"

LayersWidget::LayersWidget(Scene* scene, SceneRenderer* renderer, std::function<void()> onLayersChanged)
    : Widget("Layers", u8"\uf0c9"), scene_(scene), renderer_(renderer), onLayersChanged_(std::move(onLayersChanged)) {}

void LayersWidget::render() {
    if (!scene_) return;
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;

    const size_t n = scene_->layerCount();
    ImGui::Text("Scene layers: %zu", n);
    ImGuiHelpers::SetTooltipIfHovered(
        "Each layer is an independent octree. The renderer column decides which GPU pipeline "
        "its chunks publish into (Solid = opaque, Water = transparent).");

    // Header
    ImGui::Separator();
    ImGui::Columns(5, "layers_cols", false);
    ImGui::TextDisabled("#"); ImGui::NextColumn();
    ImGui::TextDisabled("Enabled"); ImGui::NextColumn();
    ImGui::TextDisabled("Name"); ImGui::NextColumn();
    ImGui::TextDisabled("Renderer"); ImGui::NextColumn();
    ImGui::TextDisabled("Chunks"); ImGui::NextColumn();
    ImGui::Separator();

    static const char* kRendererNames[] = { "Solid", "Water" };
    for (size_t i = 0; i < n; ++i) {
        const Layer layer = static_cast<Layer>(i);
        ImGui::PushID(static_cast<int>(i));
        ImGui::Text("%zu", i); ImGui::NextColumn();

        bool enabled = scene_->layerEnabled(layer);
        if (ImGui::Checkbox("##en", &enabled)) {
            if (!enabled && renderer_) renderer_->removeAllLayerMeshes(layer);
            scene_->setLayerEnabled(layer, enabled);
            if (!enabled) {
                // Hiding is immediate (slots freed); re-enabling needs a
                // Generate Map / reload to repopulate (documented below).
            }
        }
        ImGui::NextColumn();

        std::string name = scene_->layerName(layer);
        char buf[64];
        snprintf(buf, sizeof(buf), "%s", name.c_str());
        if (ImGui::InputText("##name", buf, sizeof(buf))) {
            scene_->setLayerName(layer, buf);
        }
        ImGui::NextColumn();

        int cur = (scene_->layerRenderer(layer) == LayerRendererType::Water) ? 1 : 0;
        if (ImGui::Combo("##renderer", &cur, kRendererNames, 2)) {
            const auto next = (cur == 1) ? LayerRendererType::Water : LayerRendererType::Solid;
            // Free the layer's slots from the OLD pool before remapping so
            // nothing renders twice; repopulation needs Generate Map.
            if (renderer_) renderer_->removeAllLayerMeshes(layer);
            scene_->setLayerRenderer(layer, next);
        }
        ImGui::NextColumn();

        const size_t chunks = renderer_ ? renderer_->getLayerModelCount(layer) : 0;
        ImGui::Text("%zu", chunks);
        ImGui::NextColumn();
        ImGui::PopID();
    }
    ImGui::Columns(1);
    ImGui::Separator();

    ImGui::TextWrapped(
        "Renderer switches and re-enabling a layer need File > Generate Map (or reload) "
        "to repopulate GPU meshes. Removing drops the LAST layer only, keeping "
        "earlier indices stable.");
    ImGui::Spacing();

    if (!allowStructureEdit_) {
        ImGui::TextDisabled("Add/remove disabled in remote mode (server owns the layers).");
        return;
    }
    // Add layer row
    char addBuf[64];
    snprintf(addBuf, sizeof(addBuf), "%s", newLayerName_.c_str());
    ImGui::InputTextWithHint("##newname", "New layer name", addBuf, sizeof(addBuf));
    newLayerName_ = addBuf;
    ImGui::SameLine();
    if (ImGui::Button("+ Add Layer")) {
        std::string nm = newLayerName_.empty()
            ? ("Layer " + std::to_string(n)) : newLayerName_;
        scene_->addLayer(nm, LayerRendererType::Solid);
        newLayerName_.clear();
        if (renderer_) renderer_->ensureLayerStates(scene_->layerCount());
        if (onLayersChanged_) onLayersChanged_();
    }
    ImGui::SameLine();
    const bool canRemove = (n > 1);
    if (!canRemove) ImGui::BeginDisabled();
    if (ImGui::Button("- Remove Last")) {
        if (n > 1) {
            const Layer last = static_cast<Layer>(n - 1);
            if (renderer_) renderer_->removeAllLayerMeshes(last);
            scene_->removeLayer(last);
            if (onLayersChanged_) onLayersChanged_();
        }
    }
    if (!canRemove) ImGui::EndDisabled();
}
