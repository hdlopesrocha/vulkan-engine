#include "LayersWidget.hpp"
#include "../vulkan/renderer/SceneRenderer.hpp"
#include <imgui.h>
#include <cmath>
#include "components/ImGuiHelpers.hpp"

LayersWidget::LayersWidget(Scene* scene, SceneRenderer* renderer,
                         std::function<void()> onLayersChanged,
                         std::function<void(Layer::Id)> onLayerRepopulate)
    : Widget("Layers", u8"\uf0c9"), scene_(scene), renderer_(renderer),
      onLayersChanged_(std::move(onLayersChanged)),
      onLayerRepopulate_(std::move(onLayerRepopulate)) {}

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
    ImGui::Columns(6, "layers_cols", false);
    ImGui::TextDisabled("#"); ImGui::NextColumn();
    ImGui::TextDisabled("Enabled"); ImGui::NextColumn();
    ImGui::TextDisabled("Name"); ImGui::NextColumn();
    ImGui::TextDisabled("SDF Hue"); ImGui::NextColumn();
    ImGui::TextDisabled("Renderer"); ImGui::NextColumn();
    ImGui::TextDisabled("Chunks"); ImGui::NextColumn();
    ImGui::Separator();

    static const char* kRendererNames[] = { "Solid", "Water", "Video" };
    for (size_t i = 0; i < n; ++i) {
        Layer* layer = scene_->layerAt(i);
        if (!layer) continue;
        const Layer::Id layerId = layer->id();
        ImGui::PushID(static_cast<int>(i));
        ImGui::Text("%zu", i); ImGui::NextColumn();

        bool enabled = layer->enabled();
        if (ImGui::Checkbox("##en", &enabled)) {
            if (!enabled) {
                // Hiding is immediate (slots freed).
                if (renderer_) renderer_->removeAllLayerMeshes(layerId);
                layer->setEnabled(enabled);
            } else {
                layer->setEnabled(enabled);
                // Re-queue the layer's chunks so it renders again without a
                // manual Generate Map.
                if (onLayerRepopulate_) onLayerRepopulate_(layerId);
            }
        }
        ImGui::NextColumn();

        std::string name = layer->name();
        char buf[64];
        snprintf(buf, sizeof(buf), "%s", name.c_str());
        if (ImGui::InputText("##name", buf, sizeof(buf))) {
            layer->setName(buf);
        }
        ImGui::NextColumn();

        // Same stable golden-ratio hue the octree SDF raymarch debug view
        // derives from Layer::Id, so the per-layer colours in that view can
        // be matched back to this list without a separate legend.
        {
            const float hue = fmodf(static_cast<float>(layerId) * 0.618033988749895f, 1.0f);
            float r = 0.0f, g = 0.0f, b = 0.0f;
            ImGui::ColorConvertHSVtoRGB(hue, 0.68f, 0.95f, r, g, b);
            ImGui::ColorButton("##sdfhue", ImVec4(r, g, b, 1.0f),
                               ImGuiColorEditFlags_NoTooltip |
                               ImGuiColorEditFlags_NoPicker |
                               ImGuiColorEditFlags_NoDragDrop,
                               ImVec2(16.0f, 16.0f));
            ImGuiHelpers::SetTooltipIfHovered(
                "Hue this layer uses in the 'Octree SDF Raymarch (per layer)' debug view.");
        }
        ImGui::NextColumn();

        int cur = static_cast<int>(layer->renderer());
        if (cur < 0 || cur > 2) cur = 0;
        if (ImGui::Combo("##renderer", &cur, kRendererNames, 3)) {
            const auto next = static_cast<LayerRendererType>(cur);
            // Free the layer's slots from the OLD pool before remapping so
            // nothing renders twice, then re-queue into the new pool.
            if (renderer_) renderer_->removeAllLayerMeshes(layerId);
            layer->setRenderer(next);
            if (onLayerRepopulate_) onLayerRepopulate_(layerId);
        }
        ImGui::NextColumn();

        const size_t chunks = renderer_ ? renderer_->getLayerModelCount(layerId) : 0;
        ImGui::Text("%zu", chunks);
        ImGui::NextColumn();
        ImGui::PopID();
    }
    ImGui::Columns(1);
    ImGui::Separator();

    ImGui::TextWrapped(
        "Disabling a layer frees its GPU meshes immediately; re-enabling or "
        "switching renderers repopulates it automatically (progressively). "
        "Removing drops the LAST layer only, keeping earlier indices stable.");
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
        Layer* added = scene_->addLayer(nm, LayerRendererType::Solid);
        newLayerName_.clear();
        if (renderer_ && added) renderer_->ensureLayerStates(static_cast<size_t>(added->id()) + 1);
        if (onLayersChanged_) onLayersChanged_();
    }
    ImGui::SameLine();
    const bool canRemove = (n > 1);
    if (!canRemove) ImGui::BeginDisabled();
    if (ImGui::Button("- Remove Last")) {
        if (n > 1) {
            Layer* last = scene_->layerAt(n - 1);
            if (last) {
                const Layer::Id lastId = last->id();
                if (renderer_) renderer_->removeAllLayerMeshes(lastId);
                scene_->removeLayer(*last);
                if (onLayersChanged_) onLayersChanged_();
            }
        }
    }
    if (!canRemove) ImGui::EndDisabled();
}
