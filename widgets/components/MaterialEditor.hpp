#pragma once

#include <imgui.h>
#include <cfloat>
#include <string>
#include <vector>

#include "types/MaterialProperties.hpp"
#include "ColumnLayout.hpp"

// Shared MaterialProperties editor. Used identically by the Textures, Texture
// Mixer and Video widgets so all three define materials the same way. The image
// a material/brush samples is selected by the brush map (Brush mapper), not
// here; this component edits the material's kind, Angular origin, UV scale and
// blend mode.
namespace MaterialEditor {

// Renders the material's shader parameters. `videoSlotNames` (may be empty)
// drives the video-slot combo for Video-kind materials. Returns true when any
// field changed. `idPrefix` namespaces the ImGui IDs so the component can be
// embedded in several widgets.
inline bool render(MaterialProperties& mat,
                   const std::vector<std::string>& videoSlotNames,
                   const char* idPrefix) {
    using namespace ImGuiComponents;
    ImGui::PushID(idPrefix);
    bool changed = false;

    constexpr int kMaterialKindCount = static_cast<int>(MaterialKind::Water) + 1;
    const char* kindItems[kMaterialKindCount];
    for (int i = 0; i < kMaterialKindCount; ++i)
        kindItems[i] = materialKindName(static_cast<MaterialKind>(i));
    int kind = static_cast<int>(mat.kind);
    FieldLabel("Material Type",
               "What the material samples: a scene texture layer, a video array slot, or the water palette.");
    ImGui::SetNextItemWidth(kColumnWidth);
    if (ImGui::Combo("##material_kind", &kind, kindItems, kMaterialKindCount)) {
        mat.kind = static_cast<MaterialKind>(kind);
        changed = true;
    }

    if (mat.kind == MaterialKind::Video) {
        if (!videoSlotNames.empty()) {
            std::vector<const char*> slotItems;
            slotItems.reserve(videoSlotNames.size());
            for (const std::string& name : videoSlotNames) slotItems.push_back(name.c_str());
            int slot = mat.textureIndex;
            FieldLabel("Video Slot", "Slot of the video texture array sampled by this material.");
            ImGui::SetNextItemWidth(kColumnWidth);
            if (ImGui::Combo("##material_video_slot", &slot, slotItems.data(),
                             static_cast<int>(slotItems.size()))) {
                mat.textureIndex = slot;
                changed = true;
            }
        } else {
            FieldLabel("Video Slot (index)", "Video array slot sampled by this material.");
            ImGui::SetNextItemWidth(kColumnWidth);
            int slot = mat.textureIndex;
            if (ImGui::InputInt("##material_video_slot_idx", &slot)) {
                mat.textureIndex = slot < 0 ? -1 : (slot > 255 ? 255 : slot);
                changed = true;
            }
        }
    }

    FieldLabel("Angular Center", "World-space center of the sphere used by Angular mapping.");
    ImGui::SetNextItemWidth(kColumnWidth);
    if (ImGui::DragFloat3("##material_angular_center", &mat.mappingOrigin.x, 1.0f)) changed = true;

    FieldLabel("Angular Radius", "Sphere radius for Angular mapping; 0 keeps the default world-space sphere.");
    ImGui::SetNextItemWidth(kColumnWidth);
    if (ImGui::DragFloat("##material_angular_radius", &mat.mappingOrigin.w, 0.05f, 0.0f, FLT_MAX, "%.3f")) changed = true;

    if (SliderFloatField("UV Scale U", &mat.triplanarScaleU, 0.0001f, 10.0f, "%.4f",
                         "Projection UV scale (Triplanar / ProjectionX/Y/Z).")) changed = true;
    if (SliderFloatField("UV Scale V", &mat.triplanarScaleV, 0.0001f, 10.0f, "%.4f",
                         "Projection UV scale (Triplanar / ProjectionX/Y/Z).")) changed = true;

    constexpr int kBlendModeCount = static_cast<int>(MaterialBlendMode::Luminosity) + 1;
    const char* blendItems[kBlendModeCount];
    for (int i = 0; i < kBlendModeCount; ++i)
        blendItems[i] = materialBlendModeName(static_cast<MaterialBlendMode>(i));
    int blend = static_cast<int>(mat.blendMode);
    FieldLabel("Blend Mode",
               "Compositing mode applied against the scene color by the video-layer composite.");
    ImGui::SetNextItemWidth(kColumnWidth);
    if (ImGui::Combo("##material_blend", &blend, blendItems, kBlendModeCount)) {
        mat.blendMode = static_cast<MaterialBlendMode>(blend);
        changed = true;
    }

    ImGui::PopID();
    return changed;
}

} // namespace MaterialEditor
