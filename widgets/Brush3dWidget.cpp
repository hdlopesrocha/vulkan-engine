#include "Brush3dWidget.hpp"
#include "../vulkan/TextureArrayManager.hpp"
#include <imgui.h>
#include <cstring>

// Static label arrays for combo boxes
const char* Brush3dWidget::sdfTypeNames[] = {
    "Sphere", "Box", "Capsule", "Octahedron", "Pyramid", "Torus", "Cone", "Cylinder"
};
const char* Brush3dWidget::brushModeNames[] = { "Add", "Remove" };
const char* Brush3dWidget::layerNames[] = { "Opaque", "Transparent" };
const char* Brush3dWidget::effectTypeNames[] = {
    "Perlin Distort", "Perlin Carve", "Sine Distort", "Voronoi Carve"
};

Brush3dWidget::Brush3dWidget(TextureArrayManager* texMgr, uint32_t loadedLayers)
    : Widget("Brush 3D"),
      textureArrayManager(texMgr),
      loadedTextureLayers(loadedLayers)
{
}

void Brush3dWidget::render() {
    if (!isOpen) return;

    ImGui::SetNextWindowSize(ImVec2(460, 600), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(title.c_str(), &isOpen)) {
        ImGui::End();
        return;
    }

    // Add / remove entry buttons
    if (ImGui::Button("+ Add Brush Entry")) {
        entries.emplace_back();
        dirty = true;
    }
    ImGui::SameLine();
    if (!entries.empty() && ImGui::Button("- Remove Last")) {
        entries.pop_back();
        dirty = true;
    }

    ImGui::Separator();

    // Render each entry in a collapsible header
    for (int i = 0; i < static_cast<int>(entries.size()); ++i) {
        ImGui::PushID(i);
        char label[64];
        snprintf(label, sizeof(label), "Entry %d: %s (%s)", i,
                 sdfTypeNames[entries[i].sdfType],
                 layerNames[entries[i].targetLayer]);
        if (ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_DefaultOpen)) {
            renderEntry(i);
        }
        ImGui::PopID();
    }

    ImGui::Separator();

    // Apply button triggers rebuild only when dirty
    bool canApply = dirty;
    if (!canApply) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button("Apply Brush", ImVec2(-1, 30))) {
        if (rebuildCallback) {
            rebuildCallback();
        }
        dirty = false;
    }
    if (!canApply) {
        ImGui::EndDisabled();
    }

    ImGui::End();
}

void Brush3dWidget::renderEntry(int index) {
    BrushEntry& e = entries[index];

    // SDF Type
    if (ImGui::Combo("SDF Type", &e.sdfType, sdfTypeNames, IM_ARRAYSIZE(sdfTypeNames))) {
        dirty = true;
    }

    // Brush Mode
    if (ImGui::Combo("Mode", &e.brushMode, brushModeNames, IM_ARRAYSIZE(brushModeNames))) {
        dirty = true;
    }

    // Target Layer
    if (ImGui::Combo("Layer", &e.targetLayer, layerNames, IM_ARRAYSIZE(layerNames))) {
        dirty = true;
    }

    // Material picker with thumbnail
    renderMaterialPicker(e);

    // Transform
    if (ImGui::DragFloat3("Translate", &e.translate.x, 1.0f, -100000.0f, 100000.0f)) {
        dirty = true;
    }
    if (ImGui::DragFloat3("Scale", &e.scale.x, 1.0f, 0.1f, 100000.0f)) {
        dirty = true;
    }
    if (ImGui::DragFloat("Yaw", &e.yaw, 1.0f, -360.0f, 360.0f)) dirty = true;
    if (ImGui::DragFloat("Pitch", &e.pitch, 1.0f, -360.0f, 360.0f)) dirty = true;
    if (ImGui::DragFloat("Roll", &e.roll, 1.0f, -360.0f, 360.0f)) dirty = true;

    // Min size
    if (ImGui::DragFloat("Min Size", &e.minSize, 1.0f, 0.1f, 1000.0f)) {
        dirty = true;
    }

    // Capsule-specific parameters
    if (e.sdfType == 2) { // Capsule
        ImGui::Text("Capsule Parameters:");
        if (ImGui::DragFloat3("Point A", &e.capsuleA.x, 1.0f)) dirty = true;
        if (ImGui::DragFloat3("Point B", &e.capsuleB.x, 1.0f)) dirty = true;
        if (ImGui::DragFloat("Radius", &e.capsuleRadius, 0.1f, 0.01f, 10000.0f)) dirty = true;
    }

    // Torus-specific parameters
    if (e.sdfType == 5) { // Torus
        ImGui::Text("Torus Parameters:");
        if (ImGui::DragFloat("Major Radius", &e.torusRadii.x, 0.01f, 0.01f, 1.0f)) dirty = true;
        if (ImGui::DragFloat("Minor Radius", &e.torusRadii.y, 0.01f, 0.01f, 1.0f)) dirty = true;
    }

    // Effect
    if (ImGui::Checkbox("Use Effect", &e.useEffect)) dirty = true;
    if (e.useEffect) {
        ImGui::Indent();
        if (ImGui::Combo("Effect", &e.effectType, effectTypeNames, IM_ARRAYSIZE(effectTypeNames))) {
            dirty = true;
        }
        if (ImGui::DragFloat("Amplitude", &e.effectAmplitude, 0.5f, 0.0f, 1000.0f)) dirty = true;
        if (ImGui::DragFloat("Frequency", &e.effectFrequency, 0.0001f, 0.0001f, 1.0f, "%.4f")) dirty = true;

        // PerlinCarve threshold
        if (e.effectType == 1) {
            if (ImGui::DragFloat("Threshold", &e.effectThreshold, 0.01f, 0.0f, 1.0f)) dirty = true;
        }
        // Voronoi cell size
        if (e.effectType == 3) {
            if (ImGui::DragFloat("Cell Size", &e.effectCellSize, 1.0f, 1.0f, 1000.0f)) dirty = true;
        }
        if (ImGui::DragFloat("Brightness", &e.effectBrightness, 0.01f, -1.0f, 1.0f)) dirty = true;
        if (ImGui::DragFloat("Contrast", &e.effectContrast, 0.01f, -2.0f, 2.0f)) dirty = true;

        ImGui::Unindent();
    }

    // Delete button
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.1f, 0.1f, 1.0f));
    char delLabel[32];
    snprintf(delLabel, sizeof(delLabel), "Delete##%d", index);
    if (ImGui::Button(delLabel)) {
        entries.erase(entries.begin() + index);
        dirty = true;
    }
    ImGui::PopStyleColor();
}

void Brush3dWidget::renderMaterialPicker(BrushEntry& entry) {
    if (!textureArrayManager) {
        ImGui::Text("Material: %d (no texture manager)", entry.materialIndex);
        return;
    }

    ImGui::Text("Material:");
    ImGui::SameLine();

    // Show current material thumbnail
    int maxMat = static_cast<int>(loadedTextureLayers);
    if (maxMat <= 0) maxMat = 1;

    // Thumbnail of current material
    if (entry.materialIndex < maxMat) {
        ImTextureID texId = textureArrayManager->getImTexture(
            static_cast<size_t>(entry.materialIndex), 0); // 0 = albedo
        if (texId) {
            ImGui::Image(texId, ImVec2(32, 32));
            ImGui::SameLine();
        }
    }

    // Slider for quick selection
    if (ImGui::SliderInt("##matIdx", &entry.materialIndex, 0, maxMat - 1)) {
        dirty = true;
    }

    // Scrollable material picker with thumbnails
    if (ImGui::TreeNode("Browse Materials")) {
        float thumbSize = 48.0f;
        float windowWidth = ImGui::GetContentRegionAvail().x;
        int cols = static_cast<int>(windowWidth / (thumbSize + 8.0f));
        if (cols < 1) cols = 1;

        for (int i = 0; i < maxMat; ++i) {
            ImTextureID texId = textureArrayManager->getImTexture(static_cast<size_t>(i), 0);
            if (!texId) continue;

            if (i % cols != 0) ImGui::SameLine();

            ImGui::PushID(i);
            bool selected = (i == entry.materialIndex);
            if (selected) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 1.0f, 1.0f));
            }
            if (ImGui::ImageButton("##matBtn", texId, ImVec2(thumbSize, thumbSize))) {
                entry.materialIndex = i;
                dirty = true;
            }
            if (selected) {
                ImGui::PopStyleColor();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Material %d", i);
            }
            ImGui::PopID();
        }
        ImGui::TreePop();
    }
}
