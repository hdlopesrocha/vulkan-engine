#include "TextureViewerWidget.hpp"
#include <string>
#include "components/ScrollablePicker.hpp"

void TextureViewer::render() {
    if (!arrayManager || !materials) return;

    if (!ImGui::Begin(title.c_str(), &isOpen)) {
        ImGui::End();
        return;
    }
    
    size_t tc = materials->size();
    if (tc == 0) {
        ImGui::Text("No textures loaded");
        ImGui::End();
        return;
    }

    // clamp currentIndex
    if (currentIndex >= tc) currentIndex = 0;

    // Navigation styled like TextureMixer: "Texture X of Y" with Prev/Next and direct index input
    ImGui::PushID(0);
    ImGui::Text("Texture %zu of %zu", currentIndex + 1, tc);
    ImGui::SameLine(); if (ImGui::Button("Prev") && currentIndex > 0) --currentIndex;
    ImGui::SameLine(); if (ImGui::Button("Next") && currentIndex + 1 < tc) ++currentIndex;
    ImGui::SameLine(); int idxInput = static_cast<int>(currentIndex);
    ImGui::NewLine();
    ImGui::Text("Choose Albedo (click thumbnail)");
    {
        size_t idx = currentIndex;
        if (ImGuiComponents::ScrollableTexturePicker("PickerAlbedo", arrayManager ? arrayManager->layerAmount : 0, idx, [this](size_t l){ return arrayManager ? arrayManager->getImTexture(l, 0) : nullptr; }, 48.0f, false, true)) {
            currentIndex = idx;
        }
    }


    if (ImGui::InputInt("##tex_idx", &idxInput)) {
        if (idxInput < 0) idxInput = 0;
        if (static_cast<size_t>(idxInput) >= tc) idxInput = static_cast<int>(tc - 1);
        currentIndex = static_cast<size_t>(idxInput);
    }
    ImGui::PopID();

    // Compact pickers live in the Albedo/Normal/Height tabs below — remove the duplicated top-row picker
    ImGui::Separator();
    ImGui::Text("Use the tabs below to pick Albedo / Normal / Height thumbnails.");
    ImGui::Separator();

    std::string tabBarId = std::string("tabs_") + std::to_string(currentIndex);
    if (ImGui::BeginTabBar(tabBarId.c_str())) {
        const float previewSize = 256.0f; // 25% of previous size
        const float thumb = 48.0f; // thumbnail size for compact picker
        if (ImGui::BeginTabItem("Albedo")) {
            ImTextureID tex = arrayManager->getImTexture(currentIndex, 0);
            if (tex) {
                ImGui::Image(tex, ImVec2(previewSize, previewSize));
            } else {
                ImGui::Text("Texture preview not available");
                if (ImGui::Button("Recreate descriptor")) {
                    arrayManager->getImTexture(currentIndex, 0); // attempt to force creation
                }
                ImGui::SameLine();
                if (ImGui::Button("Log info")) {
                    fprintf(stderr, "[TextureViewer] Preview NULL: layer=%zu map=Albedo layerInitialized=%d layerAmount=%u albedo.image=%p albedoSampler=%p\n",
                        currentIndex,
                        arrayManager->isLayerInitialized(static_cast<uint32_t>(currentIndex)) ? 1 : 0,
                        arrayManager->layerAmount,
                        (void*)arrayManager->albedoArray.image,
                        (void*)arrayManager->albedoSampler);
                }
            }

            ImGui::Separator();
            ImGui::Text("Choose Albedo (click thumbnail)");
            {
                size_t idx = currentIndex;
                if (ImGuiComponents::ScrollableTexturePicker("PickerAlbedo", arrayManager ? arrayManager->layerAmount : 0, idx, [this](size_t l){ return arrayManager ? arrayManager->getImTexture(l, 0) : nullptr; }, 48.0f, 2, true, true)) {
                    currentIndex = idx;
                }
            }

            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Normal")) {
            ImTextureID tex = arrayManager->getImTexture(currentIndex, 1);
            if (tex) {
                ImGui::Image(tex, ImVec2(previewSize, previewSize));
            } else {
                ImGui::Text("Texture preview not available");
                if (ImGui::Button("Recreate descriptor")) { arrayManager->getImTexture(currentIndex, 1); }
                ImGui::SameLine();
                if (ImGui::Button("Log info")) {
                    fprintf(stderr, "[TextureViewer] Preview NULL: layer=%zu map=Normal layerInitialized=%d layerAmount=%u normal.image=%p normalSampler=%p\n",
                        currentIndex,
                        arrayManager->isLayerInitialized(static_cast<uint32_t>(currentIndex)) ? 1 : 0,
                        arrayManager->layerAmount,
                        (void*)arrayManager->normalArray.image,
                        (void*)arrayManager->normalSampler);
                }
            }

            ImGui::Separator();
            ImGui::Text("Choose Normal (click thumbnail)");
            {
                size_t idx = currentIndex;
                if (ImGuiComponents::ScrollableTexturePicker("PickerNormal", arrayManager ? arrayManager->layerAmount : 0, idx, [this](size_t l){ return arrayManager ? arrayManager->getImTexture(l, 1) : nullptr; }, 48.0f, 2, true, true)) {
                    currentIndex = idx;
                }
            }

            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Height")) {
            ImTextureID tex = arrayManager->getImTexture(currentIndex, 2);
            if (tex) {
                ImGui::Image(tex, ImVec2(previewSize, previewSize));
            } else {
                ImGui::Text("Texture preview not available");
                if (ImGui::Button("Recreate descriptor")) { arrayManager->getImTexture(currentIndex, 2); }
                ImGui::SameLine();
                if (ImGui::Button("Log info")) {
                    fprintf(stderr, "[TextureViewer] Preview NULL: layer=%zu map=Height layerInitialized=%d layerAmount=%u bump.image=%p bumpSampler=%p\n",
                        currentIndex,
                        arrayManager->isLayerInitialized(static_cast<uint32_t>(currentIndex)) ? 1 : 0,
                        arrayManager->layerAmount,
                        (void*)arrayManager->bumpArray.image,
                        (void*)arrayManager->bumpSampler);
                }
            }

            ImGui::Separator();
            ImGui::Text("Choose Height (click thumbnail)");
            {
                size_t idx = currentIndex;
                if (ImGuiComponents::ScrollableTexturePicker("PickerHeight", arrayManager ? arrayManager->layerAmount : 0, idx, [this](size_t l){ return arrayManager ? arrayManager->getImTexture(l, 2) : nullptr; }, 48.0f, 2, true, true)) {
                    currentIndex = idx;
                }
            }

            ImGui::EndTabItem();
        }
        // Material tab removed from tabs — material UI will be shown under the preview
        ImGui::EndTabBar();

        // Material controls shown under the preview tabs
        {
            MaterialProperties& mat = (*materials)[currentIndex];
            bool dirty = false;

            ImGui::Spacing();
            ImGui::Text("Material");
            ImGui::Separator();

            ImGui::Text("Normal/Tangent Adjustments");
            ImGui::Separator();

            // Normal map conventions
            bool flipY = mat.normalFlipY;
            if (ImGui::Checkbox("Flip Normal Y (green)", &flipY)) { mat.normalFlipY = flipY; dirty = true; }
            bool swapXZ = mat.normalSwapXZ;
            if (ImGui::Checkbox("Swap Normal X/Z (R <-> B)", &swapXZ)) { mat.normalSwapXZ = swapXZ; dirty = true; }

            bool triplanarBool = mat.triplanar;
            if (ImGui::Checkbox("Enable Triplanar Mapping", &triplanarBool)) { mat.triplanar = triplanarBool; dirty = true; }
            if (mat.triplanar) {
                ImGui::Indent();
                if (ImGui::SliderFloat("Triplanar Scale U", &mat.triplanarScaleU, 0.01f, 10.0f, "%.3f")) dirty = true;
                if (ImGui::SliderFloat("Triplanar Scale V", &mat.triplanarScaleV, 0.01f, 10.0f, "%.3f")) dirty = true;
                ImGui::Unindent();
            }

            ImGui::Spacing();
            ImGui::Text("Lighting");
            ImGui::Separator();

            if (ImGui::SliderFloat("Ambient Factor", &mat.ambientFactor, 0.0f, 1.0f, "%.2f")) dirty = true;
            if (ImGui::SliderFloat("Specular Strength", &mat.specularStrength, 0.0f, 2.0f, "%.2f")) dirty = true;
            if (ImGui::SliderFloat("Shininess", &mat.shininess, 1.0f, 256.0f, "%.0f")) dirty = true;

            ImGui::Spacing();
            ImGui::Text("Mapping (Tessellation + Bump)");
            ImGui::Separator();
            bool mappingEnabled = mat.mappingMode;
            if (ImGui::Checkbox("Enable Tessellation & Bump Mapping", &mappingEnabled)) { mat.mappingMode = mappingEnabled; dirty = true; }
            bool heightDirect = mat.invertHeight;
            if (ImGui::Checkbox("Height Is Direct (white=high)", &heightDirect)) { mat.invertHeight = heightDirect; dirty = true; }
            if (mat.mappingMode) {
                int tess = static_cast<int>(mat.tessLevel + 0.5f);
                if (ImGui::SliderInt("Tessellation Level", &tess, 1, 64)) { mat.tessLevel = static_cast<float>(tess); dirty = true; }
                if (ImGui::SliderFloat("Tess Height Scale", &mat.tessHeightScale, 0.0f, 1.0f, "%.3f")) dirty = true;
            }

            if (dirty && onMaterialChanged) onMaterialChanged(currentIndex);
        }
    }

    ImGui::End();
}

TextureViewer::TextureViewer() : Widget("Textures") {}

void TextureViewer::init(TextureArrayManager* arrayManager, std::vector<MaterialProperties>* materials) {
    this->arrayManager = arrayManager;
    this->materials = materials;
}
