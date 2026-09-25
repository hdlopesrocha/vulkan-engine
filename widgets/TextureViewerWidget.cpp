#include "TextureViewerWidget.hpp"

#include <string>
#include <iostream>
#include <cfloat>
#include "components/ColumnLayout.hpp"
#include "components/ScrollablePicker.hpp"
#include "../services/TextureMixer.hpp"
#include "components/ImGuiHelpers.hpp"

namespace {
constexpr float kPreviewSize = 256.0f;
constexpr float kPreviewColumnWidth = 360.0f;
}

void TextureViewer::render() {
    using namespace ImGuiComponents;

    if (!arrayManager || !materials) return;

    ImGui::SetNextWindowPos(ImVec2(0, 24), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(1280, 680), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(320, 300), ImVec2(FLT_MAX, FLT_MAX));
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen,
        ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    if (!wg.visible()) return;

    size_t tc = materials->size();
    if (tc == 0) {
        ImGui::Text("No textures loaded");
        return;
    }
    if (currentIndex >= tc) currentIndex = 0;

    bool materialDirty = false;

    std::vector<ColumnSection> sections;
    sections.reserve(5);

    sections.push_back({[&]() {
        ImGui::Text("Texture");
        ColSeparator();
        ImGui::Text("Texture %zu of %zu", currentIndex + 1, tc);
        if (ImGui::Button("Prev") && currentIndex > 0) --currentIndex;
        ImGui::SameLine();
        if (ImGui::Button("Next") && currentIndex + 1 < tc) ++currentIndex;
        FieldLabel("Texture Index", "Texture layer to inspect (0-based).");
        ImGui::SetNextItemWidth(kColumnWidth);
        ImGui::PushID("tex_idx");
        int idxInput = static_cast<int>(currentIndex);
        if (ImGui::InputInt("##v", &idxInput)) {
            if (idxInput < 0) idxInput = 0;
            if (static_cast<size_t>(idxInput) >= tc) idxInput = static_cast<int>(tc - 1);
            currentIndex = static_cast<size_t>(idxInput);
        }
        ImGui::PopID();
        ImGui::TextWrapped("Use the Preview tabs to pick Albedo / Normal / Height / Roughness / AO thumbnails.");
    }});

    sections.push_back({[&]() {
        ImGui::Text("Preview");
        ColSeparator(kPreviewColumnWidth);
        static std::string tabBarId;
        static size_t tabBarIdIndex = static_cast<size_t>(-1);
        if (tabBarIdIndex != currentIndex) {
            tabBarIdIndex = currentIndex;
            tabBarId = std::string("tabs_") + std::to_string(currentIndex);
        }
        ImGui::PushStyleVar(ImGuiStyleVar_TabBarBorderSize, 0.0f);
        if (ImGui::BeginTabBar(tabBarId.c_str())) {
            if (ImGui::BeginTabItem("Albedo")) {
                ImTextureID tex = arrayManager->getImTexture(currentIndex, 0);
                if (tex) {
                    ImGui::Image(tex, ImVec2(kPreviewSize, kPreviewSize));
                } else {
                    ImGui::Text("Texture preview not available");
                    if (ImGui::Button("Recreate descriptor")) {
                        arrayManager->getImTexture(currentIndex, 0);
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Log info")) {
                        std::cerr << "[TextureViewer] Preview NULL: layer=" << currentIndex
                                  << " map=Albedo"
                                  << " layerInitialized=" << (arrayManager->isLayerInitialized(static_cast<uint32_t>(currentIndex)) ? 1 : 0)
                                  << " layerAmount=" << arrayManager->layerAmount
                                  << " albedo.image=" << (void*)arrayManager->albedoArray.image
                                  << " albedoSampler=" << (void*)arrayManager->albedoSampler
                                  << std::endl;
                    }
                }

                ImGui::Spacing();
                ColSeparator();
                ImGui::Spacing();
                FieldLabel("Choose Albedo", "Click a thumbnail to select this texture layer.");
                size_t idx = currentIndex;
                if (ScrollableTexturePicker("PickerAlbedo", arrayManager->layerAmount, idx,
                        [this](size_t l) { return arrayManager->getImTexture(l, 0); },
                        48.0f, 2, true, true, kColumnWidth)) {
                    currentIndex = idx;
                    if (textureMixer) {
                        MixerParameters mp{};
                        mp.targetLayer = currentIndex;
                        mp.primaryTextureIdx = static_cast<uint32_t>(currentIndex);
                        mp.secondaryTextureIdx = static_cast<uint32_t>(currentIndex);
                        textureMixer->enqueueGenerate(mp, 0);
                    }
                }
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Normal")) {
                ImTextureID tex = arrayManager->getImTexture(currentIndex, 1);
                if (tex) {
                    ImGui::Image(tex, ImVec2(kPreviewSize, kPreviewSize));
                } else {
                    ImGui::Text("Texture preview not available");
                    if (ImGui::Button("Recreate descriptor")) {
                        arrayManager->getImTexture(currentIndex, 1);
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Log info")) {
                        std::cerr << "[TextureViewer] Preview NULL: layer=" << currentIndex
                                  << " map=Normal"
                                  << " layerInitialized=" << (arrayManager->isLayerInitialized(static_cast<uint32_t>(currentIndex)) ? 1 : 0)
                                  << " layerAmount=" << arrayManager->layerAmount
                                  << " normal.image=" << (void*)arrayManager->normalArray.image
                                  << " normalSampler=" << (void*)arrayManager->normalSampler
                                  << std::endl;
                    }
                }

                ImGui::Spacing();
                ColSeparator();
                ImGui::Spacing();
                FieldLabel("Choose Normal", "Click a thumbnail to select this texture layer.");
                size_t idx = currentIndex;
                if (ScrollableTexturePicker("PickerNormal", arrayManager->layerAmount, idx,
                        [this](size_t l) { return arrayManager->getImTexture(l, 1); },
                        48.0f, 2, true, true, kColumnWidth)) {
                    currentIndex = idx;
                }
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Height")) {
                ImTextureID tex = arrayManager->getImTexture(currentIndex, 2);
                if (tex) {
                    ImGui::Image(tex, ImVec2(kPreviewSize, kPreviewSize));
                } else {
                    ImGui::Text("Texture preview not available");
                    if (ImGui::Button("Recreate descriptor")) {
                        arrayManager->getImTexture(currentIndex, 2);
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Log info")) {
                        std::cerr << "[TextureViewer] Preview NULL: layer=" << currentIndex
                                  << " map=Height"
                                  << " layerInitialized=" << (arrayManager->isLayerInitialized(static_cast<uint32_t>(currentIndex)) ? 1 : 0)
                                  << " layerAmount=" << arrayManager->layerAmount
                                  << " bump.image=" << (void*)arrayManager->bumpArray.image
                                  << " bumpSampler=" << (void*)arrayManager->bumpSampler
                                  << std::endl;
                    }
                }

                ImGui::Spacing();
                ColSeparator();
                ImGui::Spacing();
                FieldLabel("Choose Height", "Click a thumbnail to select this texture layer.");
                size_t idx = currentIndex;
                if (ScrollableTexturePicker("PickerHeight", arrayManager->layerAmount, idx,
                        [this](size_t l) { return arrayManager->getImTexture(l, 2); },
                        48.0f, 2, true, true, kColumnWidth)) {
                    currentIndex = idx;
                }
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Roughness")) {
                ImTextureID tex = arrayManager->getImTexture(currentIndex, 3);
                if (tex) {
                    ImGui::Image(tex, ImVec2(kPreviewSize, kPreviewSize));
                } else {
                    ImGui::Text("Texture preview not available");
                    if (ImGui::Button("Recreate descriptor")) {
                        arrayManager->getImTexture(currentIndex, 3);
                    }
                }

                ImGui::Spacing();
                ColSeparator();
                ImGui::Spacing();
                FieldLabel("Choose Roughness", "Click a thumbnail to select this texture layer.");
                size_t idx = currentIndex;
                if (ScrollableTexturePicker("PickerRoughness", arrayManager->layerAmount, idx,
                        [this](size_t l) { return arrayManager->getImTexture(l, 3); },
                        48.0f, 2, true, true, kColumnWidth)) {
                    currentIndex = idx;
                }
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("AO")) {
                ImTextureID tex = arrayManager->getImTexture(currentIndex, 4);
                if (tex) {
                    ImGui::Image(tex, ImVec2(kPreviewSize, kPreviewSize));
                } else {
                    ImGui::Text("Texture preview not available");
                    if (ImGui::Button("Recreate descriptor")) {
                        arrayManager->getImTexture(currentIndex, 4);
                    }
                }

                ImGui::Spacing();
                ColSeparator();
                ImGui::Spacing();
                FieldLabel("Choose AO", "Click a thumbnail to select this texture layer.");
                size_t idx = currentIndex;
                if (ScrollableTexturePicker("PickerAO", arrayManager->layerAmount, idx,
                        [this](size_t l) { return arrayManager->getImTexture(l, 4); },
                        48.0f, 2, true, true, kColumnWidth)) {
                    currentIndex = idx;
                }
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::PopStyleVar();
    }, kPreviewColumnWidth});

    sections.push_back({[&]() {
        ImGui::Text("Normal / Tangent Adjustments");
        ColSeparator();
        MaterialProperties& mat = (*materials)[currentIndex];
        if (CheckboxField("Flip Normal Y (green)", &mat.normalFlipY)) materialDirty = true;
        if (CheckboxField("Swap Normal X/Z (R <-> B)", &mat.normalSwapXZ)) materialDirty = true;
        if (CheckboxField("Enable Triplanar Mapping", &mat.triplanar)) materialDirty = true;
        if (mat.triplanar) {
            if (SliderFloatField("Triplanar Scale U", &mat.triplanarScaleU, 0.01f, 10.0f, "%.3f")) materialDirty = true;
            if (SliderFloatField("Triplanar Scale V", &mat.triplanarScaleV, 0.01f, 10.0f, "%.3f")) materialDirty = true;
        }
    }});

    sections.push_back({[&]() {
        ImGui::Text("Lighting");
        ColSeparator();
        MaterialProperties& mat = (*materials)[currentIndex];
        if (SliderFloatField("Ambient Factor", &mat.ambientFactor, 0.0f, 1.0f, "%.2f")) materialDirty = true;
        if (SliderFloatField("Specular Strength", &mat.specularStrength, 0.0f, 2.0f, "%.2f")) materialDirty = true;
        if (SliderFloatField("Shininess", &mat.shininess, 1.0f, 256.0f, "%.0f")) materialDirty = true;
        if (SliderFloatField("Reflection Strength", &mat.reflectionStrength, 0.0f, 1.0f, "%.2f")) materialDirty = true;
        if (SliderFloatField("Roughness Factor", &mat.roughnessFactor, 0.0f, 1.0f, "%.2f")) materialDirty = true;
        if (SliderFloatField("AO Factor", &mat.aoFactor, 0.0f, 1.0f, "%.2f")) materialDirty = true;
        if (CheckboxField("Enable Ambient Occlusion", &mat.useAO)) materialDirty = true;
    }});

    sections.push_back({[&]() {
        ImGui::Text("Mapping (Tessellation + Bump)");
        ColSeparator();
        MaterialProperties& mat = (*materials)[currentIndex];
        if (CheckboxField("Enable Tessellation & Bump Mapping", &mat.mappingMode)) materialDirty = true;
        if (CheckboxField("Invert Height (mirror V)", &mat.invertHeight)) materialDirty = true;
        if (CheckboxField("Invert Width (mirror U)", &mat.invertWidth)) materialDirty = true;
        if (mat.mappingMode) {
            int tess = static_cast<int>(mat.tessLevel + 0.5f);
            if (SliderIntField("Tessellation Level", &tess, 1, 64)) {
                mat.tessLevel = static_cast<float>(tess);
                materialDirty = true;
            }
            if (SliderFloatField("Tess Height Scale", &mat.tessHeightScale, 0.0f, 64.0f, "%.3f")) materialDirty = true;
            ImGui::TextUnformatted("Distance-based range");
            int minLvl = static_cast<int>(mat.tessMinLevel + 0.5f);
            int maxLvl = static_cast<int>(mat.tessMaxLevel + 0.5f);
            if (SliderIntField("Tess Min Level", &minLvl, 1, 64)) {
                mat.tessMinLevel = static_cast<float>(minLvl);
                materialDirty = true;
            }
            if (SliderIntField("Tess Max Level", &maxLvl, 1, 64)) {
                mat.tessMaxLevel = static_cast<float>(maxLvl);
                materialDirty = true;
            }
        }
    }});

    static std::vector<float> cachedH;
    const std::vector<float> estimates = {
        EstimateSectionHeight(4),
        500.0f,
        EstimateSectionHeight(4),
        EstimateSectionHeight(7),
        EstimateSectionHeight(8),
    };
    LayoutSections(sections, cachedH, estimates);

    if (materialDirty && onMaterialChanged) onMaterialChanged(currentIndex);
}

TextureViewer::TextureViewer() : Widget("Textures", u8"\uf03e") {}

void TextureViewer::init(TextureArrayManager* arrayManager_, std::vector<MaterialProperties>* materials_) {
    this->arrayManager = arrayManager_;
    this->materials = materials_;
}
