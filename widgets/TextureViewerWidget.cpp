#include "TextureViewerWidget.hpp"

#include <string>
#include <iostream>
#include <cfloat>
#include "components/ColumnLayout.hpp"
#include "components/ScrollablePicker.hpp"
#include "components/ImGuiHelpers.hpp"

namespace {
constexpr float kPreviewSize = 256.0f;
constexpr float kPreviewColumnWidth = 360.0f;

// One preview tab: label/id, picker wiring and (for maps with array debug
// members) the optional "Log info" diagnostics. The table preserves the
// original per-tab ImGui labels and picker ids exactly.
struct PreviewTab {
    const char* label;
    const char* pickerId;
    const char* chooseLabel;
    int mapIndex;
    const char* logMapName;      // null: this tab has no "Log info" button
    const char* logFieldName;
    TextureImage TextureArrayManager::*array;
    VkSampler TextureArrayManager::*sampler;
};

const PreviewTab kPreviewTabs[] = {
    { "Albedo",    "PickerAlbedo",    "Choose Albedo",    0, "Albedo", "albedo", &TextureArrayManager::albedoArray, &TextureArrayManager::albedoSampler },
    { "Normal",    "PickerNormal",    "Choose Normal",    1, "Normal", "normal", &TextureArrayManager::normalArray, &TextureArrayManager::normalSampler },
    { "Height",    "PickerHeight",    "Choose Height",    2, "Height", "bump",   &TextureArrayManager::bumpArray,   &TextureArrayManager::bumpSampler },
    { "Roughness", "PickerRoughness", "Choose Roughness", 3, nullptr,  nullptr,  nullptr, nullptr },
    { "AO",        "PickerAO",        "Choose AO",        4, nullptr,  nullptr,  nullptr, nullptr },
};
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
            for (const PreviewTab &tab : kPreviewTabs) {
                if (ImGui::BeginTabItem(tab.label)) {
                    const int map = tab.mapIndex;
                    ImTextureID tex = arrayManager->getImTexture(currentIndex, map);
                    if (tex) {
                        ImGui::Image(tex, ImVec2(kPreviewSize, kPreviewSize));
                    } else {
                        ImGui::Text("Texture preview not available");
                        if (ImGui::Button("Recreate descriptor")) {
                            arrayManager->getImTexture(currentIndex, map);
                        }
                        if (tab.logMapName) {
                            ImGui::SameLine();
                            if (ImGui::Button("Log info")) {
                                std::cerr << "[TextureViewer] Preview NULL: layer=" << currentIndex
                                          << " map=" << tab.logMapName
                                          << " layerInitialized=" << (arrayManager->isLayerInitialized(static_cast<uint32_t>(currentIndex)) ? 1 : 0)
                                          << " layerAmount=" << arrayManager->layerAmount
                                          << " " << tab.logFieldName << ".image=" << (void*)(arrayManager->*tab.array).image
                                          << " " << tab.logFieldName << "Sampler=" << (void*)(arrayManager->*tab.sampler)
                                          << std::endl;
                            }
                        }
                    }

                    ImGui::Spacing();
                    ColSeparator();
                    ImGui::Spacing();
                    FieldLabel(tab.chooseLabel, "Click a thumbnail to select this texture layer.");
                    size_t idx = currentIndex;
                    if (ScrollableTexturePicker(tab.pickerId, arrayManager->layerAmount, idx,
                            [this, map](size_t l) { return arrayManager->getImTexture(l, map); },
                            48.0f, 2, true, true, kColumnWidth)) {
                        currentIndex = idx;
                    }
                    ImGui::EndTabItem();
                }
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
