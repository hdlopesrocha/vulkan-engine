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

// Material texture indices map to array layers: -1 means "own layer" and the
// engine arrays are bounded to 256 layers.
int clampTextureIndex(int value) {
    if (value < -1) return -1;
    if (value > 255) return 255;
    return value;
}

// Labelled combo over an enum range. `items` must hold `count` persistent
// names, mirroring the Slider*Field helpers in ColumnLayout.hpp.
bool EnumComboField(const char* id, const char* label, const char* desc,
                    const char* const* items, int count, int* value) {
    ImGuiComponents::FieldLabel(label, desc);
    ImGui::SetNextItemWidth(ImGuiComponents::kColumnWidth);
    ImGui::PushID(id);
    const bool changed = ImGui::Combo("##v", value, items, count);
    ImGui::PopID();
    ImGuiComponents::TooltipOnHover(desc);
    return changed;
}

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
    sections.reserve(6);

    sections.push_back({[&]() {
        ImGui::Text("Material");
        ColSeparator();
        MaterialProperties& mat = (*materials)[currentIndex];

        constexpr int kMaterialKindCount = static_cast<int>(MaterialKind::Water) + 1;
        const char* kindItems[kMaterialKindCount];
        for (int i = 0; i < kMaterialKindCount; ++i)
            kindItems[i] = materialKindName(static_cast<MaterialKind>(i));
        int kind = static_cast<int>(mat.kind);
        if (EnumComboField("mat_kind", "Material Type",
                "What the material samples: a scene texture layer, a video array slot, or the water palette.",
                kindItems, kMaterialKindCount, &kind)) {
            mat.kind = static_cast<MaterialKind>(kind);
            materialDirty = true;
        }

        if (mat.kind == MaterialKind::Video) {
            std::vector<std::string> slotNames;
            if (videoSlotNamesProvider) slotNames = videoSlotNamesProvider();
            if (!slotNames.empty()) {
                std::vector<const char*> slotItems;
                slotItems.reserve(slotNames.size());
                for (const std::string& name : slotNames) slotItems.push_back(name.c_str());
                // Keep the raw index: -1/out-of-range shows an empty preview but
                // still lets the user pick slot 0 (which must report a change).
                int slot = mat.textureIndex;
                if (EnumComboField("mat_video_slot", "Video Slot",
                        "Slot of the video texture array sampled by this material.",
                        slotItems.data(), static_cast<int>(slotItems.size()), &slot)) {
                    mat.textureIndex = slot;
                    materialDirty = true;
                }
            } else {
                FieldLabel("Video Slot (index)", "No video slot names available; enter the video array slot directly.");
                ImGui::SetNextItemWidth(kColumnWidth);
                ImGui::PushID("mat_video_slot_idx");
                int slot = mat.textureIndex;
                if (ImGui::InputInt("##v", &slot)) {
                    mat.textureIndex = clampTextureIndex(slot);
                    materialDirty = true;
                }
                ImGui::PopID();
            }
        } else {
            FieldLabel("Texture Index (-1 = own layer)",
                "Scene texture layer sampled by this material; -1 uses the material's own layer.");
            ImGui::SetNextItemWidth(kColumnWidth);
            ImGui::PushID("mat_texture_idx");
            int index = mat.textureIndex;
            if (ImGui::InputInt("##v", &index)) {
                mat.textureIndex = clampTextureIndex(index);
                materialDirty = true;
            }
            ImGui::PopID();
        }
        ImGui::Text("Selected %s: %d", mat.kind == MaterialKind::Video ? "slot" : "layer", mat.textureIndex);

        constexpr int kTextureMappingCount = static_cast<int>(TextureMappingMode::Custom) + 1;
        const char* mappingItems[kTextureMappingCount];
        for (int i = 0; i < kTextureMappingCount; ++i)
            mappingItems[i] = textureMappingName(static_cast<TextureMappingMode>(i));
        int mapping = static_cast<int>(mat.textureMapping);
        if (EnumComboField("mat_mapping", "Texture Coordinates",
                "Texture coordinate technique: triplanar, an axis projection, an SDF sphere or mesh UVs.",
                mappingItems, kTextureMappingCount, &mapping)) {
            mat.textureMapping = static_cast<TextureMappingMode>(mapping);
            materialDirty = true;
        }
        if (mat.textureMapping == TextureMappingMode::SdfGiven) {
            FieldLabel("SDF Center", "World-space center of the sphere used by SDF Given mapping.");
            ImGui::SetNextItemWidth(kColumnWidth);
            ImGui::PushID("mat_sdf_center");
            if (ImGui::DragFloat3("##v", &mat.mappingOrigin.x, 1.0f)) materialDirty = true;
            ImGui::PopID();

            FieldLabel("SDF Radius", "Sphere radius for SDF Given mapping; 0 keeps the default world-space sphere.");
            ImGui::SetNextItemWidth(kColumnWidth);
            ImGui::PushID("mat_sdf_radius");
            if (ImGui::DragFloat("##v", &mat.mappingOrigin.w, 0.05f, 0.0f, FLT_MAX, "%.3f")) materialDirty = true;
            ImGui::PopID();
        }

        constexpr int kBlendModeCount = static_cast<int>(MaterialBlendMode::Luminosity) + 1;
        const char* blendItems[kBlendModeCount];
        for (int i = 0; i < kBlendModeCount; ++i)
            blendItems[i] = materialBlendModeName(static_cast<MaterialBlendMode>(i));
        int blend = static_cast<int>(mat.blendMode);
        if (EnumComboField("mat_blend", "Blend Mode",
                "Photoshop-style blend mode applied when this material is composited over the scene.",
                blendItems, kBlendModeCount, &blend)) {
            mat.blendMode = static_cast<MaterialBlendMode>(blend);
            materialDirty = true;
        }
    }});

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
        EstimateSectionHeight(7),
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
