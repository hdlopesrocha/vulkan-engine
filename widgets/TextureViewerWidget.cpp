#include "TextureViewerWidget.hpp"

#include <string>
#include <iostream>
#include <algorithm>
#include <cfloat>
#include "components/ColumnLayout.hpp"
#include "components/MaterialEditor.hpp"
#include "components/ScrollablePicker.hpp"
#include "../services/VideoTextureService.hpp"

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

    auto wg = ImGuiComponents::BeginColumnWindow(displayTitle().c_str(), &isOpen);
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
        std::vector<std::string> slotNames;
        if (videoSlotNamesProvider) slotNames = videoSlotNamesProvider();
        if (MaterialEditor::render(mat, slotNames, "tex_mat")) materialDirty = true;
    }});

    sections.push_back({[&]() {
        ImGui::Text("Texture");
        ColSeparator();
        // Base images only: the TextureMixer and video panels own those sources.
        const size_t navCount = (imageCount > 0) ? std::min<size_t>(imageCount, tc) : tc;
        if (currentIndex >= navCount) currentIndex = 0;
        ImGui::Text("Texture %zu of %zu", currentIndex + 1, navCount);
        if (ImGui::Button("Prev") && currentIndex > 0) --currentIndex;
        ImGui::SameLine();
        if (ImGui::Button("Next") && currentIndex + 1 < navCount) ++currentIndex;
        FieldLabel("Base Image", "Base texture layer to inspect (0-based).");
        ImGui::SetNextItemWidth(kColumnWidth);
        ImGui::PushID("tex_idx");
        int idxInput = static_cast<int>(currentIndex);
        if (ImGui::InputInt("##v", &idxInput)) {
            if (idxInput < 0) idxInput = 0;
            if (static_cast<size_t>(idxInput) >= navCount) idxInput = static_cast<int>(navCount - 1);
            currentIndex = static_cast<size_t>(idxInput);
        }
        ImGui::PopID();
        ImGui::TextWrapped("Use the Preview tabs to pick Albedo / Normal / Height / Roughness / AO thumbnails.");

        // ── Video clips ────────────────────────────────────────────────────
        // Videos are first-class texture sources here: their live thumbnails
        // can be assigned to the current material, which turns it into a Video
        // material so the blend mode and lighting controls above apply. Play
        // and loop are exposed next to the assignment so a paused clip is one
        // click away from playing.
        if (videoService && videoService->count() > 0) {
            ImGui::Spacing();
            ColSeparator();
            ImGui::Text("Video clips");
            MaterialProperties& videoMat = (*materials)[currentIndex];
            const std::vector<VideoTextureService::VideoState> videoStates = videoService->states();
            int selectedVideo = -1;
            for (size_t i = 0; i < videoStates.size(); ++i) {
                const auto& vs = videoStates[i];
                if (videoMat.kind == MaterialKind::Video &&
                    videoMat.textureIndex == static_cast<int>(vs.slot)) {
                    selectedVideo = static_cast<int>(i);
                }
                ImGui::PushID(static_cast<int>(i));
                const ImTextureID thumb = videoService->previewTexture(vs.slot);
                bool assigned = false;
                if (thumb) {
                    assigned = ImGui::ImageButton("clip", thumb, ImVec2(44.0f, 44.0f));
                } else {
                    assigned = ImGui::Button("empty", ImVec2(44.0f, 44.0f));
                }
                if (assigned) {
                    videoMat.kind = MaterialKind::Video;
                    videoMat.textureIndex = static_cast<int>(vs.slot);
                    videoService->setPlaying(i, true);
                    videoService->setLooping(i, true);
                    materialDirty = true;
                    selectedVideo = static_cast<int>(i);
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s (slot %u)\nClick to assign it to material %zu.",
                                      vs.name.c_str(), vs.slot, currentIndex);
                }
                ImGui::PopID();
                if ((i + 1) % 6 != 0) ImGui::SameLine();
            }
            ImGui::NewLine();
            if (selectedVideo >= 0) {
                const auto& vs = videoStates[static_cast<size_t>(selectedVideo)];
                ImGui::Text("%s  %.2f / %.2f s", vs.name.c_str(), vs.clock, vs.duration);
                if (!vs.playing) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.15f, 1.0f), "paused");
                }
                bool playing = vs.playing;
                if (ImGui::Checkbox("play", &playing))
                    videoService->setPlaying(static_cast<size_t>(selectedVideo), playing);
                ImGui::SameLine();
                bool looping = vs.looping;
                if (ImGui::Checkbox("loop", &looping))
                    videoService->setLooping(static_cast<size_t>(selectedVideo), looping);
            } else if (videoMat.kind == MaterialKind::Video) {
                ImGui::TextDisabled("textureIndex %d does not name a loaded clip", videoMat.textureIndex);
            }
            if (videoMaterialIndex >= 0 && static_cast<size_t>(videoMaterialIndex) < tc) {
                if (ImGui::Button("Select video cube material")) {
                    currentIndex = static_cast<size_t>(videoMaterialIndex);
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Jump to material %d (kind Video on the video-layer demo cube).",
                                      videoMaterialIndex);
                }
                if (onFocusVideoCube) {
                    ImGui::SameLine();
                    if (ImGui::Button("Focus video cube")) onFocusVideoCube();
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Move the camera in front of the video-layer demo cube.");
                    }
                }
            }
            if (ImGui::SmallButton("Play all")) {
                for (size_t i = 0; i < videoStates.size(); ++i) videoService->setPlaying(i, true);
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Pause all")) {
                for (size_t i = 0; i < videoStates.size(); ++i) videoService->setPlaying(i, false);
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Loop all")) {
                for (size_t i = 0; i < videoStates.size(); ++i) videoService->setLooping(i, true);
            }
            ImGui::TextDisabled("Video shading/blending is applied by the video layer "
                                "(demo cube material %d).", videoMaterialIndex);
        }
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
                    MaterialProperties& previewMat = (*materials)[currentIndex];
                    const bool videoMat = previewMat.kind == MaterialKind::Video && videoPreviewProvider;
                    ImTextureID tex = 0;
                    if (videoMat) {
                        if (map <= 3) {
                            const uint32_t slot = static_cast<uint32_t>(std::max(0, previewMat.textureIndex));
                            tex = videoPreviewProvider(slot, map);
                        }
                    } else {
                        tex = arrayManager->getImTexture(currentIndex, map);
                    }
                    if (tex) {
                        ImGui::Image(tex, ImVec2(kPreviewSize, kPreviewSize));
                    } else {
                        ImGui::Text("Texture preview not available");
                        if (!videoMat && ImGui::Button("Recreate descriptor")) {
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
                    if (videoMat) {
                        ImGui::TextDisabled(
                            "Video material: albedo is the live video slot; Normal/Height/"
                            "Roughness are derived from the current frame.");
                        if (map > 3) {
                            ImGui::TextDisabled("AO keeps the material default for video.");
                        }
                        if (map == 0 && videoSlotNamesProvider) {
                            const std::vector<std::string> slots = videoSlotNamesProvider();
                            if (!slots.empty()) {
                                ImGui::Spacing();
                                ImGui::TextDisabled("Video slots (click a clip to assign it):");
                                for (uint32_t s = 0; s < slots.size(); ++s) {
                                    ImTextureID clipTex = videoPreviewProvider(s, 0);
                                    ImGui::PushID(static_cast<int>(s));
                                    if (clipTex) {
                                        if (ImGui::ImageButton("clip", clipTex, ImVec2(48.0f, 48.0f))) {
                                            previewMat.textureIndex = static_cast<int>(s);
                                            materialDirty = true;
                                        }
                                    } else {
                                        ImGui::Button("empty");
                                    }
                                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", slots[s].c_str());
                                    ImGui::PopID();
                                    if ((s + 1) % 6 != 0) ImGui::SameLine();
                                }
                            }
                        }
                    } else {
                        FieldLabel(tab.chooseLabel, "Click a thumbnail to select this texture layer.");
                        size_t idx = currentIndex;
                        if (ScrollableTexturePicker(tab.pickerId, arrayManager->layerAmount, idx,
                                [this, map](size_t l) { return arrayManager->getImTexture(l, map); },
                                48.0f, 2, true, true, kColumnWidth)) {
                            currentIndex = idx;
                        }
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
        EstimateSectionHeight(13),
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
