#include "TextureMixerWidget.hpp"

#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>
#include <iostream>
#include "components/ColumnLayout.hpp"
#include "components/ImGuiHelpers.hpp"
#include "components/ScrollablePicker.hpp"
#include "components/TexturePreview.hpp"
#include "components/TexturePreviewTabs.hpp"

namespace {
constexpr float kPreviewSize = 512.0f;
}

TextureMixerWidget::TextureMixerWidget(std::shared_ptr<TextureMixer> textures_, std::vector<MixerParameters>& mixerParams_, const char* title_)
    : Widget(title_, u8"\uf1de"), textures(std::move(textures_)), mixerParams(mixerParams_) {}

void TextureMixerWidget::render() {
    ImGui::SetNextWindowPos(ImVec2(0, 24), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(1280, 680), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(320, 300), ImVec2(FLT_MAX, FLT_MAX));
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen,
        ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    if (!wg.visible()) return;

    if (!textures) {
        ImGui::Text("Texture mixer unavailable.");
        return;
    }

    const size_t count = mixerParams.size();
    if (count == 0) {
        ImGui::Text("No mixers configured.");
        if (ImGui::Checkbox("Show noise (alpha)", &showNoise)) {
            previewSource = 0;
            textures->setDebugOutput(showNoise);
        }
        ImGuiComponents::TooltipOnHover("Preview the computed Perlin noise mask instead of the mixed color.");
        return;
    }
    if (currentMixerIndex >= count) currentMixerIndex = count - 1;

    const uint32_t maxLayers = textures->getArrayLayerCount();
    bool mixerChanged = false;
    bool paramsChanged = false;
    bool sourceChanged = false;

    std::vector<ImGuiComponents::ColumnSection> sections;
    sections.reserve(5);

    sections.push_back({[&]() {
        ImGui::Text("Mixer");
        ImGuiComponents::ColSeparator();
        ImGui::Text("Mixer %zu of %zu", currentMixerIndex + 1, count);
        if (ImGui::Button("Prev") && currentMixerIndex > 0) {
            --currentMixerIndex;
            mixerChanged = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Next") && currentMixerIndex + 1 < count) {
            ++currentMixerIndex;
            mixerChanged = true;
        }
        ImGuiComponents::FieldLabel("Mixer Index", "Index of the mixer to edit (0-based).");
        ImGui::SetNextItemWidth(ImGuiComponents::kColumnWidth);
        ImGui::PushID("mixer_idx");
        int idxInput = static_cast<int>(currentMixerIndex);
        if (ImGui::InputInt("##v", &idxInput)) {
            if (idxInput < 0) idxInput = 0;
            if (static_cast<size_t>(idxInput) >= count) idxInput = static_cast<int>(count - 1);
            currentMixerIndex = static_cast<size_t>(idxInput);
            mixerChanged = true;
        }
        ImGui::PopID();
        if (mixerChanged) {
            textures->setDebugOutput(showNoise);
            if (maxLayers > 0) {
                textures->enqueueGenerate(mixerParams[currentMixerIndex]);
            } else {
                std::cerr << "[TextureMixerWidget] Skipping Perlin generation: no texture arrays available (target layer="
                          << mixerParams[currentMixerIndex].targetLayer << ")" << std::endl;
            }
        }
        if (ImGui::Checkbox("Show noise (alpha)", &showNoise)) {
            previewSource = 0;
            textures->setDebugOutput(showNoise);
        }
        ImGuiComponents::TooltipOnHover("Preview the computed Perlin noise mask instead of the mixed color.");
        const size_t pending = textures->getPendingGenerationCount();
        if (pending > 0) {
            ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.1f, 1.0f), "Generation pending: %zu", pending);
        }
    }, ImGuiComponents::kColumnWidth});

    sections.push_back({[&]() {
        ImGui::Text("Preview");
        ImGuiComponents::ColSeparator(kPreviewSize);
        ImGuiComponents::RenderTexturePreviewTabs("TextureTabBar", textures, mixerParams, currentMixerIndex, previewSource, activeMap, showNoise);
    }, kPreviewSize});

    sections.push_back({[&]() {
        ImGui::Text("Blend Sources");
        ImGuiComponents::ColSeparator();
        if (maxLayers == 0) {
            ImGui::TextWrapped("No texture arrays allocated — using editable textures.");
            return;
        }
        MixerParameters &mp = mixerParams[currentMixerIndex];
        ImGui::TextWrapped("Click a source thumbnail to preview it in the tabs.");

        ImGuiComponents::FieldLabel("Primary Source", "Click the thumbnail to preview this source layer.");
        ImGui::PushID("primary");
        ImTextureID pTex = (ImTextureID)textures->getPreviewDescriptor(activeMap, mp.primaryTextureIdx);
        if (pTex) {
            if (ImGui::ImageButton("##img", pTex, ImVec2(128, 128))) previewSource = 1;
        } else {
            ImGui::Dummy(ImVec2(128, 128));
        }
        ImGui::PopID();
        ImGuiComponents::TooltipOnHover("Click to preview this source layer.");
        size_t primaryIdx = mp.primaryTextureIdx;
        if (ImGuiComponents::ScrollableTexturePicker("MixerPrimary", maxLayers, primaryIdx,
                [this, &mp](size_t l) {
                    if (l == static_cast<size_t>(mp.targetLayer)) return (ImTextureID)nullptr;
                    return (ImTextureID)textures->getPreviewDescriptor(this->activeMap, static_cast<uint32_t>(l));
                }, 48.0f, 2, true, true, ImGuiComponents::kColumnWidth)) {
            mp.primaryTextureIdx = static_cast<uint32_t>(primaryIdx);
            sourceChanged = true;
        }

        ImGui::Spacing();
        ImGuiComponents::ColSeparator();
        ImGui::Spacing();

        ImGuiComponents::FieldLabel("Secondary Source", "Click the thumbnail to preview this source layer.");
        ImGui::PushID("secondary");
        ImTextureID sTex = (ImTextureID)textures->getPreviewDescriptor(activeMap, mp.secondaryTextureIdx);
        if (sTex) {
            if (ImGui::ImageButton("##img", sTex, ImVec2(128, 128))) previewSource = 2;
        } else {
            ImGui::Dummy(ImVec2(128, 128));
        }
        ImGui::PopID();
        ImGuiComponents::TooltipOnHover("Click to preview this source layer.");
        size_t secondaryIdx = mp.secondaryTextureIdx;
        if (ImGuiComponents::ScrollableTexturePicker("MixerSecondary", maxLayers, secondaryIdx,
                [this, &mp](size_t l) {
                    if (l == static_cast<size_t>(mp.targetLayer)) return (ImTextureID)nullptr;
                    return (ImTextureID)textures->getPreviewDescriptor(this->activeMap, static_cast<uint32_t>(l));
                }, 48.0f, 2, true, true, ImGuiComponents::kColumnWidth)) {
            mp.secondaryTextureIdx = static_cast<uint32_t>(secondaryIdx);
            sourceChanged = true;
        }
    }, ImGuiComponents::kColumnWidth});

    sections.push_back({[&]() {
        ImGui::Text("Noise Parameters");
        ImGuiComponents::ColSeparator();
        MixerParameters &mp = mixerParams[currentMixerIndex];
        int scale = static_cast<int>(mp.perlinScale);
        if (ImGuiComponents::SliderIntField("Scale", &scale, 1, 32)) {
            mp.perlinScale = static_cast<float>(scale);
            paramsChanged = true;
        }
        if (ImGuiComponents::SliderFloatField("Octaves", &mp.perlinOctaves, 1.0f, 8.0f)) paramsChanged = true;
        if (ImGuiComponents::SliderFloatField("Persistence", &mp.perlinPersistence, 0.0f, 1.0f)) paramsChanged = true;
        if (ImGuiComponents::SliderFloatField("Lacunarity", &mp.perlinLacunarity, 1.0f, 4.0f)) paramsChanged = true;
        if (ImGuiComponents::SliderFloatField("Time", &mp.perlinTime, 0.0f, 100.0f)) paramsChanged = true;
    }, ImGuiComponents::kColumnWidth});

    sections.push_back({[&]() {
        ImGui::Text("Adjustments");
        ImGuiComponents::ColSeparator();
        MixerParameters &mp = mixerParams[currentMixerIndex];
        if (ImGuiComponents::SliderFloatField("Brightness", &mp.perlinBrightness, -1.0f, 1.0f)) paramsChanged = true;
        if (ImGuiComponents::SliderFloatField("Contrast", &mp.perlinContrast, 0.0f, 5.0f)) paramsChanged = true;
    }, ImGuiComponents::kColumnWidth});

    static std::vector<float> cachedH;
    const std::vector<float> estimates = {
        ImGuiComponents::EstimateSectionHeight(4),
        620.0f,
        ImGuiComponents::EstimateSectionHeight(10),
        ImGuiComponents::EstimateSectionHeight(5),
        ImGuiComponents::EstimateSectionHeight(2),
    };
    ImGuiComponents::LayoutSections(sections, cachedH, estimates);

    if (paramsChanged || sourceChanged) {
        if (maxLayers > 0) {
            previewSource = 0;
            textures->setDebugOutput(showNoise);
            textures->enqueueGenerate(mixerParams[currentMixerIndex], activeMap);
        } else {
            std::cerr << "[TextureMixerWidget] Params changed but no texture arrays allocated — generation skipped." << std::endl;
        }
    }
}
