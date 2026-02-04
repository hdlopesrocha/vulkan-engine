#include "TextureMixerWidget.hpp"
#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>
#include "components/ScrollablePicker.hpp"

TextureMixerWidget::TextureMixerWidget(std::shared_ptr<TextureMixer> textures_, std::vector<MixerParameters>& mixerParams_, const char* title)
    : Widget(title), textures(std::move(textures_)), mixerParams(mixerParams_) {}

void TextureMixerWidget::render() {
    if (!ImGui::Begin(title.c_str(), &isOpen, ImGuiWindowFlags_None)) {
        ImGui::End();
        return;
    }

    if (ImGui::BeginTabBar("TextureTabBar")) {
        if (ImGui::BeginTabItem("Albedo")) { activeMap = 0; renderTextureTab(0); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Normal")) { activeMap = 1; renderTextureTab(1); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem("Bump"))   { activeMap = 2; renderTextureTab(2); ImGui::EndTabItem(); }
        ImGui::EndTabBar();
    }

    ImGui::Separator();
    ImGui::Text("Perlin Noise Generator");

    if (textures) {
        size_t pending = textures->getPendingGenerationCount();
        if (pending > 0) ImGui::TextColored(ImVec4(1.0f,0.9f,0.1f,1.0f), "Generation pending: %zu", pending);
        auto newLogs = textures->consumeLogs();
        for (auto &l : newLogs) diagLog.push_back(l);
        if (!diagLog.empty()) {
            ImGui::Separator();
            ImGui::Text("Diagnostics:");
            int start = diagLog.size() > 8 ? static_cast<int>(diagLog.size() - 8) : 0;
            for (int i = start; i < static_cast<int>(diagLog.size()); ++i) ImGui::Text("%s", diagLog[i].c_str());
        }
    }

    uint32_t maxLayers = textures->getArrayLayerCount();

    size_t count = mixerParams.size();
    if (count == 0) {
        ImGui::Text("No mixers configured.");
    } else {
        if (currentMixerIndex >= count) currentMixerIndex = count - 1;
        size_t previousIndex = currentMixerIndex;

        ImGui::PushID(0);
        ImGui::Text("Mixer %zu of %zu", currentMixerIndex + 1, count);
        ImGui::SameLine(); if (ImGui::Button("Prev") && currentMixerIndex > 0) --currentMixerIndex;
        ImGui::SameLine(); if (ImGui::Button("Next") && currentMixerIndex + 1 < count) ++currentMixerIndex;
        ImGui::SameLine(); int idxInput = static_cast<int>(currentMixerIndex); if (ImGui::InputInt("##mixer_idx", &idxInput)) {
            if (idxInput < 0) idxInput = 0; if (static_cast<size_t>(idxInput) >= count) idxInput = static_cast<int>(count - 1); currentMixerIndex = static_cast<size_t>(idxInput);
        }
        ImGui::PopID();

        MixerParameters &mp = mixerParams[currentMixerIndex];
        if (previousIndex != currentMixerIndex) {
            if (maxLayers > 0) textures->enqueueGenerate(mp);
            else fprintf(stderr, "[TextureMixerWidget] Skipping Perlin generation: no texture arrays available (target layer=%zu)\n", mp.targetLayer);
        }
        // Texture selection UI: allow picking primary and secondary layers via thumbnails
        if (maxLayers > 0) {
            ImGui::Separator();
            ImGui::Text("Blend Sources (click to select)");

            // Show currently selected primary/secondary previews
            ImGui::Columns(2, "preview_cols", true);
            ImGui::Text("Primary"); ImGui::NextColumn(); ImGui::Text("Secondary"); ImGui::NextColumn();
            ImTextureID pTex = (ImTextureID)textures->getPreviewDescriptor(activeMap, mp.primaryTextureIdx);
            ImTextureID sTex = (ImTextureID)textures->getPreviewDescriptor(activeMap, mp.secondaryTextureIdx);
            // Make the primary/secondary previews clickable so they become the main preview source
            if (pTex) {
                if (ImGui::ImageButton(pTex, ImVec2(128,128))) { mp.primaryTextureIdx = mp.primaryTextureIdx; previewSource = 1; }
            } else ImGui::Dummy(ImVec2(128,128));
            ImGui::NextColumn();
            if (sTex) {
                if (ImGui::ImageButton(sTex, ImVec2(128,128))) { mp.secondaryTextureIdx = mp.secondaryTextureIdx; previewSource = 2; }
            } else ImGui::Dummy(ImVec2(128,128));
            ImGui::NextColumn();
            ImGui::Columns(1);

            ImGui::Columns(2, "pickers_cols", true);
            ImGui::Text("Choose Primary (click thumbnail)"); ImGui::NextColumn();
            ImGui::Text("Choose Secondary (click thumbnail)"); ImGui::NextColumn();

            {
                size_t primaryIdx = mp.primaryTextureIdx;
                if (ImGuiComponents::ScrollableTexturePicker("MixerPrimary", maxLayers, primaryIdx, [this](size_t l){ return (ImTextureID)textures->getPreviewDescriptor(this->activeMap, static_cast<uint32_t>(l)); }, 48.0f, 2, true, true)) {
                    mp.primaryTextureIdx = static_cast<uint32_t>(primaryIdx);
                    previewSource = 1;
                    // Trigger perlin generation for this map when primary selection changes
                    if (textures) textures->enqueueGenerate(mp, activeMap);
                }
            }
            ImGui::NextColumn();
            {
                size_t secondaryIdx = mp.secondaryTextureIdx;
                if (ImGuiComponents::ScrollableTexturePicker("MixerSecondary", maxLayers, secondaryIdx, [this](size_t l){ return (ImTextureID)textures->getPreviewDescriptor(this->activeMap, static_cast<uint32_t>(l)); }, 48.0f, 2, true, true)) {
                    mp.secondaryTextureIdx = static_cast<uint32_t>(secondaryIdx);
                    previewSource = 2;
                    // Trigger perlin generation for this map when secondary selection changes
                    if (textures) textures->enqueueGenerate(mp, activeMap);
                }
            }
            ImGui::NextColumn();
            ImGui::Columns(1);
        } else {
            ImGui::Text("No texture arrays allocated — using editable textures");
        }

        ImGui::Separator();
        ImGui::Text("Noise Parameters");
        bool paramsChanged = false;
        int scale = static_cast<int>(mp.perlinScale);
        if (ImGui::SliderInt("Scale", &scale, 1, 32)) { mp.perlinScale = static_cast<float>(scale); paramsChanged = true; }
        if (ImGui::SliderFloat("Octaves", &mp.perlinOctaves, 1.0f, 8.0f)) paramsChanged = true;
        if (ImGui::SliderFloat("Persistence", &mp.perlinPersistence, 0.0f, 1.0f)) paramsChanged = true;
        if (ImGui::SliderFloat("Lacunarity", &mp.perlinLacunarity, 1.0f, 4.0f)) paramsChanged = true;
        if (ImGui::SliderFloat("Time", &mp.perlinTime, 0.0f, 100.0f)) paramsChanged = true;

        ImGui::Separator();
        ImGui::Text("Adjustments");
        if (ImGui::SliderFloat("Brightness", &mp.perlinBrightness, -1.0f, 1.0f)) paramsChanged = true;
        if (ImGui::SliderFloat("Contrast", &mp.perlinContrast, 0.0f, 5.0f)) paramsChanged = true;

        if (paramsChanged) {
            if (maxLayers > 0) textures->enqueueGenerate(mp, activeMap);
            else fprintf(stderr, "[TextureMixerWidget] Params changed but no texture arrays allocated — generation skipped.\n");
        }
    }

    ImGui::Separator();
    ImGui::End();
}

void TextureMixerWidget::renderTextureTab(int map) {
    float previewSize = 256.0f;
    ImVec2 imageSize(previewSize, previewSize);
    ImTextureID texID = nullptr;
    if (!mixerParams.empty()) {
        MixerParameters &mp = mixerParams[currentMixerIndex];
        uint32_t layer = static_cast<uint32_t>(mp.targetLayer);
        // Show main preview based on previewSource: target, primary or secondary
        if (previewSource == 1) layer = static_cast<uint32_t>(mp.primaryTextureIdx);
        else if (previewSource == 2) layer = static_cast<uint32_t>(mp.secondaryTextureIdx);
        texID = (ImTextureID)textures->getPreviewDescriptor(map, layer);
    } else {
        texID = (ImTextureID)textures->getPreviewDescriptor(map);
    }

    if (texID) {
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImVec2 end = ImVec2(pos.x + imageSize.x, pos.y + imageSize.y);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const float tile = 16.0f;
        ImU32 colA = IM_COL32(200,200,200,255);
        ImU32 colB = IM_COL32(160,160,160,255);
        for (float y = pos.y; y < end.y; y += tile) {
            for (float x = pos.x; x < end.x; x += tile) {
                bool odd = (static_cast<int>((x - pos.x) / tile) + static_cast<int>((y - pos.y) / tile)) & 1;
                ImVec2 q0(x, y);
                ImVec2 q1(std::min(x + tile, end.x), std::min(y + tile, end.y));
                dl->AddRectFilled(q0, q1, odd ? colA : colB);
            }
        }
        ImGui::Image(texID, imageSize);
    } else {
        ImGui::Text("Texture preview not available");
    }

    ImGui::Text("Size: %dx%d", textures->getLayerWidth(), textures->getLayerHeight());
    ImGui::Text("Format: RGBA8");
}
