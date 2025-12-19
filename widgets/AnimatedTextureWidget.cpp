#include "AnimatedTextureWidget.hpp"
#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>
#include <random>

AnimatedTextureWidget::AnimatedTextureWidget(std::shared_ptr<TextureMixer> textures_, std::vector<MixerParameters>& mixerParams_, const char* title)
    : Widget(title), textures(std::move(textures_)), mixerParams(mixerParams_) {}

void AnimatedTextureWidget::render() {
    // Pass the widget's isOpen flag to ImGui so the window close button updates visibility
    if (!ImGui::Begin(title.c_str(), &isOpen, ImGuiWindowFlags_None)) {
        ImGui::End();
        return;
    }

    if (ImGui::BeginTabBar("TextureTabBar")) {
        if (ImGui::BeginTabItem("Albedo")) {
            activeMap = 0;
            renderTextureTab(0);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Normal")) {
            activeMap = 1;
            renderTextureTab(1);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Bump")) {
            activeMap = 2;
            renderTextureTab(2);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }


    ImGui::Separator();

    ImGui::Text("Perlin Noise Generator");

    uint32_t maxLayers = textures->getArrayLayerCount();

    // Navigation: show only one MixerParameters entry at a time
    size_t count = mixerParams.size();
    if (count == 0) {
        ImGui::Text("No mixers configured.");
    } else {
        // Clamp current index
        if (currentMixerIndex >= count) currentMixerIndex = count - 1;
        // remember previous selection to detect changes
        size_t previousIndex = currentMixerIndex;

        ImGui::PushID(0);
        ImGui::Text("Mixer %zu of %zu", currentMixerIndex + 1, count);
        ImGui::SameLine();
        if (ImGui::Button("Prev") && currentMixerIndex > 0) {
            --currentMixerIndex;
        }
        ImGui::SameLine();
        if (ImGui::Button("Next") && currentMixerIndex + 1 < count) {
            ++currentMixerIndex;
        }
        ImGui::SameLine();
        // Optional quick jump
        int idxInput = static_cast<int>(currentMixerIndex);
        if (ImGui::InputInt("##mixer_idx", &idxInput)) {
            if (idxInput < 0) idxInput = 0;
            if (static_cast<size_t>(idxInput) >= count) idxInput = static_cast<int>(count - 1);
            currentMixerIndex = static_cast<size_t>(idxInput);
        }
        ImGui::PopID();

        MixerParameters &mp = mixerParams[currentMixerIndex];

        // If the selection changed, generate the Perlin preview for the newly selected mixer once
        if (previousIndex != currentMixerIndex) {
            textures->generatePerlinNoise(mp);
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

        if (maxLayers > 0) {
            int primarySel = static_cast<int>(mp.primaryTextureIdx);
            int secondarySel = static_cast<int>(mp.secondaryTextureIdx);
            if (ImGui::SliderInt("Primary Layer", &primarySel, 0, static_cast<int>(maxLayers) - 1)) { mp.primaryTextureIdx = static_cast<uint>(primarySel); paramsChanged = true; }
            if (ImGui::SliderInt("Secondary Layer", &secondarySel, 0, static_cast<int>(maxLayers) - 1)) { mp.secondaryTextureIdx = static_cast<uint>(secondarySel); paramsChanged = true; }
        } else {
            ImGui::Text("No texture arrays allocated — using editable textures");
        }

        if (paramsChanged) {
            // Regenerate only the currently active map to avoid updating all three editable textures
            textures->generatePerlinNoise(mp);
        }
    }

    ImGui::Separator();

    ImGui::End();
}

void AnimatedTextureWidget::renderTextureTab(int map) {
    float previewSize = 256.0f; // scaled down to 25%
    ImVec2 imageSize(previewSize, previewSize);
    // Use ImTextureID to represent ImGui Vulkan texture handles (returned by ImGui_ImplVulkan_AddTexture)
    ImTextureID texID = nullptr;
    // If we have MixerParameters configured, request the preview for the selected mixer's target layer
    if (!mixerParams.empty()) {
        uint32_t layer = static_cast<uint32_t>(mixerParams[currentMixerIndex].targetLayer);
        texID = (ImTextureID)textures->getPreviewDescriptor(map, layer);
    } else {
        texID = (ImTextureID)textures->getPreviewDescriptor(map);
    }
    // Debug: log ImTextureID pointer to help diagnose preview issues
    if (texID) {
        printf("[AnimatedTextureWidget] map=%d texID=%p\n", map, (void*)texID);

        // Draw a checkerboard background so alpha in the texture appears as transparency
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

        // Now render the texture on top
        ImGui::Image(texID, imageSize);
    } else {
        printf("[AnimatedTextureWidget] map=%d texID=null\n", map);
        ImGui::Text("Texture preview not available");
    }

    ImGui::Text("Size: %dx%d", textures->getLayerWidth(), textures->getLayerHeight());
    ImGui::Text("Format: RGBA8");
}
