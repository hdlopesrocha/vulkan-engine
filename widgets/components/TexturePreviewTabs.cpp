#include "TexturePreviewTabs.hpp"
#include "TexturePreview.hpp"

namespace ImGuiComponents {

void RenderTexturePreviewTabs(const char* id, std::shared_ptr<TextureMixer> textures, std::vector<MixerParameters>& mixerParams,
                              size_t &currentMixerIndex, int &previewSource, int &activeMap) {
    const float previewSize = 512.0f;
    if (ImGui::BeginTabBar(id)) {
        if (ImGui::BeginTabItem("Albedo")) {
            activeMap = 0;
            // Determine texture to preview
            ImTextureID texID = nullptr;
            if (!mixerParams.empty() && textures) {
                MixerParameters &mp = mixerParams[currentMixerIndex];
                uint32_t layer = static_cast<uint32_t>(mp.targetLayer);
                if (previewSource == 1) layer = static_cast<uint32_t>(mp.primaryTextureIdx);
                else if (previewSource == 2) layer = static_cast<uint32_t>(mp.secondaryTextureIdx);
                texID = (ImTextureID)textures->getPreviewDescriptor(activeMap, layer);
            } else if (textures) {
                texID = (ImTextureID)textures->getPreviewDescriptor(activeMap);
            }
            uint32_t w = textures ? textures->getLayerWidth() : 0u;
            uint32_t h = textures ? textures->getLayerHeight() : 0u;
            ImGuiComponents::RenderTexturePreview(texID, previewSize, w, h, "RGBA8");
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Normal")) {
            activeMap = 1;
            ImTextureID texID = nullptr;
            if (!mixerParams.empty() && textures) {
                MixerParameters &mp = mixerParams[currentMixerIndex];
                uint32_t layer = static_cast<uint32_t>(mp.targetLayer);
                if (previewSource == 1) layer = static_cast<uint32_t>(mp.primaryTextureIdx);
                else if (previewSource == 2) layer = static_cast<uint32_t>(mp.secondaryTextureIdx);
                texID = (ImTextureID)textures->getPreviewDescriptor(activeMap, layer);
            } else if (textures) {
                texID = (ImTextureID)textures->getPreviewDescriptor(activeMap);
            }
            uint32_t w = textures ? textures->getLayerWidth() : 0u;
            uint32_t h = textures ? textures->getLayerHeight() : 0u;
            ImGuiComponents::RenderTexturePreview(texID, previewSize, w, h, "RGBA8");
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Bump")) {
            activeMap = 2;
            ImTextureID texID = nullptr;
            if (!mixerParams.empty() && textures) {
                MixerParameters &mp = mixerParams[currentMixerIndex];
                uint32_t layer = static_cast<uint32_t>(mp.targetLayer);
                if (previewSource == 1) layer = static_cast<uint32_t>(mp.primaryTextureIdx);
                else if (previewSource == 2) layer = static_cast<uint32_t>(mp.secondaryTextureIdx);
                texID = (ImTextureID)textures->getPreviewDescriptor(activeMap, layer);
            } else if (textures) {
                texID = (ImTextureID)textures->getPreviewDescriptor(activeMap);
            }
            uint32_t w = textures ? textures->getLayerWidth() : 0u;
            uint32_t h = textures ? textures->getLayerHeight() : 0u;
            ImGuiComponents::RenderTexturePreview(texID, previewSize, w, h, "RGBA8");
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

} // namespace ImGuiComponents