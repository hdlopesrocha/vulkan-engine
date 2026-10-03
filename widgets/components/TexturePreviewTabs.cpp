#include "TexturePreviewTabs.hpp"
#include "TexturePreview.hpp"

namespace ImGuiComponents {

namespace {

struct PreviewTab {
    const char* label;
    int mapIndex;
};

const PreviewTab kPreviewTabs[] = {
    { "Albedo", 0 },
    { "Normal", 1 },
    { "Bump",   2 },
};

} // namespace

void RenderTexturePreviewTabs(const char* id, std::shared_ptr<TextureMixer> textures, std::vector<MixerParameters>& mixerParams,
                              size_t &currentMixerIndex, int &previewSource, int &activeMap, bool showNoise) {
    const float previewSize = 512.0f;
    ImGui::PushStyleVar(ImGuiStyleVar_TabBarBorderSize, 0.0f);
    if (ImGui::BeginTabBar(id)) {
        for (const PreviewTab &tab : kPreviewTabs) {
            if (ImGui::BeginTabItem(tab.label)) {
                activeMap = tab.mapIndex;
                // Determine texture to preview
                ImTextureID texID = 0;
                if (!mixerParams.empty() && textures) {
                    MixerParameters &mp = mixerParams[currentMixerIndex];
                    uint32_t layer = static_cast<uint32_t>(mp.targetLayer);
                    if (previewSource == 1) layer = static_cast<uint32_t>(mp.primaryTextureIdx);
                    else if (previewSource == 2) layer = static_cast<uint32_t>(mp.secondaryTextureIdx);
                    if (showNoise && previewSource == 0) {
                        texID = (ImTextureID)textures->getNoiseDescriptor(layer);
                    } else {
                        texID = (ImTextureID)textures->getPreviewDescriptor(activeMap, layer);
                    }
                }
                uint32_t w = textures ? textures->getLayerWidth() : 0u;
                uint32_t h = textures ? textures->getLayerHeight() : 0u;
                ImGuiComponents::RenderTexturePreview(texID, previewSize, w, h, "RGBA8");
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
    }
    ImGui::PopStyleVar();
}

} // namespace ImGuiComponents
