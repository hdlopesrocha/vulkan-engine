#pragma once

#include <imgui.h>
#include <memory>
#include <vector>
#include "../../vulkan/TextureMixer.hpp"

namespace ImGuiComponents {

    // Render a 3-tab preview bar (Albedo/Normal/Bump) with the same preview behavior
    // as the legacy `renderTextureTab` and update `activeMap`/`previewSource`/`currentMixerIndex` in-place.
    void RenderTexturePreviewTabs(const char* id, std::shared_ptr<TextureMixer> textures, std::vector<MixerParameters>& mixerParams,
                                  size_t &currentMixerIndex, int &previewSource, int &activeMap);
}
