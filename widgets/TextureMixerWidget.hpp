#pragma once
#include "Widget.hpp"
#include "../services/TextureMixer.hpp"

#include <vector>
#include <memory>

class TextureMixerWidget : public Widget {
public:
    TextureMixerWidget(std::shared_ptr<TextureMixer> textures, std::vector<MixerParameters>& mixerParams, const char* title_ = "Texture Mixer");
    void render() override;

private:
    // The tabbed preview UI is rendered by ImGuiComponents::RenderTexturePreviewTabs

    std::shared_ptr<TextureMixer> textures;
    std::vector<MixerParameters>& mixerParams;
    std::vector<std::string> diagLog;
    size_t currentMixerIndex = 0;
    int activeMap = 0; // 0=albedo,1=normal,2=bump
    // 0 = targetLayer preview, 1 = primary preview, 2 = secondary preview
    int previewSource = 0;
    bool showNoise = false; // if enabled, preview the computed noise mask instead of color
    // C2 (perf report 23): sticky "params changed" flag so the generation is
    // committed when the edit ends. ImGui reports changes during a drag but
    // not on the release frame, so a bare !IsAnyItemActive() gate would miss
    // the final value.
    bool paramsDirty = false;
};