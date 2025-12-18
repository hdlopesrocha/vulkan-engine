#pragma once

#include "Widget.hpp"
#include <memory>
#include <vector>
#include "../vulkan/TextureMixer.hpp"

class AnimatedTextureWidget : public Widget {
public:
    AnimatedTextureWidget(std::shared_ptr<TextureMixer> textures, std::vector<MixerParameters>& mixerParams, const char* title = "Editable Textures");
    void render() override;

private:
    std::shared_ptr<TextureMixer> textures;
    std::vector<MixerParameters>& mixerParams;

    // Currently selected mixer index to edit (navigate between entries)
    size_t currentMixerIndex = 0;
    // Remember last selected index so we can generate preview when selection changes
    size_t lastSelectedMixer = SIZE_MAX;

    // Per-item UI state is stored in the provided `mixerParams` entries.
    // Active tab (0=albedo,1=normal,2=bump) — used to regenerate only the selected map
    int activeMap = 0;

    void renderTextureTab(EditableTexture& texture, int map);
};