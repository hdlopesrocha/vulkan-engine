#pragma once
#include "Widget.hpp"
#include "../vulkan/TextureMixer.hpp"

#include <vector>
#include <memory>

class TextureMixerWidget : public Widget {
public:
    TextureMixerWidget(std::shared_ptr<TextureMixer> textures, std::vector<MixerParameters>& mixerParams, const char* title = "Texture Mixer");
    void render() override;

private:
    void renderTextureTab(int map);

    std::shared_ptr<TextureMixer> textures;
    std::vector<MixerParameters>& mixerParams;
    std::vector<std::string> diagLog;
    size_t currentMixerIndex = 0;
    int activeMap = 0; // 0=albedo,1=normal,2=bump
};