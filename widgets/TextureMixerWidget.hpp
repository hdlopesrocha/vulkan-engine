#pragma once
#include "Widget.hpp"
#include "../services/TextureMixer.hpp"
#include "types/MaterialProperties.hpp"

#include <functional>
#include <vector>
#include <memory>

class TextureMixerWidget : public Widget {
public:
    TextureMixerWidget(std::shared_ptr<TextureMixer> textures, std::vector<MixerParameters>& mixerParams, const char* title_ = "Texture Mixer");
    void render() override;

    // Materials this panel defines (shared editor, same as the Textures and
    // Video widgets). The edited material is the one at the selected mix's
    // target layer (a mix layer is also a material index).
    void setMaterials(std::vector<MaterialProperties>* mats) { materials = mats; }
    void setOnMaterialChanged(std::function<void(size_t)> cb) { onMaterialChanged = std::move(cb); }

private:
    // The tabbed preview UI is rendered by ImGuiComponents::RenderTexturePreviewTabs

    std::shared_ptr<TextureMixer> textures;
    std::vector<MixerParameters>& mixerParams;
    std::vector<MaterialProperties>* materials = nullptr;
    std::function<void(size_t)> onMaterialChanged;
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