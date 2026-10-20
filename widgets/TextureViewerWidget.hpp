#pragma once

#include <imgui.h>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include "../vulkan/resources/TextureArrayManager.hpp"
#include "../vulkan/resources/MaterialManager.hpp"
#include "types/MaterialProperties.hpp"
#include "Widget.hpp"

class VideoTextureService;

class TextureViewer : public Widget {
public:
    TextureViewer();
    void init(TextureArrayManager* arrayManager_, std::vector<MaterialProperties>* materials_);
    void render() override;
    void setOnMaterialChanged(std::function<void(size_t)> cb) { onMaterialChanged = cb; }
    void setVideoSlotNamesProvider(std::function<std::vector<std::string>()> fn) { videoSlotNamesProvider = std::move(fn); }
    // Live preview for Video materials: (slot, map index) -> ImGui texture.
    // map index matches the PreviewTab order (0 albedo, 1 normal, 2 height,
    // 3 roughness, 4 AO); derived maps are produced from the current frame.
    void setVideoPreviewProvider(std::function<ImTextureID(uint32_t, int)> fn) { videoPreviewProvider = std::move(fn); }
    // Full video service: lets this panel list clips, assign one to the
    // current material, and expose play/pause/loop so playback is verifiable
    // from the same place the blend mode is edited.
    void setVideoService(VideoTextureService* service) { videoService = service; }
    // Material painted on the video-layer demo cube (blend modes composite
    // there); lets the panel jump straight to it.
    void setVideoMaterialIndex(int index) { videoMaterialIndex = index; }

private:
    TextureArrayManager* arrayManager = nullptr;
    std::vector<MaterialProperties>* materials = nullptr;
    size_t currentIndex = 0;
    std::function<void(size_t)> onMaterialChanged;
    std::function<std::vector<std::string>()> videoSlotNamesProvider;
    std::function<ImTextureID(uint32_t, int)> videoPreviewProvider;
    VideoTextureService* videoService = nullptr;
    int videoMaterialIndex = -1;
};
