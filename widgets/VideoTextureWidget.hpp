#pragma once

#include "Widget.hpp"
#include "../services/VideoTextureService.hpp"

#include <memory>
#include <vector>

class TextureArrayManager;

// Video textures: one panel per loaded clip (playback control, per-clip
// instrumentation, load/unload) plus global transport controls that keep the
// clips synchronised to a shared clock. Clips live in their own video texture
// array; materials of kind Video reference a slot through textureIndex, so the
// same clip can be used by several materials.
class VideoTextureWidget : public Widget {
public:
    VideoTextureWidget(std::shared_ptr<VideoTextureService> videos, TextureArrayManager* arrays,
                       const char* title_ = "Video Textures");

    void render() override;

private:
    std::shared_ptr<VideoTextureService> videos;
    TextureArrayManager* textureArrays = nullptr;
    std::vector<VideoTextureService::AvailableVideo> available;
    bool scanned = false;
    int selected = 0;
    float seekSeconds = 0.0f;
    float rate = 1.0f;
    bool loopAll = true;
};
