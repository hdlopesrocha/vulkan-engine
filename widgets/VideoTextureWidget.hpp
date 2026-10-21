#pragma once

#include "Widget.hpp"
#include "../services/VideoTextureService.hpp"
#include "types/MaterialProperties.hpp"

#include <functional>
#include <memory>
#include <vector>

class TextureArrayManager;

// Video textures: one panel per loaded clip (live video preview, playback
// control, per-clip instrumentation, load/unload) plus global transport
// controls that keep the clips synchronised to a shared clock. Clips live in
// their own video texture array; materials of kind Video reference a slot
// through textureIndex, so the same clip can be used by several materials.
class VideoTextureWidget : public Widget {
public:
    explicit VideoTextureWidget(std::shared_ptr<VideoTextureService> videos);

    // Materials this panel defines (shared editor, same as the Textures and
    // Texture Mixer widgets). The edited material is the one whose video slot
    // matches the selected clip when one exists, else `materialIndex`.
    void setMaterials(std::vector<MaterialProperties>* mats) { materials = mats; }
    // Called with the edited material index so the app can re-upload it.
    void setOnMaterialChanged(std::function<void(size_t)> cb) { onMaterialChanged = std::move(cb); }

    void render() override;

private:
    std::shared_ptr<VideoTextureService> videos;
    std::vector<MaterialProperties>* materials = nullptr;
    std::function<void(size_t)> onMaterialChanged;
    std::vector<VideoTextureService::AvailableVideo> available;
    bool scanned = false;
    int selected = 0;
    int materialIndex = 0;
    float seekSeconds = 0.0f;
    float rate = 1.0f;
    bool loopAll = true;
    bool playbackPrimed = false;
};
