#pragma once

#include "Widget.hpp"
#include "components/FilePicker.hpp"
#include "../math/BrushMode.hpp"
#include "../utils/Scene.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <functional>

#include "types/BrushEntry.hpp"
#include "types/ImageType.hpp"
#include "../utils/Brush3dManager.hpp"

class TextureArrayManager;
class SceneRenderer;
class VulkanApp;
class LocalScene;
class EventManager;


class Brush3dWidget : public Widget {
public:

    // Construct with a reference to a shared Brush3dManager (owned by caller)
    Brush3dWidget(TextureArrayManager* texMgr, uint32_t loadedLayers, Brush3dManager& mgr, EventManager* eventMgr);

    void render() override;

    // Dynamic scene layers for the target-layer combo (set by MyApp after the
    // world exists; may be null, in which case the static fallback names are
    // used). Not owned.
    void setScene(Scene* scene) { scene_ = scene; }
    std::string layerDisplayName(int targetLayer) const;

    // Brush mapper: the per-brush image source table (Image / TextureMixer /
    // Video). MyApp passes the table and the source ranges; the widget edits an
    // entry and calls the change callback so the app re-uploads it.
    void setBrushMap(BrushMap* map, uint32_t imageCount, uint32_t mixerCount) {
        brushMap_ = map;
        imageCount_ = imageCount;
        mixerCount_ = mixerCount;
    }
    void setOnBrushMapChanged(std::function<void(size_t)> cb) { onBrushMapChanged_ = std::move(cb); }

    // Note: widget now publishes a `RebuildBrushEvent` via the provided EventManager

private:
    void renderEntry(int index);
    void renderMaterialPicker(BrushEntry& entry);
    void renderBrushMapper();

    // Reference to caller-owned manager object
    Brush3dManager& manager;
    TextureArrayManager* textureArrayManager;
    uint32_t loadedTextureLayers;
    bool dirty = false;
    EventManager* eventManager = nullptr;
    Scene* scene_ = nullptr;
    BrushMap* brushMap_ = nullptr;
    uint32_t imageCount_ = 0;
    uint32_t mixerCount_ = 0;
    int mapperBrush_ = 0;
    std::function<void(size_t)> onBrushMapChanged_;
    // "Upload SVG..." file dialog for the SVG extrusion brush (sdfType 10).
    FilePicker svgPicker_{"Select SVG File", ".svg"};
    std::string svgStatus_;
    // selected index is owned by the manager

    static const char* sdfTypeNames[];
    static const char* brushModeNames[];
    static const char* layerNames[];
    static const char* effectTypeNames[];
};
