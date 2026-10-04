#pragma once

#include "Widget.hpp"
#include "../math/BrushMode.hpp"
#include "../utils/Scene.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <functional>

#include "types/BrushEntry.hpp"
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

    // Note: widget now publishes a `RebuildBrushEvent` via the provided EventManager

private:
    void renderEntry(int index);
    void renderMaterialPicker(BrushEntry& entry);

    // Reference to caller-owned manager object
    Brush3dManager& manager;
    TextureArrayManager* textureArrayManager;
    uint32_t loadedTextureLayers;
    bool dirty = false;
    EventManager* eventManager = nullptr;
    // selected index is owned by the manager

    static const char* sdfTypeNames[];
    static const char* brushModeNames[];
    static const char* layerNames[];
    static const char* effectTypeNames[];
};
