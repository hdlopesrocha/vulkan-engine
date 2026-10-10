#pragma once

#include "Widget.hpp"
#include "../utils/Scene.hpp"
#include <functional>
#include <string>

class SceneRenderer;

// LayersWidget: dynamic Scene layer list. Each row selects the renderer
// (Solid/Water) for that layer; add/remove edits the Scene. Renderer switches
// and add/remove take full effect after Generate Map / file load (existing
// GPU slots are cleared for the touched layer so nothing renders twice).
class LayersWidget : public Widget {
public:
    // onLayersChanged is called after add/remove so the app can rebuild its
    // per-layer collectors/handlers (MyApp::syncLayers). Not owned.
    LayersWidget(Scene* scene, SceneRenderer* renderer, std::function<void()> onLayersChanged = {});

    void render() override;
    void setScene(Scene* scene) { scene_ = scene; }
    // Structural edits (add/remove) only make sense for the local scene;
    // MyApp disables them while rendering the remote stream.
    void setAllowStructureEdit(bool allow) { allowStructureEdit_ = allow; }

private:
    Scene* scene_ = nullptr; // not owned (active scene; MyApp re-points on remote toggle)
    SceneRenderer* renderer_ = nullptr; // not owned
    std::function<void()> onLayersChanged_;
    std::string newLayerName_;
    bool allowStructureEdit_ = true;
};
