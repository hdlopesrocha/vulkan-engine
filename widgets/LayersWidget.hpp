#pragma once

#include "Widget.hpp"
#include "../utils/Scene.hpp"
#include <functional>
#include <string>

class SceneRenderer;

// LayersWidget: dynamic Scene layer list. Each row selects the renderer
// (Solid/Water) for that layer; add/remove edits the Scene. Disabling a
// layer frees its GPU slots immediately; re-enabling or switching renderers
// automatically repopulates the layer (re-emit + re-tessellate) via
// onLayerRepopulate, so no manual Generate Map is needed.
class LayersWidget : public Widget {
public:
    // onLayersChanged is called after add/remove so the app can rebuild its
    // per-layer collectors/handlers (MyApp::syncLayers). onLayerRepopulate is
    // called after a layer is re-enabled or its renderer is switched so the
    // app can re-queue that layer's chunks (MyApp::repopulateLayer). Not owned.
    LayersWidget(Scene* scene, SceneRenderer* renderer,
                 std::function<void()> onLayersChanged = {},
                 std::function<void(Layer::Id)> onLayerRepopulate = {});

    void render() override;
    void setScene(Scene* scene) { scene_ = scene; }
    // Structural edits (add/remove) only make sense for the local scene;
    // MyApp disables them while rendering the remote stream.
    void setAllowStructureEdit(bool allow) { allowStructureEdit_ = allow; }

private:
    Scene* scene_ = nullptr; // not owned (active scene; MyApp re-points on remote toggle)
    SceneRenderer* renderer_ = nullptr; // not owned
    std::function<void()> onLayersChanged_;
    std::function<void(Layer::Id)> onLayerRepopulate_;
    std::string newLayerName_;
    bool allowStructureEdit_ = true;
};
