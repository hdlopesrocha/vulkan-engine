#pragma once

#include <memory>
#include <mutex>
#include <string>

#include "LayerRendererType.hpp"

class Octree;

// One scene layer. A layer owns its octree (shared: worker tasks keep a
// reference while they walk it) and selects which renderer publishes its
// chunks (Solid / Water / Video).
//
// `Id` is a stable key used only where a value must outlive the object or
// cross a boundary: the wire protocol's layer byte, the GPU per-layer state
// arrays and the settings/undo history. It is NOT the layer's position in the
// scene — layers can be removed and appended, so callers that have a position
// use Scene::layerAt(), and callers that have an id use Scene::findLayer().
class Layer {
public:
    using Id = int;

    Layer(Id id, std::string name, LayerRendererType renderer,
          std::shared_ptr<Octree> octree);

    Id id() const;

    std::string name() const;
    void setName(const std::string& name);

    LayerRendererType renderer() const;
    void setRenderer(LayerRendererType renderer);

    bool enabled() const;
    void setEnabled(bool enabled);

    Octree* octree();
    const Octree* octree() const;
    // Worker-safe share: request* copies this under the scene lock and walks
    // without holding the layer mutex, so an Octree cannot be destroyed while
    // an in-flight task still references it.
    std::shared_ptr<Octree> octreeRef() const;
    bool hasOctree() const;

private:
    Id id_;
    std::string name_;
    LayerRendererType renderer_;
    bool enabled_ = true;
    std::shared_ptr<Octree> octree_;
    mutable std::mutex mutex_;
};
