#include "Layer.hpp"

#include <utility>

Layer::Layer(Id id, std::string name, LayerRendererType renderer,
             std::shared_ptr<Octree> octree)
    : id_(id), name_(std::move(name)), renderer_(renderer),
      octree_(std::move(octree)) {}

Layer::Id Layer::id() const {
    return id_;
}

std::string Layer::name() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return name_;
}

void Layer::setName(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    name_ = name;
}

LayerRendererType Layer::renderer() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return renderer_;
}

void Layer::setRenderer(LayerRendererType renderer) {
    std::lock_guard<std::mutex> lock(mutex_);
    renderer_ = renderer;
}

bool Layer::enabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_;
}

void Layer::setEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = enabled;
}

Octree* Layer::octree() {
    return octree_.get();
}

const Octree* Layer::octree() const {
    return octree_.get();
}

std::shared_ptr<Octree> Layer::octreeRef() const {
    return octree_;
}

bool Layer::hasOctree() const {
    return octree_ != nullptr;
}
