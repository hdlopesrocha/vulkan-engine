#include "World.hpp"

World::World()
    : scene_(std::make_unique<LocalScene>())
{
}

World::~World() = default;

void World::stopPools() {
    if (scene_) scene_->stopPools();
    if (brushScene_) brushScene_->stopPools();
}
