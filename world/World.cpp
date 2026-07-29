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

Chunk* World::getOrCreateChunk(ChunkId id, uint32_t layer) {
    std::lock_guard<std::mutex> lock(chunkMutex_);
    auto it = chunkMap_.find(id);
    if (it != chunkMap_.end()) {
        return it->second.get();
    }
    auto chunk = std::make_unique<Chunk>(id);
    chunk->layer = layer;
    Chunk* ptr = chunk.get();
    chunkMap_[id] = std::move(chunk);
    return ptr;
}

Chunk* World::getChunk(ChunkId id) {
    std::lock_guard<std::mutex> lock(chunkMutex_);
    auto it = chunkMap_.find(id);
    return (it != chunkMap_.end()) ? it->second.get() : nullptr;
}

const Chunk* World::getChunk(ChunkId id) const {
    std::lock_guard<std::mutex> lock(chunkMutex_);
    auto it = chunkMap_.find(id);
    return (it != chunkMap_.end()) ? it->second.get() : nullptr;
}

void World::removeChunk(ChunkId id) {
    {
        std::lock_guard<std::mutex> lock(chunkMutex_);
        chunkMap_.erase(id);
    }
    chunkManager_.removeChunk(id);
}

void World::removeAllChunks() {
    {
        std::lock_guard<std::mutex> lock(chunkMutex_);
        chunkMap_.clear();
    }
    chunkManager_.removeAll();
}

bool World::markChunkDirty(ChunkId id, uint32_t version, uint32_t layer) {
    // Ensure the chunk exists in the world map.
    getOrCreateChunk(id, layer);

    // Forward to the state machine (thread-safe, deduplicates).
    bool newlyQueued = chunkManager_.markDirty(id, version);

    if (newlyQueued && dirtyCallback_) {
        dirtyCallback_(id, version, layer);
    }

    return newlyQueued;
}

void World::notifyProxySwapped(ChunkId id, std::shared_ptr<const RenderProxy> newProxy) {
    std::lock_guard<std::mutex> lock(chunkMutex_);
    auto it = chunkMap_.find(id);
    if (it != chunkMap_.end()) {
        it->second->currentProxy = std::move(newProxy);
    }
}
