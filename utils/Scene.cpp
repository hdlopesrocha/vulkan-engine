#include "Scene.hpp"

Scene::Scene() = default;
Scene::~Scene() = default;

bool Scene::validLayer(Layer::Id id) const {
    return findLayer(id) != nullptr;
}

void Scene::invalidateTessellationCache(Layer& layer) {
    (void)layer;
}

Scene::ChunkTopologyStatus Scene::buildChunkTopology(Layer& layer, OctreeNodeData& data,
                                                    uint8_t chunkLod, uint32_t version,
                                                    OctreeTopologyInput& out) {
    (void)layer;
    (void)data;
    (void)chunkLod;
    (void)version;
    (void)out;
    return ChunkTopologyStatus::Unavailable;
}

void Scene::commitChunkTopology(Layer& layer, uintptr_t emittingNodeId, uint32_t version) {
    (void)layer;
    (void)emittingNodeId;
    (void)version;
}

bool Scene::buildSdfRegionTable(Layer& layer, const glm::vec3& min, const glm::vec3& max,
                                std::vector<OctreeGpuNode>& out, size_t maxNodes) {
    (void)layer;
    (void)min;
    (void)max;
    (void)out;
    (void)maxNodes;
    return false;
}

void Scene::requestSDFCubes(Layer& layer, OctreeNodeData& data,
                            const SdfCubeCallback& callback, ThreadPool* poolOverride) {
    (void)layer;
    (void)data;
    (void)callback;
    (void)poolOverride;
}

void Scene::requestBoundingBoxes(Layer& layer, OctreeNodeData& data,
                                 const BBoxCallback& callback, ThreadPool* poolOverride) {
    (void)layer;
    (void)data;
    (void)callback;
    (void)poolOverride;
}

glm::vec3 Scene::lodRootMin(const Layer& layer) const {
    (void)layer;
    return glm::vec3(0.0f);
}

float Scene::lodRootSize(const Layer& layer) const {
    (void)layer;
    return 0.0f;
}
