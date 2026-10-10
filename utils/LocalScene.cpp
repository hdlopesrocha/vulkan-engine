#include "LocalScene.hpp"
#include "../space/OctreeFile.hpp"
#include "../space/OctreeNode.hpp"
#include "../space/OctreeAllocator.hpp"
#include "../space/IteratorHandler.hpp"
#include "../sdf/SDF.hpp"
#include "../math/Math.hpp"
#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstring>
#include <shared_mutex>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <glm/glm.hpp>
#include <filesystem>

namespace {
struct SceneBundleHeader {
    char magic[8];
    uint32_t version;
    uint32_t hasSettings;
};

constexpr const char kSceneBundleMagic[8] = {'S', 'C', 'N', 'B', 'N', 'D', 'L', '1'};
constexpr uint32_t kSceneBundleVersionV1 = 1;
constexpr uint32_t kSceneBundleVersionV2 = 2;
constexpr uint32_t kSceneBundleVersion = kSceneBundleVersionV2;
}

LocalScene::LocalScene()
    : threadPool(std::thread::hardware_concurrency()) {
    layers_.reserve(2);
    SceneLayer opaque;
    opaque.name = "Opaque";
    opaque.renderer = LayerRendererType::Solid;
    opaque.enabled = true;
    opaque.octree = std::make_shared<Octree>(BoundingCube(glm::vec3(0.0f), 30.0f), glm::pow(2, 9));
    layers_.push_back(std::move(opaque));
    SceneLayer transp;
    transp.name = "Transparent";
    transp.renderer = LayerRendererType::Water;
    transp.enabled = true;
    transp.octree = std::make_shared<Octree>(BoundingCube(glm::vec3(0.0f), 30.0f), glm::pow(2, 9));
    layers_.push_back(std::move(transp));
}

LocalScene::~LocalScene() = default;

void LocalScene::stopPools() {
    threadPool.stop();
    std::lock_guard<std::mutex> lock(layersMutex_);
    for (auto& l : layers_) {
        if (l.octree) l.octree->threadPool.stop();
    }
}

size_t LocalScene::layerCount() const {
    std::lock_guard<std::mutex> lock(layersMutex_);
    return layers_.size();
}

std::string LocalScene::layerName(Layer layer) const {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return {};
    return layers_[static_cast<size_t>(layer)].name;
}

void LocalScene::setLayerName(Layer layer, const std::string& name) {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return;
    layers_[static_cast<size_t>(layer)].name = name;
}

LayerRendererType LocalScene::layerRenderer(Layer layer) const {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size())
        return (layer == LAYER_TRANSPARENT) ? LayerRendererType::Water : LayerRendererType::Solid;
    return layers_[static_cast<size_t>(layer)].renderer;
}

void LocalScene::setLayerRenderer(Layer layer, LayerRendererType renderer) {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return;
    layers_[static_cast<size_t>(layer)].renderer = renderer;
}

bool LocalScene::layerEnabled(Layer layer) const {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return false;
    return layers_[static_cast<size_t>(layer)].enabled;
}

void LocalScene::setLayerEnabled(Layer layer, bool enabled) {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return;
    layers_[static_cast<size_t>(layer)].enabled = enabled;
}

Layer LocalScene::addLayer(const std::string& name, LayerRendererType renderer) {
    std::lock_guard<std::mutex> lock(layersMutex_);
    SceneLayer l;
    l.name = name.empty() ? ("Layer " + std::to_string(layers_.size())) : name;
    l.renderer = renderer;
    l.enabled = true;
    // New layers share the reference root lattice so the GPU rung gate stays
    // aligned across layers.
    glm::vec3 rootMin(0.0f);
    float rootLen = 30.0f;
    float chunkSize = glm::pow(2, 9);
    if (!layers_.empty() && layers_[0].octree) {
        rootMin = layers_[0].octree->getMin();
        rootLen = layers_[0].octree->getLengthX();
        chunkSize = layers_[0].octree->chunkSize;
    }
    l.octree = std::make_shared<Octree>(BoundingCube(rootMin, rootLen), chunkSize);
    layers_.push_back(std::move(l));
    return static_cast<Layer>(layers_.size() - 1);
}

bool LocalScene::removeLayer(Layer layer) {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return false;
    if (layers_.size() <= 1) return false; // keep at least one layer
    layers_.erase(layers_.begin() + layer);
    return true;
}

Octree* LocalScene::getLayerOctree(Layer layer) {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return nullptr;
    return layers_[static_cast<size_t>(layer)].octree.get();
}

const Octree* LocalScene::getLayerOctree(Layer layer) const {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return nullptr;
    return layers_[static_cast<size_t>(layer)].octree.get();
}

Octree* LocalScene::layerOctreeLocked(Layer layer) const {
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return nullptr;
    return layers_[static_cast<size_t>(layer)].octree.get();
}

Octree& LocalScene::getOpaqueOctree() { return *getLayerOctree(LAYER_OPAQUE); }
const Octree& LocalScene::getOpaqueOctree() const {
    return *const_cast<LocalScene*>(this)->getLayerOctree(LAYER_OPAQUE);
}
Octree& LocalScene::getTransparentOctree() {
    Octree* o = getLayerOctree(LAYER_TRANSPARENT);
    if (!o) o = getLayerOctree(LAYER_OPAQUE);
    return *o;
}
const Octree& LocalScene::getTransparentOctree() const {
    const Octree* o = const_cast<LocalScene*>(this)->getLayerOctree(LAYER_TRANSPARENT);
    if (!o) o = const_cast<LocalScene*>(this)->getLayerOctree(LAYER_OPAQUE);
    return *o;
}

void LocalScene::resetAllLayers() {
    std::lock_guard<std::mutex> lock(layersMutex_);
    for (auto& l : layers_) {
        if (l.octree) l.octree->reset();
    }
    std::lock_guard<std::mutex> elock(emittedMutex_);
    emittedVersion_.clear();
}


void LocalScene::requestModel3D(Layer layer, OctreeNodeData &data, const GeometryLodCallback& callback, ThreadPool* poolOverride) {
    std::shared_ptr<Octree> tree;
    {
        std::lock_guard<std::mutex> lock(layersMutex_);
        Octree* raw = layerOctreeLocked(layer);
        if (raw && layer >= 0 && static_cast<size_t>(layer) < layers_.size())
            tree = layers_[static_cast<size_t>(layer)].octree;
    }
    if (!tree) return;
    ThreadContext context;


    // Walk starts AT the chunk node (not the whole tree): processNodeLayer already
    // passes the correct chunk with its chunkLod, so only this chunk's subtree is
    // ever visited. Hold the tree shared lock for the traversal, exactly as the
    // prior Octree::iterateMultiThreaded did (iterateTriangles re-locks shared).
    {
        std::shared_lock<std::shared_mutex> treeLock(tree->treeMutex);
        IteratorHandler handler;
        ThreadPool& pool = (poolOverride != nullptr) ? *poolOverride : tree->threadPool;
        handler.iterateMultiThreaded(*tree, data, pool,
        [this, tree,&data,&context,&callback](const Octree &treeRef, OctreeNodeData &params) {
            if(params.node->getType() != SpaceType::Surface) {
                return true;  // descend through non-surface cells toward the chunk
            }

            const uint8_t chunkLod = params.node->getChunkLod();
            if (chunkLod > 0) {
                const uintptr_t nodeId = reinterpret_cast<uintptr_t>(params.node);
                bool skip = false;
                {
                    std::lock_guard<std::mutex> lock(emittedMutex_);
                    auto it = emittedVersion_.find(nodeId);
                    skip = (it != emittedVersion_.end() && it->second == params.node->version);
                }
                if (!skip) {
                    long trianglesCount = 0;
                    Tesselator nodeTesselator(&trianglesCount);
                    tree->iterateTriangles(params.node, params.cube, params.level, nodeTesselator, &context, chunkLod);
                    {
                        std::lock_guard<std::mutex> lock(emittedMutex_);
                        emittedVersion_[nodeId] = params.node->version;
                    }
                    if(!nodeTesselator.geometry.indices.empty()) {
                        callback(nodeTesselator.geometry, chunkLod - 1, params.node->version,
                                 reinterpret_cast<uintptr_t>(params.node), params.cube, data.cube);
                    }
                }
            }
            // Only the chunk node (the traversal root) is processed; stop descent.
            return false;
        },
        [](const Octree &treeRef, OctreeNodeData &params, uint8_t order[8]) {
            for(int i = 0 ; i < 8 ; ++i) {
                order[i] = i;
            }   
        },
        [tree,&data,&context](const Octree &treeRef, OctreeNodeData &params) {
            return params.node ? params.node->chunkLod > 0 : false;
        }
        );
    }
}

void LocalScene::requestSDFCubes(Layer layer, OctreeNodeData &data, const SdfCubeCallback& callback, ThreadPool* poolOverride) {
    std::shared_ptr<Octree> tree;
    {
        std::lock_guard<std::mutex> lock(layersMutex_);
        if (layer >= 0 && static_cast<size_t>(layer) < layers_.size())
            tree = layers_[static_cast<size_t>(layer)].octree;
    }
    if (!tree) return;
    ThreadContext context;

    // Walk starts AT the chunk node (not the whole tree): only this chunk's
    // subtree is visited, so the whole-tree inSubtree filter is no longer needed.
    {
        std::shared_lock<std::shared_mutex> lock(tree->treeMutex);
        IteratorHandler handler;
        ThreadPool& pool = (poolOverride != nullptr) ? *poolOverride : tree->threadPool;
        handler.iterateMultiThreaded(*tree, data, pool,
        [tree, &data, &context, &callback](const Octree &treeRef, OctreeNodeData &params) {
            if (params.node->getType() != SpaceType::Surface) {
                return true;  // descend through non-surface cells toward the chunk
            }

            // SDF debug cubes live at lod==1 (the same rung the solid ladder uses).
            // Emit every lod==1 surface node; the lod==1 filter above selects exactly
            // the right cells within this chunk's subtree.
            if (params.node->getLod() == 1u) {
                std::array<float, 8> sdf;
                for (size_t i = 0; i < 8; ++i) sdf[i] = params.node->sdf[i];
                callback(params.cube, sdf, 1u, params.node->version,
                         reinterpret_cast<uintptr_t>(params.node),
                         static_cast<uint32_t>(params.node->vertex.brushIndex));
            }
            // Keep descending so finer SDF nodes (also at lod==1 in deeper chunks)
            // are visited; the lod==1 filter above selects exactly the right cells.
            return params.node->getLod() > 1u;
        },
        [](const Octree &treeRef, OctreeNodeData &params, uint8_t order[8]) {
            for (int i = 0; i < 8; ++i) order[i] = i;
        },
        [tree, &data, &context](const Octree &treeRef, OctreeNodeData &params) {
            return params.node ? params.node->chunkLod > 0 : false;
        }
        );
    }
}

void LocalScene::requestBoundingBoxes(Layer layer, OctreeNodeData &data, const BBoxCallback& callback, ThreadPool* poolOverride) {
    std::shared_ptr<Octree> tree;
    {
        std::lock_guard<std::mutex> lock(layersMutex_);
        if (layer >= 0 && static_cast<size_t>(layer) < layers_.size())
            tree = layers_[static_cast<size_t>(layer)].octree;
    }
    if (!tree) return;
    ThreadContext context;

    // Walk starts AT the chunk node (not the whole tree): only this chunk's
    // subtree is visited, so the whole-tree inSubtree filter is no longer needed.
    {
        std::shared_lock<std::shared_mutex> lock(tree->treeMutex);
        IteratorHandler handler;
        ThreadPool& pool = (poolOverride != nullptr) ? *poolOverride : tree->threadPool;
        handler.iterateMultiThreaded(*tree, data, pool,
        [tree, &data, &context, &callback](const Octree &treeRef, OctreeNodeData &params) {
            if (params.node->getType() != SpaceType::Surface) {
                return true;  // descend through non-surface cells toward the chunk
            }

            // Emit every surface node whose ladder level matches the CHUNK's LoD
            // (node.lod == chunk.chunkLod), so the overlay shows all node boxes at
            // the chunk's current resolution instead of a single chunk-sized box.
            if (params.node->getLod() == data.node->getChunkLod()) {
                callback(params.cube);
            }
            // Keep descending so finer nodes (also at lod==chunkLod deeper in) are
            // visited; the equality filter above selects exactly the right cells.
            return true;
        },
        [](const Octree &treeRef, OctreeNodeData &params, uint8_t order[8]) {
            for (int i = 0; i < 8; ++i) order[i] = i;
        },
        [tree, &data, &context](const Octree &treeRef, OctreeNodeData &params) {
            return params.node ? params.node->chunkLod > 0 : false;
        }
        );
    }
}

bool LocalScene::isNodeUpToDate(Layer layer, OctreeNodeData &data, uint version) {
    (void)layer;
    return data.node->version >= version;
}

void LocalScene::noteDeletedNode(uintptr_t nodeId) {
    std::lock_guard<std::mutex> lock(emittedMutex_);
    emittedVersion_.erase(nodeId);
}

int LocalScene::maxChunkLod(Layer layer, float minSize) const {    // The number of LoD levels a chunk can hold above its tessellation
    // frontier before reaching the chunk-size boundary, clamped to the
    // ladder the tree actually provides: the root carries the highest
    // chunkLod, and its mesh is the far-distance fallback, so levels beyond
    // it are never drawn. The root's chunkLod is stored +1 shifted (uint8,
    // 0 = unset), so decode it back to the 0-based level count here.
    std::lock_guard<std::mutex> lock(layersMutex_);
    const Octree* tree = layerOctreeLocked(layer);
    if (!tree) return 0;
    int rootChunkLod = -1;
    if (tree->root) {
        const int stored = tree->root->getChunkLod();
        rootChunkLod = stored > 0 ? stored - 1 : -1;
    }
    return std::max(0, std::min(tree->heightRootToChunk(0, minSize), rootChunkLod));
}

glm::vec3 LocalScene::lodRootMin(Layer layer) const {
    std::lock_guard<std::mutex> lock(layersMutex_);
    const Octree* tree = layerOctreeLocked(layer);
    if (!tree) return glm::vec3(0.0f);
    return tree->getMin();
}

void LocalScene::loadScene(SceneLoaderCallback& callback, std::vector<Octree::OctreeNodeDataHandler> updateHandlers, std::vector<Octree::OctreeNodeDataHandler> deleteHandlers) {
    std::cout << "LocalScene::loadScene() layers=" << layerCount() << std::endl;
    auto startTime = std::chrono::steady_clock::now();
    std::vector<Octree*> octrees;
    {
        std::lock_guard<std::mutex> lock(layersMutex_);
        octrees.reserve(layers_.size());
        for (auto& l : layers_) octrees.push_back(l.octree.get());
    }
    // Pad handler vectors to the layer count so the loader can index freely.
    if (updateHandlers.size() < octrees.size()) updateHandlers.resize(octrees.size());
    if (deleteHandlers.size() < octrees.size()) deleteHandlers.resize(octrees.size());
    callback.loadScene(octrees, updateHandlers, deleteHandlers);
    auto endTime = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(endTime - startTime).count();
    std::cout << "LocalScene::loadScene Ok! " << std::to_string(elapsed) << "s"  << std::endl;
}

void LocalScene::save(const std::string& filePath, const Settings* settings) {
    std::filesystem::path outPath(filePath);
    if (outPath.has_parent_path()) {
        std::filesystem::create_directories(outPath.parent_path());
    }

    std::ofstream file(filePath, std::ios::binary);
    if (!file) {
        std::cerr << "LocalScene::save() Error opening file: " << filePath << std::endl;
        return;
    }

    std::ostringstream raw;
    SceneBundleHeader header = {};
    std::memcpy(header.magic, kSceneBundleMagic, sizeof(header.magic));
    header.version = kSceneBundleVersion;
    header.hasSettings = settings ? 1u : 0u;
    raw.write(reinterpret_cast<const char*>(&header), sizeof(header));

    uint32_t layerCount = 0;
    std::vector<std::string> names;
    std::vector<uint8_t> renderers;
    std::vector<uint8_t> enabled;
    std::vector<Octree*> trees;
    {
        std::lock_guard<std::mutex> lock(layersMutex_);
        layerCount = static_cast<uint32_t>(layers_.size());
        for (auto& l : layers_) {
            names.push_back(l.name);
            renderers.push_back(static_cast<uint8_t>(l.renderer));
            enabled.push_back(l.enabled ? 1u : 0u);
            trees.push_back(l.octree.get());
        }
    }
    raw.write(reinterpret_cast<const char*>(&layerCount), sizeof(layerCount));
    for (uint32_t i = 0; i < layerCount; ++i) {
        uint32_t nlen = static_cast<uint32_t>(names[i].size());
        raw.write(reinterpret_cast<const char*>(&nlen), sizeof(nlen));
        if (nlen > 0) raw.write(names[i].data(), nlen);
        raw.write(reinterpret_cast<const char*>(&renderers[i]), sizeof(uint8_t));
        raw.write(reinterpret_cast<const char*>(&enabled[i]), sizeof(uint8_t));
        OctreeFile saver(trees[i], names[i].empty() ? ("layer" + std::to_string(i)) : names[i]);
        saver.writeToStream(raw);
    }

    if (settings) {
        raw.write(reinterpret_cast<const char*>(settings), sizeof(Settings));
    }

    std::istringstream input(raw.str());
    gzipCompressToOfstream(input, file);
    file.close();

    std::cout << "LocalScene::save('" << filePath << "') Ok! layers=" << layerCount << std::endl;
}

void LocalScene::load(const std::string& filePath, Settings* settings) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        std::cerr << "LocalScene::load() Error opening file: " << filePath << std::endl;
        return;
    }

    std::stringstream raw = gzipDecompressFromIfstream(file);

    SceneBundleHeader header = {};
    raw.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!raw || std::memcmp(header.magic, kSceneBundleMagic, sizeof(header.magic)) != 0) {
        std::cerr << "LocalScene::load() Invalid scene bundle: " << filePath << std::endl;
        return;
    }
    if (header.version != kSceneBundleVersionV1 && header.version != kSceneBundleVersionV2) {
        std::cerr << "LocalScene::load() Unsupported bundle version " << header.version << " in " << filePath << std::endl;
        return;
    }

    if (header.version == kSceneBundleVersionV1) {
        // Legacy two-layer bundle: opaque + transparent, no layer metadata.
        std::lock_guard<std::mutex> lock(layersMutex_);
        while (layers_.size() < 2) {
            SceneLayer l;
            l.name = "Layer " + std::to_string(layers_.size());
            l.renderer = LayerRendererType::Solid;
            l.octree = std::make_shared<Octree>(BoundingCube(glm::vec3(0.0f), 30.0f), glm::pow(2, 9));
            layers_.push_back(std::move(l));
        }
        OctreeFile opaqueLoader(layers_[0].octree.get(), "opaque");
        OctreeFile transparentLoader(layers_[1].octree.get(), "transparent");
        opaqueLoader.readFromStream(raw);
        transparentLoader.readFromStream(raw);
    } else {
        uint32_t layerCount = 0;
        raw.read(reinterpret_cast<char*>(&layerCount), sizeof(layerCount));
        if (!raw || layerCount == 0 || layerCount > 64) {
            std::cerr << "LocalScene::load() Bad layer count " << layerCount << std::endl;
            return;
        }
        std::lock_guard<std::mutex> lock(layersMutex_);
        layers_.clear();
        for (uint32_t i = 0; i < layerCount; ++i) {
            uint32_t nlen = 0;
            raw.read(reinterpret_cast<char*>(&nlen), sizeof(nlen));
            std::string name;
            if (nlen > 0 && nlen < 256) {
                name.resize(nlen);
                raw.read(name.data(), nlen);
            }
            uint8_t rend = 0, en = 1;
            raw.read(reinterpret_cast<char*>(&rend), sizeof(uint8_t));
            raw.read(reinterpret_cast<char*>(&en), sizeof(uint8_t));
            SceneLayer l;
            l.name = name.empty() ? ("Layer " + std::to_string(i)) : name;
            l.renderer = (rend == 1) ? LayerRendererType::Water : LayerRendererType::Solid;
            l.enabled = (en != 0);
            l.octree = std::make_shared<Octree>();
            OctreeFile loader(l.octree.get(), l.name);
            loader.readFromStream(raw);
            layers_.push_back(std::move(l));
        }
    }

    if (header.hasSettings != 0u) {
        Settings loadedSettings = {};
        raw.read(reinterpret_cast<char*>(&loadedSettings), sizeof(Settings));
        if (raw && settings) {
            *settings = loadedSettings;
        }
    }

    file.close();
    std::cout << "LocalScene::load('" << filePath << "') Ok! layers=" << layerCount() << std::endl;
}

static void notifyChunkNodes(OctreeNode* node, const BoundingCube& cube, uint level,
                             OctreeAllocator& allocator, Octree::OctreeNodeDataHandler updateHandler, Octree::OctreeNodeDataHandler deleteHandler) {
    if (!node) return;
    if (node->isChunk()) {
        updateHandler(OctreeNodeData(level, node, cube, nullptr));
        return;
    }
    OctreeNode* children[8] = {};
    node->getChildren(allocator, children);
    for (int i = 0; i < 8; ++i) {
        if (children[i])
            notifyChunkNodes(children[i], cube.getChild(i), level + 1, allocator, updateHandler, deleteHandler);
    }
    (void)deleteHandler;
}

void LocalScene::load(const std::string& filePath, std::vector<Octree::OctreeNodeDataHandler> updateHandlers, std::vector<Octree::OctreeNodeDataHandler> deleteHandlers, Settings* settings) {
    load(filePath, settings);
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (updateHandlers.size() < layers_.size()) updateHandlers.resize(layers_.size());
    if (deleteHandlers.size() < layers_.size()) deleteHandlers.resize(layers_.size());
    for (size_t i = 0; i < layers_.size(); ++i) {
        Octree* tree = layers_[i].octree.get();
        if (tree && tree->root && tree->allocator && updateHandlers[i])
            notifyChunkNodes(tree->root, *tree, 0, *tree->allocator, updateHandlers[i], deleteHandlers[i]);
    }
}
