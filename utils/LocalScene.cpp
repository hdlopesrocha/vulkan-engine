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
constexpr uint32_t kSceneBundleVersion = 1;
}

LocalScene::LocalScene()
    : opaqueOctree(BoundingCube(glm::vec3(0.0f), 30.0f), glm::pow(2, 9)),
      transparentOctree(BoundingCube(glm::vec3(0.0f), 30.0f), glm::pow(2, 9)),
      threadPool(std::thread::hardware_concurrency()),
      opaqueLayerInfo(),
      transparentLayerInfo() {}

LocalScene::~LocalScene() = default;

void LocalScene::stopPools() {
    threadPool.stop();
    opaqueOctree.threadPool.stop();
    transparentOctree.threadPool.stop();
}

Octree& LocalScene::getOpaqueOctree() { return opaqueOctree; }
const Octree& LocalScene::getOpaqueOctree() const { return opaqueOctree; }


void LocalScene::requestModel3D(Layer layer, OctreeNodeData &data, const GeometryLodCallback& callback, ThreadPool* poolOverride) {
    Octree* tree = layer == LAYER_OPAQUE ? &opaqueOctree : &transparentOctree;
    ThreadContext context = ThreadContext(data.cube);


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
    Octree* tree = layer == LAYER_OPAQUE ? &opaqueOctree : &transparentOctree;
    ThreadContext context = ThreadContext(data.cube);

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
    Octree* tree = layer == LAYER_OPAQUE ? &opaqueOctree : &transparentOctree;
    ThreadContext context = ThreadContext(data.cube);

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
    return data.node->version >= version;
}

void LocalScene::noteDeletedNode(uintptr_t nodeId) {
    std::lock_guard<std::mutex> lock(emittedMutex_);
    emittedVersion_.erase(nodeId);
}

int LocalScene::maxChunkLod(Layer layer, float minSize) const {
    // The number of LoD levels a chunk can hold above its tessellation
    // frontier before reaching the chunk-size boundary, clamped to the
    // ladder the tree actually provides: the root carries the highest
    // chunkLod, and its mesh is the far-distance fallback, so levels beyond
    // it are never drawn. The root's chunkLod is stored +1 shifted (uint8,
    // 0 = unset), so decode it back to the 0-based level count here.
    const Octree& tree = layer == LAYER_OPAQUE ? opaqueOctree : transparentOctree;
    int rootChunkLod = -1;
    if (tree.root) {
        const int stored = tree.root->getChunkLod();
        rootChunkLod = stored > 0 ? stored - 1 : -1;
    }
    return std::max(0, std::min(tree.heightRootToChunk(0, minSize), rootChunkLod));
}

void LocalScene::loadScene(SceneLoaderCallback& callback, Octree::OctreeNodeDataHandler opaqueUpdateHandler, Octree::OctreeNodeDataHandler opaqueDeleteHandler, Octree::OctreeNodeDataHandler transparentUpdateHandler, Octree::OctreeNodeDataHandler transparentDeleteHandler) {
    std::cout << "LocalScene::loadScene() " << std::endl;
    auto startTime = std::chrono::steady_clock::now();
    callback.loadScene(opaqueOctree, opaqueUpdateHandler, opaqueDeleteHandler, transparentOctree, transparentUpdateHandler, transparentDeleteHandler);
    auto endTime = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(endTime - startTime).count();
    std::cout << "LocalScene::loadScene Ok! " << std::to_string(elapsed) << "s"  << std::endl;
}
void LocalScene::action(SceneLoaderCallback& callback, Octree::OctreeNodeDataHandler opaqueUpdateHandler, Octree::OctreeNodeDataHandler opaqueDeleteHandler, Octree::OctreeNodeDataHandler transparentUpdateHandler, Octree::OctreeNodeDataHandler transparentDeleteHandler) {
    std::cout << "LocalScene::action() " << std::endl;
    auto startTime = std::chrono::steady_clock::now();
    callback.action(opaqueOctree, opaqueUpdateHandler, opaqueDeleteHandler, transparentOctree, transparentUpdateHandler, transparentDeleteHandler);
    auto endTime = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(endTime - startTime).count();
    std::cout << "LocalScene::action Ok! " << std::to_string(elapsed) << "s"  << std::endl;
}

void LocalScene::save(const std::string& filePath, const Settings* settings) {
    OctreeFile opaqueSaver(&opaqueOctree, "opaque");
    OctreeFile transparentSaver(&transparentOctree, "transparent");

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

    opaqueSaver.writeToStream(raw);
    transparentSaver.writeToStream(raw);

    if (settings) {
        raw.write(reinterpret_cast<const char*>(settings), sizeof(Settings));
    }

    std::istringstream input(raw.str());
    gzipCompressToOfstream(input, file);
    file.close();

    std::cout << "LocalScene::save('" << filePath << "') Ok!" << std::endl;
}

void LocalScene::load(const std::string& filePath, Settings* settings) {
    OctreeFile opaqueLoader(&opaqueOctree, "opaque");
    OctreeFile transparentLoader(&transparentOctree, "transparent");

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
    if (header.version != kSceneBundleVersion) {
        std::cerr << "LocalScene::load() Unsupported bundle version " << header.version << " in " << filePath << std::endl;
        return;
    }

    opaqueLoader.readFromStream(raw);
    transparentLoader.readFromStream(raw);

    if (header.hasSettings != 0u) {
        Settings loadedSettings = {};
        raw.read(reinterpret_cast<char*>(&loadedSettings), sizeof(Settings));
        if (raw && settings) {
            *settings = loadedSettings;
        }
    }

    file.close();
    std::cout << "LocalScene::load('" << filePath << "') Ok!" << std::endl;
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
}

void LocalScene::load(const std::string& filePath, const Octree::OctreeNodeDataHandler opaqueUpdateHandler, const Octree::OctreeNodeDataHandler opaqueDeleteHandler, const Octree::OctreeNodeDataHandler transparentUpdateHandler, const Octree::OctreeNodeDataHandler transparentDeleteHandler, Settings* settings) {
    load(filePath, settings);
    if (opaqueOctree.root)
        notifyChunkNodes(opaqueOctree.root, opaqueOctree, 0, *opaqueOctree.allocator, opaqueUpdateHandler, opaqueDeleteHandler);
    if (transparentOctree.root)
        notifyChunkNodes(transparentOctree.root, transparentOctree, 0, *transparentOctree.allocator, transparentUpdateHandler, transparentDeleteHandler);
}
