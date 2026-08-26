#include "LocalScene.hpp"
#include "../space/OctreeFile.hpp"
#include "../space/OctreeNode.hpp"
#include "../space/OctreeAllocator.hpp"
#include "../sdf/SDF.hpp"
#include "../math/Math.hpp"
#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstring>
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


    tree->iterateMultiThreaded(
        [this, tree,&data,&context,&callback](const Octree &treeRef, OctreeNodeData &params) {
            // Walk the ENTIRE subtree of the chunk being added (all branches), not
            // just the center column, so every ladder level is fully covered.
            // Visit ancestors on the path to the chunk (so we can reach it) and
            // every descendant inside the chunk.
            const bool ancestor = params.cube.getLengthX() > data.cube.getLengthX();
            bool inSubtree = ancestor ? params.cube.contains(data.cube.getCenter())
                                      : data.cube.contains(params.cube);
            if(!inSubtree) return false;

            if(params.node->getType() != SpaceType::Surface) {
                return true;  // descend through non-surface cells toward the chunk
            }

            const uint8_t chunkLod = params.node->getChunkLod();
            if(chunkLod > 0 && params.level >= data.level) {
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
                    // Tessellation targetLod is compared against getLod() inside
                    // iterateTriangles (Octree.cpp:271), so it must be the cell's
                    // OWN size-based ladder level, not the chunk-relative chunkLod.
                    // Passing chunkLod (1..R+1) descends to getLod==chunkLod cells
                    // — for the chunk node that is the frontier (getLod==1), i.e.
                    // the whole chunk at max detail. Using getLod() makes this rung
                    // emit only at the chunk's own resolution, so each ladder rung
                    // carries exactly one level of detail. chunkLod is still used
                    // for the selection gate and the published band (chunkLod-1).
                    tree->iterateTriangles(params.node, params.cube, params.level, nodeTesselator, &context, chunkLod);
                    {
                        std::lock_guard<std::mutex> lock(emittedMutex_);
                        emittedVersion_[nodeId] = params.node->version;
                    }
                    if(!nodeTesselator.geometry.indices.empty()) {
                        // Publish the 0-based band level: decode the +1-shifted
                        // storage (chunkLod - 1). The renderer gates vegetation
                        // and ChunkManager tracking on lod == 0 (the frontier
                        // chunk rung), and indirect.comp clamps selectedLevel to
                        // maxLevel = maxChunkLod() (0-based) — a 1-based value
                        // makes lod 0 never exist (no vegetation, closest band
                        // empty) and the coarsest rung unselectable.
                        callback(nodeTesselator.geometry, chunkLod - 1, params.node->version,
                                 reinterpret_cast<uintptr_t>(params.node), params.cube);
                    }
                }
            }
            // Keep descending to cover the whole chunk subtree.
            return true;
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
