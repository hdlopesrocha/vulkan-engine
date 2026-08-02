#include "LocalScene.hpp"
#include "../space/OctreeFile.hpp"
#include "../space/OctreeNode.hpp"
#include "../space/OctreeAllocator.hpp"
#include "../math/Math.hpp"
#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
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


void LocalScene::requestModel3D(Layer layer, OctreeNodeData &data, const LadderCallback& callback, ThreadPool* poolOverride) {
    long tessCount = 0;
    Octree* tree = layer == LAYER_OPAQUE ? &opaqueOctree : &transparentOctree;
    ThreadContext context = ThreadContext(data.cube);
    // Use the caller-supplied pool when present (e.g. brush editing runs on a
    // dedicated pool so it never competes with solid/water streaming
    // generation); otherwise fall back to the scene's shared pool.
    ThreadPool& pool = poolOverride ? *poolOverride : threadPool;
    Processor processor(&tessCount, pool, &context, data.cube);
    // ONE Tesselator per node, from the WORLD ROOT down: the flat walk visits
    // the whole tree, but the Processor prunes everything off this chunk's
    // root path, so only the chunk and its ancestors tessellate — the chunk
    // at targetLod 0 (its frontier mesh) and each ancestor at targetLod = its
    // own chunkLod (its lod-k cell mesh). The walk hands the node info and the
    // node's own root-descended bounding cube to the handler (no reconstructed
    // cubes), and propagates every parent link (root -> chunk -> leaves) —
    // the parents are required for the root-consistent cube rebuilds inside
    // iterateTriangles.
    processor.onGeometry = [&data,&callback](int level, const OctreeNode* node, const BoundingCube &cube, Geometry&& g, int lod) {
         callback(g, lod);
    };
    tree->iterateFlat(processor, OctreeNodeData(0, tree->root, static_cast<const BoundingCube&>(*tree), &context));
}

bool LocalScene::isNodeUpToDate(Layer layer, OctreeNodeData &data, uint version) {
    return data.node->version >= version;
}

int LocalScene::maxChunkLod(Layer layer, float minSize) const {
    // The number of LoD levels a chunk can hold above its tessellation
    // frontier before reaching the chunk-size boundary, clamped to the
    // ladder the tree actually provides: the root carries the highest
    // chunkLod, and its mesh is the far-distance fallback, so levels beyond
    // it are never drawn.
    const Octree& tree = layer == LAYER_OPAQUE ? opaqueOctree : transparentOctree;
    int rootChunkLod = -1;
    if (tree.root) rootChunkLod = tree.root->getChunkLod();
    return std::max(0, std::min(tree.heightRootToChunk(0, minSize), rootChunkLod));
}

void LocalScene::loadScene(SceneLoaderCallback& callback, const OctreeChangeHandler &opaqueLayerChangeHandler, const OctreeChangeHandler &transparentLayerChangeHandler) {
    std::cout << "LocalScene::loadScene() " << std::endl;
    auto startTime = std::chrono::steady_clock::now();
    callback.loadScene(opaqueOctree, opaqueLayerChangeHandler, transparentOctree, transparentLayerChangeHandler);
    auto endTime = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(endTime - startTime).count();
    std::cout << "LocalScene::loadScene Ok! " << std::to_string(elapsed) << "s"  << std::endl;
}
void LocalScene::action(SceneLoaderCallback& callback, const OctreeChangeHandler &opaqueLayerChangeHandler, const OctreeChangeHandler &transparentLayerChangeHandler) {
    std::cout << "LocalScene::action() " << std::endl;
    auto startTime = std::chrono::steady_clock::now();
    callback.action(opaqueOctree, opaqueLayerChangeHandler, transparentOctree, transparentLayerChangeHandler);
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
                             OctreeAllocator& allocator, const OctreeChangeHandler& handler) {
    if (!node) return;
    if (node->isChunk()) {
        handler.onNodeAdded(OctreeNodeData(level, node, cube, nullptr));
        return;
    }
    OctreeNode* children[8] = {};
    node->getChildren(allocator, children);
    for (int i = 0; i < 8; ++i) {
        if (children[i])
            notifyChunkNodes(children[i], cube.getChild(i), level + 1, allocator, handler);
    }
}

void LocalScene::load(const std::string& filePath, const OctreeChangeHandler& opaqueHandler, const OctreeChangeHandler& transparentHandler, Settings* settings) {
    load(filePath, settings);
    if (opaqueOctree.root)
        notifyChunkNodes(opaqueOctree.root, opaqueOctree, 0, *opaqueOctree.allocator, opaqueHandler);
    if (transparentOctree.root)
        notifyChunkNodes(transparentOctree.root, transparentOctree, 0, *transparentOctree.allocator, transparentHandler);
}
