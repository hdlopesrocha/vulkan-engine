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
    // root path, so only the chunk and its ancestors tessellate — each at
    // targetLod = its own stored chunkLod (1 = frontier, k = ancestor, in the
    // +1-shifted uint8_t space; the onGeometry `lod` param is decoded back to
    // the 0-based LADDER level). The walk hands the node info and the
    // node's own root-descended bounding cube to the handler (no reconstructed
    // cubes), and propagates every parent link (root -> chunk -> leaves) —
    // the parents are required for the root-consistent cube rebuilds inside
    // iterateTriangles.
    processor.onGeometry = [&data,&callback,&tree](int level, const OctreeNode* node, const BoundingCube &cube, Geometry& g, uint8_t lod) {
        size_t farOut = 0; size_t sentinelPos = 0;
        glm::vec3 mn(FLT_MAX), mx(-FLT_MAX);
        for (const auto& v : g.vertices) {
            glm::vec3 p = v.position;
            if (glm::dot(p,p) > 9.0e7f) ++farOut;          // |p| > 9500 (outside root cube ~6650)
            if (std::fabs(p.x) > 1.0e4f || std::fabs(p.y) > 1.0e4f || std::fabs(p.z) > 1.0e4f) ++sentinelPos;
            mn = glm::min(mn, p); mx = glm::max(mx, p);
        }
        if (!g.vertices.empty()) {
            size_t offSurface = 0; size_t maxOff = 0; size_t farEmpty = 0;
            size_t h = 1469598103934665603ull;
            for (const auto& v : g.vertices) {
                uint32_t b[3];
                std::memcpy(&b, &v.position, sizeof(b));
                for (uint32_t x : b) { h ^= x; h *= 1099511628211ull; }
                // Surface-net vertex should sit ON the tree's zero set; a value
                // far from 0 (beyond ~0.25x the cell diagonal) means the vertex
                // is parked where the tree holds no real surface.
                float cellDiag = cube.getLengthX() * 1.73205080757f;
                float d = std::isnan(v.position.x) ? INFINITY : tree->getSdfAt(v.position);
                if (std::fabs(d) > cellDiag * 0.25f) {
                    ++offSurface;
                    float o = std::fabs(d) / cellDiag;
                    if (o > 10.0f) ++farEmpty;
                    maxOff = std::max(maxOff, (size_t)o);
                }
            }
            static int printed = 0;
            bool flag = offSurface > 0 && cube.getLengthX() <= 960.0f && printed < 20;
            std::cout << (flag ? "[LocalScene!] " : "[LocalScene] ") << "L=" << level << " lod=" << lod << " cube=" << cube.getLengthX()
                      << " tris=" << g.indices.size()/3 << " v=" << g.vertices.size()
                      << " OFF=" << offSurface << "/" << g.vertices.size() << " maxOff=" << maxOff << " farEmpty=" << farEmpty
                      << " hash=" << h
                      << " pos=[" << mn.x << "," << mn.y << "," << mn.z << "]..["
                      << mx.x << "," << mx.y << "," << mx.z << "]" << std::endl;
            if (flag) ++printed;
        }
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
