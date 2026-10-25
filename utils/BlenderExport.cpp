#include "BlenderExport.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <unordered_set>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include "GltfWriter.hpp"
#include "GrassPlacement.hpp"
#include "SceneMaterialTable.hpp"
#include "../math/geometry/BoundingCube.hpp"
#include "../space/Octree.hpp"
#include "../space/OctreeNode.hpp"
#include "../space/OctreeTriangleTessellator.hpp"
#include "../space/Tesselator.hpp"
#include "../sdf/SDF.hpp"
#include "Scene.hpp"

namespace blender_export {
namespace {

namespace fs = std::filesystem;

// ── JSON helpers ────────────────────────────────────────────────────────────
std::string jstr(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) out += ' ';
                else out += c;
                break;
        }
    }
    out += '"';
    return out;
}

std::string jnum(float v) {
    if (!std::isfinite(v)) v = 0.0f;
    std::ostringstream ss;
    ss << v;
    std::string s = ss.str();
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
        s.find('E') == std::string::npos) {
        s += ".0";
    }
    return s;
}

std::string jvec(const glm::vec3& v) {
    return "[" + jnum(v.x) + "," + jnum(v.y) + "," + jnum(v.z) + "]";
}
std::string jvec(const glm::vec4& v) {
    return "[" + jnum(v.x) + "," + jnum(v.y) + "," + jnum(v.z) + "," + jnum(v.w) + "]";
}

// ── Images ──────────────────────────────────────────────────────────────────
bool readFileBytes(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size < 0) return false;
    in.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(size));
    if (size > 0) in.read(reinterpret_cast<char*>(out.data()), size);
    return in.good() || in.eof();
}

struct Image8 {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<uint8_t> rgba;
};

bool loadImage8(const std::string& path, Image8& out, std::string& error) {
    int w = 0, h = 0, c = 0;
    stbi_uc* pixels = stbi_load(path.c_str(), &w, &h, &c, 4);
    if (!pixels) {
        error = "cannot load texture '" + path + "'";
        return false;
    }
    out.width = w;
    out.height = h;
    out.channels = 4;
    out.rgba.assign(pixels, pixels + static_cast<size_t>(w) * h * 4);
    stbi_image_free(pixels);
    return true;
}

bool writeJpegToMemory(const Image8& image, int quality, std::vector<uint8_t>& out) {
    out.clear();
    auto callback = [](void* context, void* data, int size) {
        auto* bytes = static_cast<std::vector<uint8_t>*>(context);
        const uint8_t* p = static_cast<const uint8_t*>(data);
        bytes->insert(bytes->end(), p, p + size);
    };
    return stbi_write_jpg_to_func(callback, &out, image.width, image.height, 4,
                                  image.rgba.data(), quality) != 0;
}

inline uint8_t srgb8ToLinear8(uint8_t value) {
    const float srgb = value / 255.0f;
    const float lin = (srgb <= 0.04045f)
                          ? (srgb / 12.92f)
                          : std::pow((srgb + 0.055f) / 1.055f, 2.4f);
    int v = static_cast<int>(std::lround(lin * 255.0f));
    return static_cast<uint8_t>(std::clamp(v, 0, 255));
}

inline uint8_t linear8ToSrgb8(uint8_t value) {
    const float lin = value / 255.0f;
    const float srgb = (lin <= 0.0031308f)
                           ? (lin * 12.92f)
                           : 1.055f * std::pow(lin, 1.0f / 2.4f) - 0.055f;
    int v = static_cast<int>(std::lround(srgb * 255.0f));
    return static_cast<uint8_t>(std::clamp(v, 0, 255));
}

// ── CPU twin of shaders/computation/TextureMixer.comp ──────────────────────────────────
struct MixerNoise {
    int scale = 8;
    int octaves = 4;
    float persistence = 0.5f;
    float lacunarity = 2.0f;
    float contrast = 5.0f;
    float brightness = 0.0f;
    float time = 0.0f;
    uint32_t seed = 12345u;

    uint32_t hash(uint32_t x) const {
        x += seed;
        x = ((x >> 16) ^ x) * 0x45d9f3bu;
        x = ((x >> 16) ^ x) * 0x45d9f3bu;
        x = (x >> 16) ^ x;
        return x;
    }

    float randomUnit(uint32_t n) const {
        return float(n) / float(0xffffffffu);
    }

    static int wrapAxis(int v, int period) {
        if (period <= 0) return v;
        int r = v % period;
        if (r < 0) r += period;
        return r;
    }

    glm::vec3 gradient(glm::ivec3 p) const {
        p.x = wrapAxis(p.x, scale);
        p.y = wrapAxis(p.y, scale);
        uint32_t n = hash(uint32_t(p.x) ^ hash(uint32_t(p.y) ^ hash(uint32_t(p.z))));
        const float theta = randomUnit(n) * 6.28318530718f;
        n = hash(n);
        const float phi = randomUnit(n) * 3.14159265359f;
        return glm::vec3(std::sin(phi) * std::cos(theta),
                         std::sin(phi) * std::sin(theta),
                         std::cos(phi));
    }

    static float fade(float t) {
        return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
    }

    float perlin3D(glm::vec3 p) const {
        const float period = float(scale);
        glm::vec2 wrapped(std::fmod(p.x, period), std::fmod(p.y, period));
        if (wrapped.x < 0.0f) wrapped.x += period;
        if (wrapped.y < 0.0f) wrapped.y += period;
        const glm::vec3 pw(wrapped.x, wrapped.y, p.z);

        const glm::ivec3 p0(static_cast<int>(std::floor(pw.x)),
                            static_cast<int>(std::floor(pw.y)),
                            static_cast<int>(std::floor(pw.z)));
        const glm::vec3 f = pw - glm::vec3(p0);

        const glm::ivec3 corners[8] = {
            {p0.x, p0.y, p0.z},
            {wrapAxis(p0.x + 1, scale), p0.y, p0.z},
            {p0.x, wrapAxis(p0.y + 1, scale), p0.z},
            {wrapAxis(p0.x + 1, scale), wrapAxis(p0.y + 1, scale), p0.z},
            {p0.x, p0.y, p0.z + 1},
            {wrapAxis(p0.x + 1, scale), p0.y, p0.z + 1},
            {p0.x, wrapAxis(p0.y + 1, scale), p0.z + 1},
            {wrapAxis(p0.x + 1, scale), wrapAxis(p0.y + 1, scale), p0.z + 1},
        };
        const glm::vec3 offsets[8] = {
            {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
            {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1},
        };

        float values[8];
        for (int i = 0; i < 8; ++i) {
            values[i] = glm::dot(gradient(corners[i]), f - offsets[i]);
        }

        const float fx = fade(f.x), fy = fade(f.y), fz = fade(f.z);
        const float x00 = values[0] + (values[1] - values[0]) * fx;
        const float x10 = values[2] + (values[3] - values[2]) * fx;
        const float x01 = values[4] + (values[5] - values[4]) * fx;
        const float x11 = values[6] + (values[7] - values[6]) * fx;
        const float y0 = x00 + (x10 - x00) * fy;
        const float y1 = x01 + (x11 - x01) * fy;
        return y0 + (y1 - y0) * fz;
    }

    float fbm(glm::vec3 p) const {
        float value = 0.0f;
        float amplitude = 1.0f;
        float frequency = 1.0f;
        for (int i = 0; i < octaves; ++i) {
            value += amplitude * perlin3D(p * frequency);
            frequency *= lacunarity;
            amplitude *= persistence;
        }
        return value;
    }

    std::vector<float> build(int size) const {
        std::vector<float> noise(static_cast<size_t>(size) * size, 0.0f);
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                const glm::vec3 position(float(x) / float(size) * float(scale),
                                         float(y) / float(size) * float(scale),
                                         time * 0.1f);
                float n = fbm(position);
                n = n * 0.5f + 0.5f;
                n = (n - 0.5f) * contrast + 0.5f;
                n += brightness;
                noise[static_cast<size_t>(y) * size + x] = std::clamp(n, 0.0f, 1.0f);
            }
        }
        return noise;
    }
};

// Blends two images per the mixer equations. `albedo` maps are stored linear
// in the engine (sRGB-decoded at load), so they are decoded before the blend
// and re-encoded for the sRGB texture the Blender material expects; the data
// maps (normal/rough/ao) are blended raw.
Image8 blendMaps(const Image8& primary, const Image8& secondary,
                 const std::vector<float>& noise, int size, bool albedo) {
    Image8 out;
    out.width = size;
    out.height = size;
    out.channels = 4;
    out.rgba.resize(static_cast<size_t>(size) * size * 4);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const size_t idx = static_cast<size_t>(y) * size + x;
            const float t = noise[idx];
            const int px = std::min(x, primary.width - 1);
            const int py = std::min(y, primary.height - 1);
            const int sx = std::min(x, secondary.width - 1);
            const int sy = std::min(y, secondary.height - 1);
            const uint8_t* a = primary.rgba.data() + (static_cast<size_t>(py) * primary.width + px) * 4;
            const uint8_t* b = secondary.rgba.data() + (static_cast<size_t>(sy) * secondary.width + sx) * 4;
            uint8_t* o = out.rgba.data() + idx * 4;
            for (int c = 0; c < 3; ++c) {
                if (albedo) {
                    const float la = srgb8ToLinear8(a[c]) / 255.0f;
                    const float lb = srgb8ToLinear8(b[c]) / 255.0f;
                    const float mixed = la + (lb - la) * t;
                    const int q = static_cast<int>(std::lround(mixed * 255.0f));
                    o[c] = linear8ToSrgb8(static_cast<uint8_t>(std::clamp(q, 0, 255)));
                } else {
                    const int mixed = static_cast<int>(std::lround(a[c] + (float(b[c]) - float(a[c])) * t));
                    o[c] = static_cast<uint8_t>(std::clamp(mixed, 0, 255));
                }
            }
            o[3] = 255;
        }
    }
    return out;
}

// ── Octree walk ─────────────────────────────────────────────────────────────
struct ChunkRef {
    OctreeNode* node = nullptr;
    BoundingCube cube;
    uint32_t level = 0;
};

void collectChunks(const Octree& tree, std::vector<ChunkRef>& out) {
    out.clear();
    if (!tree.root) return;
    struct Item {
        OctreeNode* node;
        BoundingCube cube;
        uint32_t level;
    };
    std::vector<Item> stack;
    stack.reserve(4096);
    stack.push_back({tree.root, static_cast<const BoundingCube&>(tree), 0});
    while (!stack.empty()) {
        Item cur = stack.back();
        stack.pop_back();
        if (!cur.node) continue;
        // Frontier rung of a chunk cell: the finest non-overlapping mesh (the
        // rung the band test draws at distance 0). Ancestors are coarser rungs
        // covering the same columns and are intentionally skipped.
        if (cur.node->getType() == SpaceType::Surface &&
            cur.node->getChunkLod() == 1 &&
            tree.isChunkNode(cur.cube.getLengthX())) {
            ChunkRef ref;
            ref.node = cur.node;
            ref.cube = cur.cube;
            ref.level = cur.level;
            out.push_back(ref);
            continue;
        }
        OctreeNode* children[8] = {};
        cur.node->getChildren(*tree.allocator, children);
        for (int i = 7; i >= 0; --i) {
            if (children[i]) {
                stack.push_back({children[i], cur.cube.getChild(i), cur.level + 1});
            }
        }
    }
}

// The offline exporter has no GPU, so it is the ONLY place allowed to solve a
// cell's Cartesian surface point/normal on the host. The engine runtime never
// calls these functions: its chunk vertices are written by the GPU topology
// pass and are never read back. Keeping the solve behind this lambda makes the
// restriction explicit at every CPU call site.
static CellVertex exportCellVertex(OctreeNode* node, const BoundingCube& cube) {
    CellVertex v;
    float sdf[8];
    std::memcpy(sdf, node->sdf, sizeof(sdf));
    v.position = SDF::getPosition(sdf, cube);
    v.normal = SDF::getNormalFromPosition(sdf, cube, v.position);
    return v;
}

static const std::function<CellVertex(OctreeNode*, const BoundingCube&)> kExportVertexFn =
    exportCellVertex;

int finestSurfaceLod(const Octree& tree, OctreeNode* node) {
    if (!node) return -1;
    int best = -1;
    if (node->getType() == SpaceType::Surface && node->getLod() > 0) {
        best = node->getLod();
    }
    OctreeNode* children[8] = {};
    node->getChildren(*tree.allocator, children);
    for (int i = 0; i < 8; ++i) {
        const int c = finestSurfaceLod(tree, children[i]);
        if (c > 0 && (best < 0 || c < best)) best = c;
    }
    return best;
}

// Tessellates one chunk exactly like the engine's GPU topology pass: the
// chunk's ladder rung first, then the finest Surface rung present as a fallback
// so no drawable geometry is missing. The CPU walker receives the exporter's
// SDF vertex solver; it never solves a Cartesian vertex itself.
bool tessellateChunk(const Octree& tree, const ChunkRef& chunk, Geometry& geometry) {
    geometry = Geometry();
    const uint8_t chunkLod = chunk.node->getChunkLod();
    if (chunkLod == 0) return false;

    long triCount = 0;
    Tesselator tess(&triCount);
    OctreeTriangleTessellator(tree).iterateTriangles(
        chunk.node, chunk.cube, tess, chunkLod, kExportVertexFn);
    if (!tess.geometry.indices.empty()) {
        geometry = std::move(tess.geometry);
        return true;
    }

    const int fine = finestSurfaceLod(tree, chunk.node);
    if (fine > 0 && fine != chunkLod) {
        long triCount2 = 0;
        Tesselator tess2(&triCount2);
        OctreeTriangleTessellator(tree).iterateTriangles(
            chunk.node, chunk.cube, tess2, fine, kExportVertexFn);
        if (!tess2.geometry.indices.empty()) {
            geometry = std::move(tess2.geometry);
            return true;
        }
    }
    return false;
}

// ── Merged mesh accumulation ────────────────────────────────────────────────
struct MeshBuffers {
    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> uvs;
    std::vector<float> textureIndex;
    std::vector<float> hsv;
    std::vector<uint32_t> indices;

    size_t vertexCount() const { return positions.size() / 3; }
    size_t triangleCount() const { return indices.size() / 3; }
};

void appendGeometry(MeshBuffers& out, const Geometry& geometry,
                    bool withMaterial, std::set<int>* usedLayers) {
    const uint32_t base = static_cast<uint32_t>(out.vertexCount());
    out.positions.reserve(out.positions.size() + geometry.vertices.size() * 3);
    out.normals.reserve(out.normals.size() + geometry.vertices.size() * 3);
    out.uvs.reserve(out.uvs.size() + geometry.vertices.size() * 2);
    if (withMaterial) {
        out.textureIndex.reserve(out.textureIndex.size() + geometry.vertices.size());
        out.hsv.reserve(out.hsv.size() + geometry.vertices.size() * 3);
    }
    for (const Vertex& v : geometry.vertices) {
        out.positions.push_back(v.position.x);
        out.positions.push_back(v.position.y);
        out.positions.push_back(v.position.z);
        out.normals.push_back(v.normal.x);
        out.normals.push_back(v.normal.y);
        out.normals.push_back(v.normal.z);
        out.uvs.push_back(v.texCoord.x);
        out.uvs.push_back(v.texCoord.y);
        if (withMaterial) {
            const int index = std::max(0, v.brushIndex);
            out.textureIndex.push_back(float(index));
            out.hsv.push_back(v.hsv.x);
            out.hsv.push_back(v.hsv.y);
            out.hsv.push_back(v.hsv.z);
            if (usedLayers) usedLayers->insert(index);
        }
    }
    out.indices.reserve(out.indices.size() + geometry.indices.size());
    // The walker deliberately emits a shared-edge polygon from both adjacent
    // cells (watertight chunk seams), so one chunk's triangle list repeats
    // most triangles ~3-4x with identical indices after the local vertex
    // dedup. Blender does not need that redundancy — collapse exact repeats
    // per chunk (cross-chunk seam duplicates stay: they have distinct index
    // ranges and are coincident, not visible).
    struct TriKey {
        uint32_t a, b, c;
        bool operator==(const TriKey& other) const {
            return a == other.a && b == other.b && c == other.c;
        }
    };
    struct TriKeyHash {
        size_t operator()(const TriKey& k) const {
            uint64_t h = (uint64_t(k.a) * 0x9E3779B97F4A7C15ull) ^
                         (uint64_t(k.b) * 0xC2B2AE3D27D4EB4Full) ^
                         (uint64_t(k.c) * 0x165667B19E3779F9ull);
            return static_cast<size_t>(h ^ (h >> 32));
        }
    };
    std::unordered_set<TriKey, TriKeyHash> seen;
    seen.reserve(geometry.indices.size() / 3);
    // The walker emits the engine's Vulkan front-face order, which is CW
    // around the SDF gradient (outward normal) in a right-handed Y-up space.
    // glTF/Blender use CCW front faces, so swap the last two indices.
    for (size_t i = 0; i + 2 < geometry.indices.size(); i += 3) {
        uint32_t a = geometry.indices[i + 0];
        uint32_t b = geometry.indices[i + 2];
        uint32_t c = geometry.indices[i + 1];
        TriKey key = (a < b) ? ((b < c) ? TriKey{a, b, c} : ((a < c) ? TriKey{a, c, b}
                                                                  : TriKey{c, a, b}))
                             : ((a < c) ? TriKey{b, a, c} : ((b < c) ? TriKey{b, c, a}
                                                                     : TriKey{c, b, a}));
        if (!seen.insert(key).second) continue;
        out.indices.push_back(base + a);
        out.indices.push_back(base + b);
        out.indices.push_back(base + c);
    }
}

std::string materialLabel(const std::vector<SceneTextureTriple>& triples,
                          const std::vector<SceneMixerDesc>& mixers, int index) {
    if (index >= 0 && static_cast<size_t>(index) < triples.size()) {
        // Folder-per-texture layout: textures/<name>/COLOR.jpg, so the material
        // name is the parent directory. Fall back to the flat layout (strip the
        // "_COLOR" suffix from the stem) if the file is not inside a folder.
        const fs::path file = triples[index].albedo;
        const std::string folder = file.parent_path().filename().string();
        if (!folder.empty() && folder != "textures") return folder;
        std::string name = file.stem().string();
        const std::string suffix = "_COLOR";
        if (name.size() > suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            name = name.substr(0, name.size() - suffix.size());
        }
        return name;
    }
    for (const SceneMixerDesc& m : mixers) {
        if (static_cast<int>(m.target) == index) {
            const std::string a = materialLabel(triples, {}, static_cast<int>(m.primary));
            const std::string b = materialLabel(triples, {}, static_cast<int>(m.secondary));
            return "mix_" + a + "_" + b;
        }
    }
    return "layer_" + std::to_string(index);
}

// ── Script discovery ────────────────────────────────────────────────────────
std::string findBlenderScript(const std::string& explicitPath) {
    std::vector<fs::path> candidates;
    if (!explicitPath.empty()) candidates.emplace_back(explicitPath);

    const fs::path relative("scripts/blender/octree_import.py");
    candidates.push_back(relative);
    candidates.push_back(fs::path("..") / relative);

    std::error_code ec;
    const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    if (!ec && !exe.empty()) {
        const fs::path dir = exe.parent_path();
        candidates.push_back(dir / relative);
        candidates.push_back(dir / ".." / relative);
    }

    for (const fs::path& candidate : candidates) {
        if (fs::exists(candidate, ec) && fs::is_regular_file(candidate, ec)) {
            return candidate.string();
        }
    }
    return {};
}

} // namespace

Stats exportSceneToBlender(Scene& scene, const Options& options, const Extras& extras,
                           const ProgressFn& progress) {
    Stats stats;
    const auto report = [&progress](float f, const std::string& phase) {
        if (progress) progress(std::clamp(f, 0.0f, 1.0f), phase);
    };
    const auto cancelled = [&options]() {
        return options.cancel != nullptr &&
               options.cancel->load(std::memory_order_relaxed);
    };
    size_t maxGrass = options.maxGrassInstances;
    if (const char* env = std::getenv("BLENDER_EXPORT_MAX_GRASS")) {
        try {
            maxGrass = static_cast<size_t>(std::stoull(env));
        } catch (...) {
        }
    }

    if (options.outputPath.empty()) {
        stats.error = "empty output path";
        return stats;
    }

    fs::path requested(options.outputPath);
    std::string ext = requested.extension().string();
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const bool blendMode = (ext == ".blend");
    if (!blendMode && ext != ".glb") requested += ".glb";
    fs::path outDir = requested.parent_path();
    if (outDir.empty()) outDir = ".";
    const std::string stem = requested.stem().string();
    // The interchange file is always a .glb next to the requested output; a
    // .blend request runs Blender on it afterwards.
    const fs::path outPath = outDir / (stem + ".glb");
    const fs::path blendPath = outDir / (stem + ".blend");
    std::error_code ec;
    fs::create_directories(outDir, ec);
    if (ec) {
        stats.error = "cannot create output directory '" + outDir.string() + "': " + ec.message();
        return stats;
    }
    const std::vector<SceneTextureTriple> triples = sceneTextureTriples();
    const std::vector<SceneMixerDesc> mixers = sceneMixerDescs();
    const size_t materialCount =
        triples.size() + mixers.size() + kScenePaintMargin + 1; // + spare/paint slot
    const std::vector<MaterialProperties> materials =
        sceneMaterialProperties(materialCount);

    MeshBuffers solid;
    MeshBuffers water;

    struct GrassPoint {
        glm::vec3 position;
        glm::vec3 normal;
        int variant = 0;
        float rotation = 0.0f;
        float height = 1.0f;
    };
    std::vector<GrassPoint> grassPoints;

    std::set<int> usedLayers;
    size_t grassDropped = 0;

    const size_t layerCount = scene.layerCount();
    for (size_t layer = 0; layer < layerCount; ++layer) {
        if (cancelled()) {
            stats.error = "cancelled";
            return stats;
        }
        const Layer layerIndex = static_cast<Layer>(layer);
        if (!scene.layerEnabled(layerIndex)) continue;
        Octree* tree = scene.getLayerOctree(layerIndex);
        if (!tree) continue;

        const bool isWater = scene.layerRenderer(layerIndex) == LayerRendererType::Water;
        if (isWater && !options.includeWater) continue;

        report(layerCount > 0 ? float(layer) / float(layerCount) : 0.0f,
               "tessellating layer " + scene.layerName(layerIndex));

        // Hold the layer's shared lock for the whole walk: chunk pointers and
        // node data stay valid while the worker tessellates. Brush edits
        // (unique lock) wait for the export to finish.
        std::shared_lock<std::shared_mutex> treeLock(tree->treeMutex);
        std::vector<ChunkRef> chunks;
        collectChunks(*tree, chunks);

        for (const ChunkRef& chunk : chunks) {
            if (cancelled()) {
                stats.error = "cancelled";
                return stats;
            }
            Geometry geometry;
            if (!tessellateChunk(*tree, chunk, geometry)) continue;
            ++stats.chunks;

            if (isWater) {
                appendGeometry(water, geometry, false, nullptr);
            } else {
                appendGeometry(solid, geometry, true, &usedLayers);
                if (options.includeGrass) {
                    const uintptr_t nodeId = reinterpret_cast<uintptr_t>(chunk.node);
                    std::vector<uint32_t> slots;
                    grass::collectSlots(geometry, nodeId, slots);
                    std::vector<grass::Instance> placed;
                    placed.reserve(slots.size() / 3);
                    grass::placeInstances(geometry, slots, nodeId,
                                          options.billboardCount, placed);
                    for (const grass::Instance& inst : placed) {
                        GrassPoint p;
                        p.position = inst.position;
                        p.normal = inst.normal;
                        p.variant = inst.variant;
                        p.rotation = inst.rotation;
                        p.height = grass::heightScaleAt(glm::vec2(inst.position.x, inst.position.z));
                        grassPoints.push_back(p);
                    }
                }
            }
        }
    }
    // Thin uniformly over the whole map when the cap is exceeded: keeping a
    // prefix would leave whole regions bald. Position-hash order gives a
    // deterministic, spatially uniform subset.
    if (grassPoints.size() > maxGrass) {
        const size_t total = grassPoints.size();
        std::vector<std::pair<uint32_t, size_t>> keys(total);
        for (size_t i = 0; i < total; ++i) {
            keys[i] = {grass::posHash(grassPoints[i].position), i};
        }
        std::nth_element(keys.begin(), keys.begin() + maxGrass, keys.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
        std::vector<GrassPoint> kept;
        kept.reserve(maxGrass);
        for (size_t i = 0; i < maxGrass; ++i) kept.push_back(grassPoints[keys[i].second]);
        grassPoints = std::move(kept);
        grassDropped = total - maxGrass;
    }
    stats.layers = layerCount;
    stats.solidTriangles = solid.triangleCount();
    stats.solidVertices = solid.vertexCount();
    stats.waterTriangles = water.triangleCount();
    stats.grassInstances = grassPoints.size();
    stats.sdfInstances = extras.sdfInstances.size();
    stats.usedMaterials = usedLayers.size();

    if (solid.indices.empty() && water.indices.empty() && grassPoints.empty()) {
        stats.error = "nothing to export (scene has no tessellated chunks)";
        return stats;
    }

    report(0.55f, "baking textures");

    // ── Textures ────────────────────────────────────────────────────────────
    GltfWriter writer;
    struct LayerImages {
        int albedo = -1;
        int normal = -1;
        int rough = -1;
        int ao = -1;
    };
    std::vector<LayerImages> layerImages(materials.size());
    std::vector<bool> layerImageBuilt(materials.size(), false);

    // Bakes the generated mixer layers once (no source files exist for them).
    std::vector<Image8> mixedAlbedo(mixers.size()), mixedNormal(mixers.size()),
        mixedRough(mixers.size()), mixedAo(mixers.size());
    std::vector<bool> mixedBuilt(mixers.size(), false);

    auto ensureMixed = [&](size_t mixerIndex) -> bool {
        if (mixedBuilt[mixerIndex]) return true;
        const SceneMixerDesc& mixer = mixers[mixerIndex];
        if (mixer.primary >= triples.size() || mixer.secondary >= triples.size()) return false;
        const SceneTextureTriple& pa = triples[mixer.primary];
        const SceneTextureTriple& pb = triples[mixer.secondary];

        Image8 srcAlbedoA, srcAlbedoB, srcNormalA, srcNormalB, srcRoughA, srcRoughB, srcAoA, srcAoB;
        std::string err;
        if (!loadImage8(pa.albedo, srcAlbedoA, err) || !loadImage8(pb.albedo, srcAlbedoB, err) ||
            !loadImage8(pa.normal, srcNormalA, err) || !loadImage8(pb.normal, srcNormalB, err) ||
            !loadImage8(pa.rough, srcRoughA, err) || !loadImage8(pb.rough, srcRoughB, err) ||
            !loadImage8(pa.ao, srcAoA, err) || !loadImage8(pb.ao, srcAoB, err)) {
            std::cerr << "[BlenderExport] mixer layer " << mixer.target << ": " << err << "\n";
            return false;
        }
        const int size = std::max(srcAlbedoA.width, 1);
        MixerNoise noiseGen;
        noiseGen.scale = static_cast<int>(mixer.perlinScale);
        noiseGen.octaves = static_cast<int>(mixer.perlinOctaves);
        noiseGen.persistence = mixer.perlinPersistence;
        noiseGen.lacunarity = mixer.perlinLacunarity;
        noiseGen.contrast = mixer.perlinContrast;
        noiseGen.brightness = mixer.perlinBrightness;
        noiseGen.time = mixer.perlinTime;
        noiseGen.seed = mixer.perlinSeed;
        const std::vector<float> noise = noiseGen.build(size);

        mixedAlbedo[mixerIndex] = blendMaps(srcAlbedoA, srcAlbedoB, noise, size, true);
        mixedNormal[mixerIndex] = blendMaps(srcNormalA, srcNormalB, noise, size, false);
        mixedRough[mixerIndex] = blendMaps(srcRoughA, srcRoughB, noise, size, false);
        mixedAo[mixerIndex] = blendMaps(srcAoA, srcAoB, noise, size, false);
        mixedBuilt[mixerIndex] = true;
        return true;
    };

    auto ensureLayerImages = [&](int layerIndex) -> const LayerImages* {
        if (layerIndex < 0 || static_cast<size_t>(layerIndex) >= layerImages.size()) return nullptr;
        if (layerImageBuilt[layerIndex]) return &layerImages[layerIndex];

        LayerImages& out = layerImages[layerIndex];
        const std::string prefix = "layer_" + std::to_string(layerIndex);
        auto embedBytes = [&](const std::string& name, const std::vector<uint8_t>& bytes,
                              const std::string& mime) {
            return writer.addImage(name, mime, bytes);
        };

        if (static_cast<size_t>(layerIndex) < triples.size()) {
            const SceneTextureTriple& t = triples[layerIndex];
            auto addFile = [&](const std::string& path, const char* suffix, int& slot) {
                std::vector<uint8_t> bytes;
                if (readFileBytes(path, bytes) && !bytes.empty()) {
                    slot = embedBytes(prefix + "_" + suffix, bytes, "image/jpeg");
                } else {
                    std::cerr << "[BlenderExport] missing texture '" << path << "' (skipping)\n";
                }
            };
            addFile(t.albedo, "albedo", out.albedo);
            addFile(t.normal, "normal", out.normal);
            addFile(t.rough, "rough", out.rough);
            addFile(t.ao, "ao", out.ao);
        } else {
            size_t mixerIndex = mixers.size();
            for (size_t i = 0; i < mixers.size(); ++i) {
                if (mixers[i].target == static_cast<uint32_t>(layerIndex)) {
                    mixerIndex = i;
                    break;
                }
            }
            if (mixerIndex < mixers.size() && ensureMixed(mixerIndex)) {
                auto embedImage = [&](const Image8& image, const char* suffix) {
                    std::vector<uint8_t> jpeg;
                    if (writeJpegToMemory(image, 95, jpeg) && !jpeg.empty()) {
                        return embedBytes(prefix + "_" + suffix, jpeg, "image/jpeg");
                    }
                    return -1;
                };
                out.albedo = embedImage(mixedAlbedo[mixerIndex], "albedo");
                out.normal = embedImage(mixedNormal[mixerIndex], "normal");
                out.rough = embedImage(mixedRough[mixerIndex], "rough");
                out.ao = embedImage(mixedAo[mixerIndex], "ao");
            }
        }

        layerImageBuilt[layerIndex] = true;
        stats.embeddedImages += (out.albedo >= 0) + (out.normal >= 0) + (out.rough >= 0) + (out.ao >= 0);
        return &out;
    };

    for (int layerIndex : usedLayers) {
        if (layerIndex >= 0 && static_cast<size_t>(layerIndex) < materials.size()) {
            ensureLayerImages(layerIndex);
        }
    }

    // ── glTF materials / meshes ─────────────────────────────────────────────
    GltfWriter::Material terrainFallback;
    terrainFallback.name = "OctreeTerrainFallback";
    terrainFallback.baseColor = glm::vec4(0.62f, 0.62f, 0.62f, 1.0f);
    terrainFallback.roughness = 0.9f;
    const int terrainMaterial = writer.addMaterial(terrainFallback);

    GltfWriter::Material waterMaterial;
    waterMaterial.name = "OctreeWater";
    waterMaterial.baseColor = glm::vec4(0.10f, 0.32f, 0.45f, 1.0f);
    waterMaterial.roughness = 0.05f;
    waterMaterial.metallic = 0.0f;
    waterMaterial.alphaBlend = true;
    waterMaterial.alpha = 0.72f;
    const int waterMaterialIndex = writer.addMaterial(waterMaterial);

    GltfWriter::Material grassMaterial;
    grassMaterial.name = "OctreeGrassFallback";
    grassMaterial.baseColor = glm::vec4(0.25f, 0.45f, 0.12f, 1.0f);
    grassMaterial.roughness = 0.8f;
    const int grassMaterialIndex = writer.addMaterial(grassMaterial);

    std::vector<GltfWriter::Primitive> terrainPrimitives;
    if (!solid.indices.empty()) {
        GltfWriter::Primitive prim;
        prim.name = "Terrain";
        prim.mode = 4;
        prim.material = terrainMaterial;
        prim.positions = std::move(solid.positions);
        prim.normals = std::move(solid.normals);
        prim.uvs = std::move(solid.uvs);
        prim.indices = std::move(solid.indices);
        const size_t vertexCount = solid.textureIndex.size();
        prim.octreeMat.resize(vertexCount * 4);
        for (size_t i = 0; i < vertexCount; ++i) {
            prim.octreeMat[i * 4 + 0] = solid.textureIndex[i];
            prim.octreeMat[i * 4 + 1] = solid.hsv[i * 3 + 0];
            prim.octreeMat[i * 4 + 2] = solid.hsv[i * 3 + 1];
            prim.octreeMat[i * 4 + 3] = solid.hsv[i * 3 + 2];
        }
        terrainPrimitives.push_back(std::move(prim));
    }
    const int terrainMesh = writer.addMesh("OctreeTerrain", std::move(terrainPrimitives));
    const int terrainNode = writer.addNode("OctreeTerrain", terrainMesh);

    int waterNode = -1;
    if (!water.indices.empty()) {
        std::vector<GltfWriter::Primitive> waterPrimitives;
        GltfWriter::Primitive prim;
        prim.name = "Water";
        prim.mode = 4;
        prim.material = waterMaterialIndex;
        prim.positions = std::move(water.positions);
        prim.normals = std::move(water.normals);
        prim.uvs = std::move(water.uvs);
        prim.indices = std::move(water.indices);
        waterPrimitives.push_back(std::move(prim));
        const int waterMesh = writer.addMesh("OctreeWater", std::move(waterPrimitives));
        waterNode = writer.addNode("OctreeWater", waterMesh);
    }

    int grassNode = -1;
    if (!grassPoints.empty()) {
        GltfWriter::Primitive prim;
        prim.name = "GrassPoints";
        prim.mode = 0;
        prim.material = grassMaterialIndex;
        prim.positions.reserve(grassPoints.size() * 3);
        prim.normals.reserve(grassPoints.size() * 3);
        prim.veg.reserve(grassPoints.size() * 4);
        for (const GrassPoint& p : grassPoints) {
            prim.positions.push_back(p.position.x);
            prim.positions.push_back(p.position.y);
            prim.positions.push_back(p.position.z);
            prim.normals.push_back(p.normal.x);
            prim.normals.push_back(p.normal.y);
            prim.normals.push_back(p.normal.z);
            prim.veg.push_back(float(p.variant));
            prim.veg.push_back(p.rotation);
            prim.veg.push_back(p.height);
            prim.veg.push_back(0.0f);
        }
        std::vector<GltfWriter::Primitive> grassPrimitives;
        grassPrimitives.push_back(std::move(prim));
        const int grassMesh = writer.addMesh("GrassPoints", std::move(grassPrimitives));
        grassNode = writer.addNode("GrassPoints", grassMesh);
    }
    (void)terrainNode;
    (void)waterNode;
    (void)grassNode;

    std::string writeError;
    if (!writer.write(outPath.string(), writeError)) {
        stats.error = writeError;
        return stats;
    }
    stats.outputPath = outPath.string();

    report(0.75f, "writing Blender setup");

    // ── Sidecar JSON ────────────────────────────────────────────────────────
    std::ostringstream json;
    json << "{";
    json << "\"format\":\"vulkan-engine-octree-blender-export\",\"version\":1";
    json << ",\"glb\":" << jstr(outPath.filename().string());
    json << ",\"tilesDir\":" << jstr(stem + "_tiles");
    json << ",\"meshes\":{\"terrain\":\"OctreeTerrain\",\"water\":\"OctreeWater\",\"grass\":\"GrassPoints\"}";
    json << ",\"coordinateSystem\":{\"up\":\"Y\",\"handedness\":\"right\","
            "\"note\":\"World-space Y-up right-handed; the Blender glTF importer rotates to Z-up automatically.\"}";
    json << ",\"triplanar\":{\"threshold\":" << jnum(options.triplanarThreshold)
         << ",\"exponent\":" << jnum(options.triplanarExponent) << "}";
    json << ",\"light\":{\"direction\":" << jvec(options.lightDirection)
         << ",\"color\":" << jvec(options.lightColor) << "}";
    if (options.haveCamera) {
        json << ",\"camera\":{\"position\":" << jvec(options.cameraPosition)
             << ",\"target\":" << jvec(options.cameraTarget)
             << ",\"fovY\":" << jnum(options.cameraFovYDegrees) << "}";
    }
    json << ",\"water\":{\"color\":" << jvec(glm::vec3(0.10f, 0.32f, 0.45f))
         << ",\"alpha\":0.72}";
    json << ",\"grass\":{\"count\":" << grassPoints.size()
         << ",\"billboardScale\":" << jnum(options.billboardScale)
         << ",\"billboardCount\":" << options.billboardCount
         << ",\"bladeCount\":12"
         << ",\"variants\":[{\"root\":[0.075,0.21,0.045],\"tip\":[0.32,0.55,0.16]},"
            "{\"root\":[0.06,0.17,0.05],\"tip\":[0.38,0.60,0.20]},"
            "{\"root\":[0.16,0.14,0.05],\"tip\":[0.55,0.47,0.22]}]}";

    json << ",\"materials\":[";
    bool firstMaterial = true;
    for (int layerIndex : usedLayers) {
        if (layerIndex < 0 || static_cast<size_t>(layerIndex) >= materials.size()) continue;
        if (!firstMaterial) json << ",";
        firstMaterial = false;
        const LayerImages& images = layerImages[layerIndex];
        const MaterialProperties& m = materials[static_cast<size_t>(
            std::clamp<int>(layerIndex, 0, static_cast<int>(materials.size()) - 1))];
        json << "{\"index\":" << layerIndex
             << ",\"name\":" << jstr(materialLabel(triples, mixers, layerIndex))
             << ",\"images\":{"
             << "\"albedo\":" << jstr(images.albedo >= 0 ? "layer_" + std::to_string(layerIndex) + "_albedo" : "")
             << ",\"normal\":" << jstr(images.normal >= 0 ? "layer_" + std::to_string(layerIndex) + "_normal" : "")
             << ",\"rough\":" << jstr(images.rough >= 0 ? "layer_" + std::to_string(layerIndex) + "_rough" : "")
             << ",\"ao\":" << jstr(images.ao >= 0 ? "layer_" + std::to_string(layerIndex) + "_ao" : "")
             << "}";
        int mixerIndex = -1;
        for (size_t i = 0; i < mixers.size(); ++i) {
            if (mixers[i].target == static_cast<uint32_t>(layerIndex)) {
                mixerIndex = static_cast<int>(i);
                break;
            }
        }
        json << ",\"generated\":{\"primary\":" << (mixerIndex >= 0 ? int(mixers[mixerIndex].primary) : -1)
             << ",\"secondary\":" << (mixerIndex >= 0 ? int(mixers[mixerIndex].secondary) : -1) << "}";
        json << ",\"mappingMode\":" << (m.mappingMode ? "true" : "false")
             << ",\"normalFlipY\":" << (m.normalFlipY ? "true" : "false")
             << ",\"normalSwapXZ\":" << (m.normalSwapXZ ? "true" : "false")
             << ",\"invertWidth\":" << (m.invertWidth ? "true" : "false")
             << ",\"invertHeight\":" << (m.invertHeight ? "true" : "false")
             << ",\"roughnessFactor\":" << jnum(m.roughnessFactor)
             << ",\"aoFactor\":" << jnum(m.aoFactor)
             << ",\"useAO\":" << (m.useAO ? "true" : "false")
             << ",\"ambientFactor\":" << jnum(m.ambientFactor)
             << ",\"specularStrength\":" << jnum(m.specularStrength)
             << ",\"shininess\":" << jnum(m.shininess)
             << ",\"reflectionStrength\":" << jnum(m.reflectionStrength)
             << ",\"tessHeightScale\":" << jnum(m.tessHeightScale)
             << "}";
    }
    json << "]";
    json << ",\"materialCount\":" << materials.size();

    json << ",\"sdf\":{\"definitions\":[";
    for (size_t i = 0; i < extras.sdfDefinitions.size(); ++i) {
        const SdfPrimitiveInfo& d = extras.sdfDefinitions[i];
        if (i) json << ",";
        json << "{\"type\":" << d.type << ",\"params0\":" << jvec(d.params0)
             << ",\"params1\":" << jvec(d.params1) << "}";
    }
    json << "],\"materials\":[";
    for (size_t i = 0; i < extras.sdfMaterials.size(); ++i) {
        const SdfMaterialInfo& m = extras.sdfMaterials[i];
        if (i) json << ",";
        json << "{\"baseColor\":" << jvec(m.baseColor)
             << ",\"emission\":" << jvec(m.emission)
             << ",\"emissionIntensity\":" << jnum(m.emissionIntensity)
             << ",\"roughness\":" << jnum(m.roughness)
             << ",\"metallic\":" << jnum(m.metallic)
             << ",\"opacity\":" << jnum(m.opacity)
             << ",\"textureLayer\":" << jnum(m.textureLayer)
             << ",\"textureTiling\":" << jnum(m.textureTiling) << "}";
    }
    json << "],\"instances\":[";
    for (size_t i = 0; i < extras.sdfInstances.size(); ++i) {
        const SdfInstanceInfo& in = extras.sdfInstances[i];
        if (i) json << ",";
        json << "{\"position\":" << jvec(in.position)
             << ",\"rotation\":" << jvec(in.rotation)
             << ",\"scale\":" << jnum(in.scale)
             << ",\"heightScale\":" << jnum(in.heightScale)
             << ",\"radiusScale\":" << jnum(in.radiusScale)
             << ",\"intensity\":" << jnum(in.intensity)
             << ",\"seed\":" << jnum(in.seed)
             << ",\"defIdx\":" << in.defIdx
             << ",\"matIdx\":" << in.matIdx << "}";
    }
    json << "]}";

    json << ",\"stats\":{\"layers\":" << stats.layers
         << ",\"chunks\":" << stats.chunks
         << ",\"solidTriangles\":" << stats.solidTriangles
         << ",\"solidVertices\":" << stats.solidVertices
         << ",\"waterTriangles\":" << stats.waterTriangles
         << ",\"grassInstances\":" << stats.grassInstances
         << ",\"sdfInstances\":" << stats.sdfInstances << "}";
    json << "}";

    const fs::path configPath = outDir / (stem + ".octree.json");
    {
        std::ofstream config(configPath, std::ios::binary | std::ios::trunc);
        if (!config) {
            stats.error = "cannot write '" + configPath.string() + "'";
            return stats;
        }
        const std::string text = json.str();
        config.write(text.data(), static_cast<std::streamsize>(text.size()));
        if (!config.good()) {
            stats.error = "write failed for '" + configPath.string() + "'";
            return stats;
        }
    }
    stats.configPath = configPath.string();

    // ── Blender setup script ────────────────────────────────────────────────
    const std::string scriptSource = findBlenderScript(options.blenderScriptSource);
    if (scriptSource.empty()) {
        stats.error = "Blender setup script not found (scripts/blender/octree_import.py). "
                      "The .glb and sidecar were written; copy the script manually.";
        return stats;
    }
    const fs::path scriptPath = outDir / (stem + "_blender.py");
    {
        std::ifstream src(scriptSource, std::ios::binary);
        std::ofstream dst(scriptPath, std::ios::binary | std::ios::trunc);
        if (!src || !dst) {
            stats.error = "cannot copy Blender setup script to '" + scriptPath.string() + "'";
            return stats;
        }
        dst << src.rdbuf();
        if (!dst.good()) {
            stats.error = "write failed for '" + scriptPath.string() + "'";
            return stats;
        }
    }
    stats.scriptPath = scriptPath.string();
    stats.tileDir = (outDir / (stem + "_tiles")).string();

    // ── README ──────────────────────────────────────────────────────────────
    const fs::path readmePath = outDir / (stem + "_README.txt");
    {
        std::ofstream readme(readmePath, std::ios::binary | std::ios::trunc);
        if (readme) {
            readme << "Octree Blender export\n"
                   << "=====================\n\n"
                   << "Files:\n";
            if (blendMode) {
                readme << "  " << blendPath.filename().string()
                       << "   ready-to-render Blender scene (camera + sun + packed textures)\n";
            }
            readme << "  " << outPath.filename().string()
                   << "   tessellated octree (.glb, intermediate)\n"
                   << "  " << configPath.filename().string()
                   << "   material table / scene sidecar\n"
                   << "  " << scriptPath.filename().string()
                   << "   Blender importer + material builder\n"
                   << "  " << (stem + "_tiles/")
                   << "   texture tiles written by the script\n";
            if (blendMode) {
                readme << "  " << (stem + "_blender.log")
                       << "   headless Blender output (import log)\n";
            }
            readme << "\n";
            if (blendMode) {
                readme << "Open " << blendPath.filename().string()
                       << " directly: the scene is configured (materials, camera, sun) and\n"
                       << "the textures are packed, so it renders without the other files.\n\n"
                       << "To rebuild the scene from the source data:\n";
            } else {
                readme << "Usage (Blender 4.x):\n";
            }
            readme << "  blender --background --python " << scriptPath.filename().string()
                   << " -- \"" << configPath.filename().string() << "\"\n"
                   << "      [--save-blend <file.blend>] [--render <png>]\n"
                   << "  or open " << scriptPath.filename().string()
                   << " in Blender's Scripting workspace and press Run Script.\n"
                   << "Do not run it with a plain python3 interpreter: bpy/mathutils\n"
                   << "only exist inside Blender's bundled Python.\n\n"
                   << "The script imports the glb, extracts the embedded textures, builds the\n"
                   << "triplanar material blend, creates grass blades and SDF proxy objects,\n"
                   << "places the engine camera (or auto-frames the terrain when the export\n"
                   << "carried none), adds a sun and packs the textures when saving a .blend.\n\n"
                   << "Grass tufts are capped (BLENDER_EXPORT_MAX_GRASS overrides the limit);\n"
                   << "the cap is logged when it drops instances.\n";
        }
    }
    stats.readmePath = readmePath.string();

    // ── .blend step: run Blender headlessly on the setup script ────────────
    if (blendMode) {
        if (cancelled()) {
            stats.error = "cancelled";
            return stats;
        }
        report(0.9f, "running Blender (headless)");
        std::string exe = options.blenderExecutable;
        if (const char* env = std::getenv("BLENDER_EXECUTABLE")) {
            if (*env) exe = env;
        }
        const fs::path logPath = outDir / (stem + "_blender.log");
        stats.blenderLogPath = logPath.string();
        std::ostringstream cmd;
        cmd << exe << " --background --python-exit-code 1 --python '" << scriptPath.string()
            << "' -- '" << configPath.string() << "' --save-blend '" << blendPath.string()
            << "' > '" << logPath.string() << "' 2>&1";
        std::cout << "[BlenderExport] " << cmd.str() << std::endl;
        const int rc = std::system(cmd.str().c_str());
        std::error_code blendEc;
        const bool blendExists = fs::exists(blendPath, blendEc) &&
                                 fs::file_size(blendPath, blendEc) > 0;
        if (rc == 0 && blendExists) {
            stats.blendWritten = true;
            stats.blendPath = blendPath.string();
            std::cout << "[BlenderExport] wrote '" << stats.blendPath << "'" << std::endl;
            if (options.pruneIntermediateGlb) {
                std::error_code rmEc;
                if (fs::remove(outPath, rmEc)) {
                    std::cout << "[BlenderExport] removed intermediate '" << outPath.string()
                              << "'" << std::endl;
                    stats.outputPath.clear();
                }
            }
        } else {
            stats.warning = "Blender step failed (exit " + std::to_string(rc) + "); see '" +
                            logPath.string() +
                            "'. The .glb, sidecar and setup script are ready to import manually.";
            std::cerr << "[BlenderExport] " << stats.warning << std::endl;
        }
    }

    report(1.0f, "done");
    stats.ok = true;
    if (grassDropped > 0) {
        std::cerr << "[BlenderExport] grass capped at " << maxGrass
                  << " tufts; " << grassDropped << " tufts omitted\n";
    }
    const std::string wrotePath = stats.blendWritten ? stats.blendPath : stats.outputPath;
    std::cout << "[BlenderExport] wrote '" << wrotePath << "' ("
              << stats.solidTriangles << " solid tris, " << stats.waterTriangles
              << " water tris, " << stats.grassInstances << " grass tufts, "
              << stats.sdfInstances << " SDF instances, " << stats.embeddedImages
              << " embedded images)" << std::endl;
    return stats;
}

} // namespace blender_export
