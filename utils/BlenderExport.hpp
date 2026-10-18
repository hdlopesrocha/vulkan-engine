#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "../types/MaterialProperties.hpp"

class Scene;

// Offline Blender export of the tessellated octree.
//
// Pipeline (Vulkan-free; runs on a worker thread or headless):
//   1. Walk every scene layer and rebuild each chunk's tessellated geometry
//      with the engine's own CPU walker (OctreeTriangleTessellator +
//      Tesselator) — the reference implementation the GPU topology pass
//      mirrors. No GPU readback is involved.
//   2. Re-run the renderer's deterministic grass placement
//      (utils/GrassPlacement) on the same chunk geometry and export the tufts
//      as a point cloud carrying variant/rotation/height attributes.
//   3. Bake the generated mixer layers (Perlin blends of two source layers)
//      on the CPU with the same equations as shaders/TextureMixer.comp.
//   4. Serialize the merged meshes, embedded textures, app-generated SDF
//      effect instances and a JSON sidecar, and copy the Blender setup script
//      next to the .glb.
//
// The Blender script builds the real materials (triplanar blending, per-vertex
// material indices, normal/roughness/AO maps, grass blades, SDF proxies). The
// glb keeps a plain PBR fallback so generic viewers can open it.

namespace blender_export {

struct SdfPrimitiveInfo {
    uint32_t type = 0;       // SdfPrimitiveType
    glm::vec4 params0{0.0f};
    glm::vec4 params1{0.0f};
};

struct SdfMaterialInfo {
    glm::vec4 baseColor{1.0f};
    glm::vec3 emission{0.0f};
    float emissionIntensity = 0.0f;
    float roughness = 0.5f;
    float metallic = 0.0f;
    float opacity = 1.0f;
    float textureLayer = -1.0f;  // terrain texture layer for surface SDFs
    float textureTiling = 64.0f; // world metres per texture repeat
};

struct SdfInstanceInfo {
    glm::vec3 position{0.0f};
    glm::vec3 rotation{0.0f}; // euler radians, R = Rx*Ry*Rz
    float scale = 1.0f;
    float heightScale = 1.0f;
    float radiusScale = 1.0f;
    float intensity = 1.0f;
    float seed = 0.0f;
    uint32_t defIdx = 0;
    uint32_t matIdx = 0;
};

struct Extras {
    // sdf_gpu effect instances (flames, smoke); filled by the app from
    // the renderer's CPU-side SdfScene. Empty in headless exports.
    std::vector<SdfPrimitiveInfo> sdfDefinitions;
    std::vector<SdfMaterialInfo> sdfMaterials;
    std::vector<SdfInstanceInfo> sdfInstances;
};

struct Options {
    std::string glbPath; // output path (extension .glb)
    // Optional cooperative cancellation (checked between chunks/layers).
    std::atomic<bool>* cancel = nullptr;
    bool includeWater = true;
    bool includeGrass = true;
    bool includeSdf = true;

    // Vegetation runtime settings that shape the deterministic placement.
    uint32_t billboardCount = 3;
    float billboardScale = 10.0f;
    size_t maxGrassInstances = 120000;

    // Optional explicit path of scripts/blender/octree_import.py. When empty
    // the exporter probes the usual locations relative to the working dir and
    // the executable.
    std::string blenderScriptSource;

    glm::vec3 lightDirection{-0.35f, -0.86f, -0.37f};
    glm::vec3 lightColor{1.0f, 1.0f, 0.9f};
    float triplanarThreshold = 0.12f;
    float triplanarExponent = 1.0f;
};

struct Stats {
    bool ok = false;
    std::string error;

    size_t layers = 0;
    size_t chunks = 0;
    size_t solidTriangles = 0;
    size_t solidVertices = 0;
    size_t waterTriangles = 0;
    size_t grassInstances = 0;
    size_t sdfInstances = 0;
    size_t usedMaterials = 0;
    size_t embeddedImages = 0;

    std::string glbPath;
    std::string scriptPath;
    std::string configPath;
    std::string readmePath;
    std::string tileDir;
};

using ProgressFn = std::function<void(float fraction, const std::string& phase)>;

// Exports `scene` to `options.glbPath` and writes the sidecar JSON, the
// Blender setup script and a short README next to it. Thread-safe with respect
// to the scene (holds each layer's shared lock while walking it; brush edits
// that need the unique lock wait for the export to finish).
Stats exportSceneToBlender(Scene& scene, const Options& options, const Extras& extras,
                           const ProgressFn& progress = {});

} // namespace blender_export
