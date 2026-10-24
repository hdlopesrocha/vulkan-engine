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
// This is the only host-side consumer allowed to solve SDF cells into
// Cartesian surface points/normals (SDF::getPosition / getNormalFromPosition):
// the offline exporter has no GPU. The engine runtime stores SDF samples only;
// its Cartesian chunk vertices are written by the GPU topology pass and are
// never read back.
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
    // Output path. `.glb` writes the interchange file + sidecar + setup script;
    // `.blend` additionally runs Blender headlessly to build the scene and
    // save a ready-to-render .blend (the glb stays as the intermediate).
    std::string outputPath;
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

    // `.blend` step: Blender executable (PATH lookup; BLENDER_EXECUTABLE
    // overrides) and whether the intermediate .glb is deleted once the .blend
    // exists (the .blend packs its textures, so nothing references it).
    std::string blenderExecutable = "blender";
    bool pruneIntermediateGlb = true;

    // Optional engine camera carried into the Blender scene: the script places
    // the camera at `cameraPosition` looking at `cameraTarget` with the
    // engine's vertical FOV. Without it the script auto-frames the terrain.
    bool haveCamera = false;
    glm::vec3 cameraPosition{0.0f};
    glm::vec3 cameraTarget{0.0f};
    float cameraFovYDegrees = 60.0f;
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

    std::string outputPath;   // the .glb that was written (empty if pruned)
    std::string blendPath;    // the .blend, when one was produced
    std::string scriptPath;
    std::string configPath;
    std::string readmePath;
    std::string tileDir;
    std::string blenderLogPath;
    // Non-fatal problem (e.g. Blender missing): the glb + script are still
    // valid, the caller should surface this as a warning.
    std::string warning;
    bool blendWritten = false;
};

using ProgressFn = std::function<void(float fraction, const std::string& phase)>;

// Exports `scene` to `options.glbPath` and writes the sidecar JSON, the
// Blender setup script and a short README next to it. Thread-safe with respect
// to the scene (holds each layer's shared lock while walking it; brush edits
// that need the unique lock wait for the export to finish).
Stats exportSceneToBlender(Scene& scene, const Options& options, const Extras& extras,
                           const ProgressFn& progress = {});

} // namespace blender_export
