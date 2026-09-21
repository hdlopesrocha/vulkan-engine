#pragma once

// Hybrid rasterization + ray tracing resources.
//
// Rasterization owns primary visibility (tessellation, displacement, LOD,
// water surface, vegetation, materials, depth, sky) and CSM owns the
// authoritative macro/directional sun shadows. This class owns the SECONDARY
// visibility representation used by hardware ray tracing:
//
//   * A stable AABB proxy BLAS/TLAS (base/proxy geometry, NOT the
//     camera-dependent tessellated/displaced raster mesh). Rebuilt only when
//     the underlying chunk set actually changes — never on camera moves, LOD
//     band switches, or tessellation-factor changes.
//   * A dedicated water RT pipeline (rgen/miss/hit) + Shader Binding Table
//     producing half-res reflection / refraction+thickness outputs.
//   * Inline ray queries (VK_KHR_ray_query) in the raster shaders sample the
//     same TLAS for solid reflections and selective local/contact shadows.
//
// Design rules (see AGENTS.md + hybrid-RT task):
//   - No per-frame acceleration-structure rebuilds from LOD/tessellation.
//   - No GPU->CPU readbacks of rendered geometry. Proxy AABBs come from the
//     CPU-side chunk bounds already owned by the scene (octree), staged
//     through host-visible buffers.
//   - Synchronization2 only; no vkDeviceWaitIdle/vkQueueWaitIdle in the frame
//     path (only init/cleanup/resize may block).
//   - Graceful fallback: when RT is unsupported every accessor reports it and
//     raster shaders sample sky/environment instead (CSM + raster keep working).

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <cstdint>
#include <vector>

#include "../Buffer.hpp"

class VulkanApp;

// One stable proxy entry per scene chunk. The BLAS stores the AABB; the
// metadata buffer (indexed by primitive ID in hit shaders / ray queries)
// carries the shading data the box alone cannot provide.
struct RTProxyMeta {
    glm::vec4 minAndMatId = glm::vec4(0.0f); // xyz = AABB min, w = material id (float)
    glm::vec4 maxAndFlags = glm::vec4(0.0f); // xyz = AABB max, w = flags (bit0 = water volume)
    glm::vec4 albedoRough = glm::vec4(0.5f, 0.5f, 0.5f, 0.9f); // rgb = avg albedo, a = roughness
    glm::vec4 extra = glm::vec4(0.0f); // reserved (metallic, emissive class, ...)
};

// CPU-side proxy source (chunk bounds already known by the scene graph).
struct RTProxyBox {
    glm::vec3 minp = glm::vec3(0.0f);
    glm::vec3 maxp = glm::vec3(0.0f);
    float materialId = 0.0f;
    float flags = 0.0f; // bit0 = water volume
    glm::vec3 albedo = glm::vec3(0.5f);
    float roughness = 0.9f;
};

// Shared ray-tracing parameters UBO. Must match shaders/includes/rt_params.glsl.
// Small (<1 KB); contents stream per frame via mapped memcpy, handle stable.
// debug.y = TLAS ready, debug.z = self-skip distance, debug.w = water RT
// pipeline selector (1 = sample pipeline outputs, 0 = inline ray queries).
struct RayTracingParams {
    glm::vec4 toggles = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f); // x=solid reflections y=refractions z=thickness w=localShadows
    glm::vec4 distances = glm::vec4(500.0f, 300.0f, 12.0f, 0.6f); // x=maxReflect y=maxRefract z=maxShadowDist w=roughnessThreshold
    glm::vec4 water = glm::vec4(1.333f, 6.0f, 0.0f, 1.0f); // x=IOR, y=maxWaterThickness (both mirrored from water layer 0 for the layer-unaware pipeline path), z=coarseBoxSize, w=maxReflectionBounces
    glm::vec4 absorption = glm::vec4(0.35f, 0.12f, 0.08f, 1.0f); // rgb=Beer-Lambert coeff, a=thicknessScale (mirrored from water layer 0; inline path reads WaterParams)
    glm::vec4 debug = glm::vec4(0.0f); // x=RT debug view, y=tlasReady, z=selfSkip, w=useWaterPipeline
    glm::mat4 invViewProj = glm::mat4(1.0f);
    glm::mat4 prevViewProj = glm::mat4(1.0f); // previous frame's view-projection (temporal SSR reprojection)
    glm::vec4 viewPos = glm::vec4(0.0f);
    glm::vec4 rtResolution = glm::vec4(0.0f); // xy=dispatch size, zw=1/size
    glm::vec4 clipPlanes = glm::vec4(0.1f, 8092.0f, 0.0f, 0.0f); // x=near, y=far
    glm::vec4 sunDir = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f); // xyz=direction TO sun
    glm::vec4 sunColor = glm::vec4(1.0f, 1.0f, 0.9f, 1.0f);
    // Ray-budget controls (runtime A/B without recompiling):
    // x=rayScaleMode (0=full-rate reference, 1=checkerboard half-rate inline),
    // y=contribMin (skip the inline ray when the lobe contribution is below),
    // z=singleRay (1=Fresnel stochastic reflection-xor-refraction for water,
    //   0=dual-trace reference), w=waterReflections (1 = water reflection rays enabled).
    glm::vec4 rayParams = glm::vec4(1.0f, 0.02f, 1.0f, 1.0f);
};

class RayTracingResources {
public:
    static constexpr uint32_t kMaxSolidProxies = 16384; // one box per 4x4 height-grid cell per chunk
    static constexpr uint32_t kWaterProxyStart = 16384; // water boxes live in slots [16384, 16896)
    static constexpr uint32_t kMaxWaterProxies = 512;
    static constexpr uint32_t kMaxProxies = 16896; // total box slots (solids + water)
    static constexpr float kOutputScale = 0.5f; // half-res RT outputs (perf, §16)
    // Per-frame RT param slots (must match VulkanApp::MAX_FRAMES_IN_FLIGHT):
    // each in-flight frame reads its own params, so the camera matrices the
    // shaders see always match the frame that recorded them.
    static constexpr uint32_t kParamFrames = 3;

    // TLAS instance masks: solid rays see everything, water-originated rays
    // see solids only (the water surface is never a target for its own rays —
    // no self-hits). RT shadows use both (shoreline contact).
    static constexpr uint32_t kMaskSolid = 0x01;
    static constexpr uint32_t kMaskWater = 0x02;
    static constexpr uint32_t kMaskAll = 0x03;
    // Real scene-geometry instances (exact chunk triangles), split by content
    // so each ray type can early-out on the nearest hit of its partition:
    // solids (mask 0x04) and the water mesh (mask 0x08) each have their own
    // TLAS instance and BLAS. Refraction traces solids only (the water BLAS
    // holds the undisplaced base surface, a phantom boundary for a downward
    // Snell ray); reflections trace both.
    static constexpr uint32_t kMaskScene = 0x04;
    static constexpr uint32_t kMaskSceneWater = 0x08;
    static constexpr uint32_t kSceneInstanceIndex = 2;
    static constexpr uint32_t kSceneWaterInstanceIndex = 3;
    // Scene lookup buffers are preallocated at init (no resize), sized for the
    // maximum number of concurrently active geometry spans. Solids and water
    // mesh share them as two partitions — solids in [0, nSolid), water in
    // [nSolid, nSolid + nWater) — while each BLAS numbers its geometries from
    // 0. Shaders offset a water instance's geometry index by nSolid, which
    // recordSceneBlas publishes in rtScenePrimBase[0] (the partition tag).
    static constexpr uint32_t kMaxSceneGeoms = 4096;

    RayTracingResources() = default;
    ~RayTracingResources() = default;

    // Create buffers, empty acceleration structures, output images, RT
    // descriptor set layout + set, pipeline + SBT. Safe to call when RT is
    // unsupported (creates nothing, supported_=false, all dispatches no-op).
    void init(VulkanApp* app, uint32_t width, uint32_t height);
    void cleanup(VulkanApp* app);
    void onSwapchainResized(VulkanApp* app, uint32_t width, uint32_t height);

    bool isSupported() const { return supported_; }
    bool isPipelineReady() const { return supported_ && pipelineReady_; }
    uint32_t proxyCount() const { return activeSolidCount_ + activeWaterCount_; }
    uint32_t solidProxyCount() const { return activeSolidCount_; }
    uint32_t waterProxyCount() const { return activeWaterCount_; }

    // Stage new proxy sets (chunk bounds from the scene). Copies to the
    // host-visible staging copies immediately; the GPU BLAS/TLAS rebuild is
    // deferred to buildIfNeeded() (throttled, in-frame, sync2-barriered).
    // Cheap: memcmp-guarded, no GPU work when unchanged. Solids occupy slots
    // [0, kMaxSolidProxies), water volumes [kWaterProxyStart, kMaxProxies).
    void setProxies(const std::vector<RTProxyBox>& solids,
                    const std::vector<RTProxyBox>& waters);

    // Real scene geometry for reflection/refraction rays: one entry per active
    // chunk, referencing the raster mesh's vertex/index spans in the shared
    // merged buffers (device addresses computed by the caller). Rebuilt
    // alongside the proxies; rays then hit the real triangles so mirror
    // positions match the scene exactly. `albedo` is the chunk's average
    // linear color for hit shading (the AS carries no material data).
    // The two lists become two scene BLASes (solid triangles mask 0x04, water
    // mesh mask 0x08); water `albedo.w` carries the water layer index and the
    // `geomInfo.w` flag (1 = water) routes hit shading to the water look.
    struct SceneTriGeometry {
        VkDeviceAddress vertexAddress = 0; // chunk's first vertex (already offset)
        VkDeviceAddress indexAddress = 0;  // chunk's first index (already offset)
        uint32_t vertexCount = 0;
        uint32_t indexCount = 0;
        uint32_t baseVertex = 0;  // element offset into the merged vertex pool
        uint32_t firstIndex = 0;  // element offset into the merged index pool
        glm::vec4 albedo = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
    };
    void setSceneGeometry(std::vector<SceneTriGeometry> solids,
                          std::vector<SceneTriGeometry> waters);
    uint32_t sceneGeometryCount() const {
        return uint32_t(sceneSolidGeoms_.size() + sceneWaterGeoms_.size());
    }
    VkBuffer getSceneGeomPrimBaseBuffer() const { return scenePrimBaseBuffer_.buffer; }
    VkBuffer getSceneGeomInfoBuffer() const { return sceneGeomInfoBuffer_.buffer; }
    VkBuffer getSceneGeomMetaBuffer() const { return sceneMetaBuffer_.buffer; }

    // Rebuild BLAS/TLAS when dirty and throttle allows. Records barriers +
    // builds into cmd (any graphics/compute queue). Returns true when a build
    // was recorded. Never blocks the CPU; never called when !supported_.
    // Why here: the rasterizer owns hi-detail tessellated geometry; this
    // stable proxy exists only for secondary rays (§6/§21).
    bool buildIfNeeded(VulkanApp* app, VkCommandBuffer cmd);

    // Dispatch the water RT pipeline (half-res reflection + refraction/
    // thickness). Reads current water depth + sky (descriptor arrays indexed
    // by frameIdx), writes the single output pair for NEXT frame's water
    // shading (1-frame latency, same-queue ordered, no cross-frame hazard).
    // No-op when !isPipelineReady(). Caller brackets with timestamps.
    void dispatchWaterRT(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIdx,
                         const glm::mat4& invViewProj, const glm::vec3& viewPos);

    // Per-frame UBO stream (toggles/distances/water/absorption/debug/view).
    // Writes only the frame's own slot: with MAX_FRAMES_IN_FLIGHT slots the
    // in-flight frames can never observe another frame's camera matrices.
    void updateParams(const RayTracingParams& p, uint32_t frameIndex);

    // Accessors for descriptor wiring (SceneRenderer writes these into set 0;
    // handles stable after init except TLAS on capacity-grow recreate).
    VkAccelerationStructureKHR getTLAS() const { return tlas_; }
    VkBuffer getParamsBuffer(uint32_t frameIndex = 0) const {
        return paramsBuffers_[frameIndex % kParamFrames].buffer;
    }
    VkBuffer getMetaBuffer() const { return metaBuffer_.buffer; }
    VkImageView getReflectionView() const { return reflectView_; }
    VkImageView getRefractionView() const { return refractView_; }
    VkSampler getLinearSampler() const { return linearSampler_; }
    VkDescriptorSetLayout getRTSetLayout() const { return rtSetLayout_; }

    // Timings (CPU wall for builds — infrequent; GPU dispatch time via
    // caller's query pool). For the debug overlay.
    float lastBuildMs() const { return lastBuildMs_; }
    uint32_t buildCount() const { return buildCount_; }
    bool tlasBuilt() const { return tlasBuilt_; }

    // Runtime RT gate: when every ray-path Settings toggle is off, nothing can
    // consume the acceleration structures, so buildIfNeeded and the proxy
    // repack are skipped entirely (no BLAS/TLAS work while RT is disabled).
    // Re-enabling forces one immediate rebuild + repack so scene edits made
    // while disabled are never traced from stale boxes/geometry.
    void setRuntimeEnabled(bool enabled);
    bool runtimeEnabled() const { return runtimeEnabled_; }
    bool consumeForceProxyRefresh();

    // (Re)point the per-slot water-depth (D32) + sky equirect views. Called
    // once at init and on swapchain resize (handles stable otherwise).
    void setSceneViews(VulkanApp* app, const VkImageView waterDepthViews[3],
                       const VkImageView skyViews[3]);

private:
    // Build the real scene-geometry BLAS + stage its shader lookup buffers.
    // Records into cmd; no-op when sceneBlasDirty_ is clear.
    bool recordSceneBlas(VulkanApp* app, VkCommandBuffer cmd);
    void createProxyBuffers(VulkanApp* app);
    void createAccelStructures(VulkanApp* app);
    void createOutputImages(VulkanApp* app, uint32_t width, uint32_t height);
    void destroyOutputImages(VulkanApp* app);
    void createRTDescriptors(VulkanApp* app);
    void createRTPipeline(VulkanApp* app);
    void destroyRTPipeline(VulkanApp* app);
    bool recordBuild(VulkanApp* app, VkCommandBuffer cmd);
    void writeRTSet(VulkanApp* app);

    bool supported_ = false;
    bool pipelineReady_ = false;
    VulkanApp* app_ = nullptr;

    // VMA suballocates, so a buffer's device address is only guaranteed the
    // alignment Vulkan reports for that buffer — NOT the stricter alignments
    // the ray-tracing build API demands (notably
    // minAccelerationStructureScratchOffsetAlignment, 256 on AMD, vs 8 on
    // llvmpipe — which is why this only ever failed on discrete/integrated
    // AMD). Every address fed to an AS build or SBT region is therefore
    // manually aligned up inside an over-allocated buffer; the *_delta_ values
    // are the (alignedBase - rawBase) byte offsets applied to CPU writes.
    static VkDeviceSize alignUpAddr(VkDeviceSize v, VkDeviceSize a) {
        return (a <= 1) ? v : (v + a - 1) & ~(a - 1);
    }
    VkDeviceSize scratchAlign_ = 256; // from accelProps (set in init)
    VkDeviceAddress blasScratchAligned_ = 0;
    VkDeviceAddress waterScratchAligned_ = 0;
    VkDeviceAddress tlasScratchAligned_ = 0;
    VkDeviceSize boxBaseDelta_ = 0;   // aabbBuffer_: verts at +delta, indices at +delta+vertBytes
    VkDeviceSize instanceDelta_ = 0;  // tlasInstanceBuffer_: instance at +delta

    // Proxy staging (host-visible, coherent) + device addresses for builds.
    // One shared box/metadata store: solids in slots [0, kMaxSolidProxies),
    // water volumes in [kWaterProxyStart, kMaxProxies). Each BLAS references
    // its own partition (self-contained vertex/index ranges).
    Buffer aabbBuffer_{}; // box-triangle soup (device address, build input)
    Buffer metaBuffer_{}; // RTProxyMeta array (storage, hit shading)
    Buffer tlasInstanceBuffer_{}; // 4x VkAccelerationStructureInstanceKHR (solid/water proxies + solid/water scene)
    VkDeviceAddress aabbAddress_ = 0;
    VkDeviceAddress tlasInstanceAddress_ = 0;

    std::vector<RTProxyBox> stagedSolids_;
    std::vector<RTProxyBox> stagedWaters_;
    bool dirty_ = true;
    bool runtimeEnabled_ = true;      // set from the Settings ray-path toggles
    bool forceProxyRefresh_ = false;  // consumed once after re-enabling RT
    uint32_t activeSolidCount_ = 0;
    uint32_t activeWaterCount_ = 0;
    bool lastBuiltValid_ = false;
    uint64_t lastBuildFrame_ = 0;
    uint64_t frameCounter_ = 0;
    float lastBuildMs_ = 0.0f;
    uint32_t buildCount_ = 0;
    bool tlasBuilt_ = false; // set after the first BLAS+TLAS build completes

    // Acceleration structures + scratch (one BLAS per layer + TLAS).
    VkAccelerationStructureKHR blas_ = VK_NULL_HANDLE;
    VkAccelerationStructureKHR blasWater_ = VK_NULL_HANDLE;
    VkAccelerationStructureKHR tlas_ = VK_NULL_HANDLE;
    Buffer blasBuffer_{};
    Buffer blasWaterBuffer_{};
    Buffer tlasBuffer_{};
    Buffer blasScratch_{};
    Buffer waterScratch_{};
    Buffer tlasScratch_{};
    VkDeviceAddress blasAddress_ = 0;
    VkDeviceAddress blasWaterAddress_ = 0;

    // Water RT outputs (single pair, half-res, GENERAL during dispatch,
    // SHADER_READ_ONLY otherwise) + sampler.
    VkImage reflectImage_ = VK_NULL_HANDLE;
    VmaAllocation reflectAlloc_ = VK_NULL_HANDLE;
    VkDeviceMemory reflectMem_ = VK_NULL_HANDLE;
    VkImageView reflectView_ = VK_NULL_HANDLE;
    VkImage refractImage_ = VK_NULL_HANDLE;
    VmaAllocation refractAlloc_ = VK_NULL_HANDLE;
    VkDeviceMemory refractMem_ = VK_NULL_HANDLE;
    VkImageView refractView_ = VK_NULL_HANDLE;
    VkSampler linearSampler_ = VK_NULL_HANDLE;
    VkImageLayout reflectLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout refractLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    uint32_t outWidth_ = 0;
    uint32_t outHeight_ = 0;

    // RT pipeline descriptors: one set per frame slot (3). Each slot's set
    // binds its own water-depth + sky views (stable per slot between resizes);
    // TLAS / outputs / params / metadata are shared. Written once at init and
    // on resize — zero steady-state updates. Dispatch binds set[frameIdx].
    VkDescriptorSetLayout rtSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool rtSetPool_ = VK_NULL_HANDLE;
    VkDescriptorSet rtSets_[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet getRTSet() const { return rtSets_[0]; }
    VkDescriptorSet getRTSetForFrame(uint32_t f) const { return rtSets_[f % 3]; }
    Buffer paramsBuffers_[kParamFrames]{};

    // ── Real scene-geometry BLASes (reflection/refraction rays) ───────────
    // One geometry per active chunk, referencing the raster mesh's spans in
    // the shared merged vertex/index buffers. Rebuilt with the proxies when
    // chunks change; solids are traced with kMaskScene, the water mesh with
    // kMaskSceneWater. Combined lookup order is solids then water:
    // primBase[0] = solid geometry count (partition tag), primBase[1+i] =
    // first primitive of combined geometry i; meta[i] = average albedo.
    std::vector<SceneTriGeometry> sceneSolidGeoms_;
    std::vector<SceneTriGeometry> sceneWaterGeoms_;
    // Staging layout (bytes) for the fragment-read lookup buffers below.
    // With 3 frames in flight, a CPU memcpy straight into those buffers at
    // record time lands while older frames still trace the PREVIOUS BLAS
    // generation — they would shade with the new generation's metadata
    // (wrong chunks/materials = 1-2 frames of corrupted reflections after
    // every rebuild). New contents are therefore memcpy'd here and published
    // with vkCmdCopyBuffer ordered in-stream (same-queue FIFO keeps older
    // frames' reads before the copies, §5 keeps frame-N fragments after).
    // Safe to reuse across builds: rebuilds are >=30 frames apart, the
    // in-flight window is 3, so every copy executes long before the staging
    // range is overwritten.
    static constexpr VkDeviceSize kStageProxyMetaSize = VkDeviceSize(kMaxProxies) * sizeof(RTProxyMeta);
    static constexpr VkDeviceSize kStagePrimBaseSize = VkDeviceSize(kMaxSceneGeoms + 1) * sizeof(uint32_t);
    static constexpr VkDeviceSize kStageGeomInfoSize = VkDeviceSize(kMaxSceneGeoms) * sizeof(glm::uvec4);
    static constexpr VkDeviceSize kStageSceneMetaSize = VkDeviceSize(kMaxSceneGeoms) * sizeof(glm::vec4);
    static constexpr VkDeviceSize kStageProxyMetaOff = 0;
    static constexpr VkDeviceSize kStagePrimBaseOff = kStageProxyMetaOff + kStageProxyMetaSize;
    static constexpr VkDeviceSize kStageGeomInfoOff = kStagePrimBaseOff + kStagePrimBaseSize;
    static constexpr VkDeviceSize kStageSceneMetaOff = kStageGeomInfoOff + kStageGeomInfoSize;
    static constexpr VkDeviceSize kStageTotalSize = kStageSceneMetaOff + kStageSceneMetaSize;
    Buffer lookupStaging_{};
    Buffer sceneBlasBuffer_{};
    VkAccelerationStructureKHR sceneBlas_ = VK_NULL_HANDLE;
    VkDeviceAddress sceneBlasAddress_ = 0;
    Buffer sceneScratch_{};
    VkDeviceSize sceneScratchSize_ = 0;
    VkDeviceSize sceneBlasSize_ = 0;
    // Water mesh partition (kMaskSceneWater rays): separate AS so opaque
    // traversal early-outs on the nearest water triangle without walking
    // solids, and vice versa.
    Buffer sceneBlasWaterBuffer_{};
    VkAccelerationStructureKHR sceneBlasWater_ = VK_NULL_HANDLE;
    VkDeviceAddress sceneBlasWaterAddress_ = 0;
    Buffer sceneScratchWater_{};
    VkDeviceSize sceneScratchWaterSize_ = 0;
    VkDeviceSize sceneBlasWaterSize_ = 0;
    Buffer scenePrimBaseBuffer_{};
    // Combined solid-then-water partition: per-geometry uvec4
    // {baseVertex, firstIndex, primBase, waterFlag (1 = water mesh)} so hit
    // shading can fetch the real triangle attributes (position/normal) from
    // the merged vertex/index buffers via barycentrics and route water hits
    // to the water look. A water instance's per-BLAS geometry index is offset
    // by scenePrimBaseBuffer_[0] (solid geometry count) before indexing.
    Buffer sceneGeomInfoBuffer_{};
    Buffer sceneMetaBuffer_{};
    uint32_t scenePrimBaseCapacity_ = 0; // in uints
    uint32_t sceneMetaCapacity_ = 0;     // in vec4s
    bool sceneBlasDirty_ = false;

    // RT pipeline + SBT.
    VkPipeline rtPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout rtPipelineLayout_ = VK_NULL_HANDLE;
    Buffer sbtBuffer_{};
    VkDeviceAddress sbtAddress_ = 0;
    VkStridedDeviceAddressRegionKHR rgenRegion_{};
    VkStridedDeviceAddressRegionKHR missRegion_{};
    VkStridedDeviceAddressRegionKHR hitRegion_{};
    VkStridedDeviceAddressRegionKHR callableRegion_{};
};
