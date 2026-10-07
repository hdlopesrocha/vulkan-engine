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
#include <cstddef>
#include <glm/glm.hpp>
#include <atomic>
#include <cstdint>
#include <vector>

#include "../../resources/Buffer.hpp"
#include "../../ssbo/RTProxyMeta.hpp"

class VulkanApp;

// Per-op RT profiling counters. Layout mirrors the std430 block in
// shaders/includes/rt/RtProfile.glsl; op ids are the RT_PROFILE_OP_* constants
// there. The profile shader variants sample 1/4 of invocations
// (rt_profile.glsl: RT_PROFILE_SAMPLE_STRIDE), so counts and time are scaled
// by 4 on display. `time` is device-clock nanoseconds >> 6 (64 ns units),
// summed over invocations = "thread-time" (ops run concurrently; compare
// shares, not absolute frame time).
struct RTProfileCounters {
    static constexpr uint32_t kOpCount = 7;
    uint32_t rays[kOpCount] = {};  // ray queries issued
    uint32_t hits[kOpCount] = {};  // committed intersections
    uint32_t time[kOpCount] = {};  // 64 ns units, summed thread-time
};
static_assert(sizeof(RTProfileCounters) == RTProfileCounters::kOpCount * 3 * sizeof(uint32_t));

// RTProxyMeta (the GPU-side SSBO entry) lives in vulkan/ssbo/RTProxyMeta.hpp.

// CPU-side proxy source (chunk bounds already known by the scene graph).
struct RTProxyBox {
    glm::vec3 minp = glm::vec3(0.0f);
    glm::vec3 maxp = glm::vec3(0.0f);
    float materialId = 0.0f;
    float flags = 0.0f; // bit0 = water volume
    glm::vec3 albedo = glm::vec3(0.5f);
    float roughness = 0.9f;
};

// Shared ray-tracing parameters UBO (std140, 288 B). Single definition shared
// with the GLSL twin shaders/ubo/RayTracingParams.glsl — same names, fields
// and offsets; streamed per frame via mapped memcpy (handle stable).
// Flags are uint32 (GLSL bool is not a host-shareable block type).
// Member alignment is pinned with alignas to the std140 vector rules
// (vec2 -> 8, vec3/vec4/mat4 -> 16) so the layout never depends on glm packing.
struct alignas(16) RayTracingParams {
    glm::mat4 invViewProj{1.0f};        // offset   0
    glm::mat4 prevViewProj{1.0f};       // offset  64  previous frame's view-projection (temporal SSR)
    alignas(16) glm::vec3 viewPosition{0.0f}; // offset 128
    float maxReflectDistance = 500.0f;  // offset 140
    alignas(16) glm::vec3 sunDirection{0.0f, -1.0f, 0.0f}; // offset 144  xyz = direction TO sun
    float maxRefractDistance = 300.0f;  // offset 156
    alignas(16) glm::vec3 sunColor{1.0f, 1.0f, 0.9f}; // offset 160
    float maxShadowDistance = 12.0f;    // offset 172
    alignas(16) glm::vec3 absorptionColor{0.35f, 0.12f, 0.08f}; // offset 176  Beer-Lambert coeff
    float absorptionScale = 1.0f;       // offset 188
    alignas(8) glm::vec2 rtResolution{0.0f};    // offset 192  dispatch size
    alignas(8) glm::vec2 invRtResolution{0.0f}; // offset 200  1/size
    float roughnessThreshold = 0.6f;    // offset 208
    float waterIor = 1.333f;            // offset 212
    float maxWaterThickness = 6.0f;     // offset 216
    float coarseBoxSize = 0.0f;         // offset 220
    int32_t maxReflectionBounces = 1;   // offset 224
    int32_t debugMode = 0;              // offset 228  see debug_modes.glsl
    float selfSkipDist = 0.0f;          // offset 232
    float reflectionContribMin = 0.02f; // offset 236
    float nearPlane = 0.1f;             // offset 240
    float farPlane = 8092.0f;           // offset 244
    uint32_t reflectionsEnabled = 1u;   // offset 248
    uint32_t refractionsEnabled = 1u;   // offset 252
    uint32_t thicknessEnabled = 1u;     // offset 256
    uint32_t localShadowsEnabled = 0u;  // offset 260
    uint32_t tlasReady = 0u;            // offset 264
    uint32_t useWaterPipeline = 1u;     // offset 268
    uint32_t checkerboardReflections = 1u; // offset 272
    uint32_t singleRay = 1u;            // offset 276
    uint32_t waterReflections = 1u;     // offset 280
    uint32_t rayTracedWaterDepth = 0u;  // offset 284
    // 288 = struct end (multiple of 16; no tail padding needed).
};
static_assert(sizeof(RayTracingParams) == 288, "RayTracingParams must be 288 bytes");
static_assert(offsetof(RayTracingParams, invViewProj) == 0, "invViewProj offset");
static_assert(offsetof(RayTracingParams, prevViewProj) == 64, "prevViewProj offset");
static_assert(offsetof(RayTracingParams, viewPosition) == 128, "viewPosition offset");
static_assert(offsetof(RayTracingParams, maxReflectDistance) == 140, "maxReflectDistance offset");
static_assert(offsetof(RayTracingParams, sunDirection) == 144, "sunDirection offset");
static_assert(offsetof(RayTracingParams, maxRefractDistance) == 156, "maxRefractDistance offset");
static_assert(offsetof(RayTracingParams, sunColor) == 160, "sunColor offset");
static_assert(offsetof(RayTracingParams, maxShadowDistance) == 172, "maxShadowDistance offset");
static_assert(offsetof(RayTracingParams, absorptionColor) == 176, "absorptionColor offset");
static_assert(offsetof(RayTracingParams, absorptionScale) == 188, "absorptionScale offset");
static_assert(offsetof(RayTracingParams, rtResolution) == 192, "rtResolution offset");
static_assert(offsetof(RayTracingParams, invRtResolution) == 200, "invRtResolution offset");
static_assert(offsetof(RayTracingParams, roughnessThreshold) == 208, "roughnessThreshold offset");
static_assert(offsetof(RayTracingParams, waterIor) == 212, "waterIor offset");
static_assert(offsetof(RayTracingParams, maxWaterThickness) == 216, "maxWaterThickness offset");
static_assert(offsetof(RayTracingParams, coarseBoxSize) == 220, "coarseBoxSize offset");
static_assert(offsetof(RayTracingParams, maxReflectionBounces) == 224, "maxReflectionBounces offset");
static_assert(offsetof(RayTracingParams, debugMode) == 228, "debugMode offset");
static_assert(offsetof(RayTracingParams, selfSkipDist) == 232, "selfSkipDist offset");
static_assert(offsetof(RayTracingParams, reflectionContribMin) == 236, "reflectionContribMin offset");
static_assert(offsetof(RayTracingParams, nearPlane) == 240, "nearPlane offset");
static_assert(offsetof(RayTracingParams, farPlane) == 244, "farPlane offset");
static_assert(offsetof(RayTracingParams, reflectionsEnabled) == 248, "reflectionsEnabled offset");
static_assert(offsetof(RayTracingParams, refractionsEnabled) == 252, "refractionsEnabled offset");
static_assert(offsetof(RayTracingParams, thicknessEnabled) == 256, "thicknessEnabled offset");
static_assert(offsetof(RayTracingParams, localShadowsEnabled) == 260, "localShadowsEnabled offset");
static_assert(offsetof(RayTracingParams, tlasReady) == 264, "tlasReady offset");
static_assert(offsetof(RayTracingParams, useWaterPipeline) == 268, "useWaterPipeline offset");
static_assert(offsetof(RayTracingParams, checkerboardReflections) == 272, "checkerboardReflections offset");
static_assert(offsetof(RayTracingParams, singleRay) == 276, "singleRay offset");
static_assert(offsetof(RayTracingParams, waterReflections) == 280, "waterReflections offset");
static_assert(offsetof(RayTracingParams, rayTracedWaterDepth) == 284, "rayTracedWaterDepth offset");

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
        // Owning merged buffers (solid or water pools). The async AS build
        // records its own TRANSFER -> ACCEL_BUILD barrier for these (the
        // uploads that fill them are vkCmdCopyBuffer submissions), so the
        // build no longer depends on the cull CB's acquireBuffers barrier.
        VkBuffer vertexBuffer = VK_NULL_HANDLE;
        VkBuffer indexBuffer = VK_NULL_HANDLE;
        uint32_t vertexCount = 0;
        uint32_t indexCount = 0;
        uint32_t baseVertex = 0;  // element offset into the merged vertex pool
        uint32_t firstIndex = 0;  // element offset into the merged index pool
        glm::vec4 albedo = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
    };
    void setSceneGeometry(std::vector<SceneTriGeometry> solids,
                          std::vector<SceneTriGeometry> waters);
    VkBuffer getSceneGeomPrimBaseBuffer() const { return scenePrimBaseBuffer_.buffer; }
    VkBuffer getSceneGeomInfoBuffer() const { return sceneGeomInfoBuffer_.buffer; }
    VkBuffer getSceneGeomMetaBuffer() const { return sceneMetaBuffer_.buffer; }

    // Rebuild BLAS/TLAS when dirty and throttle allows. Records barriers +
    // builds into cmd (any graphics/compute queue). Returns true when a build
    // was recorded. Never blocks the CPU; never called when !supported_.
    // Why here: the rasterizer owns hi-detail tessellated geometry; this
    // stable proxy exists only for secondary rays (§6/§21).
    // True when a throttled AS rebuild is pending this frame. Advances the
    // internal frame counter (call exactly once per frame, before deciding
    // where to record the build) and applies the same guards as
    // buildIfNeeded(), so the caller can allocate/record the build on an
    // async queue without paying a command buffer when nothing changed.
    bool wantsBuild();

    bool buildIfNeeded(VulkanApp* app, VkCommandBuffer cmd);

    // Dispatch the water RT pipeline (half-res refraction/thickness only).
    // Reads current water depth + sky (descriptor arrays indexed by frameIdx),
    // writes the output pair for NEXT frame's water shading (1-frame latency,
    // same-queue ordered, no cross-frame hazard). The reflection output is
    // retired (always the invalid marker): the raster water stage resolves
    // reflections with its inline exact-triangle ray.
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

    // Per-op RT profiling (shader-instrumented, RT_PROFILE variants). One
    // host-visible counter block per in-flight frame: the shaders atomically
    // accumulate into the slot their frame's set 0 binds. Callers read the
    // slot only after its frame fence has signaled, then reset it (see
    // MyApp::preRenderPass — same discipline as the timestamp query pools).
    // No-op / null when RT is unsupported.
    VkBuffer getProfileBuffer(uint32_t frameIndex = 0) const {
        if (!supported_) return VK_NULL_HANDLE;
        return profileBuffers_[frameIndex % kParamFrames].buffer;
    }
    void resetProfile(uint32_t frameIndex);
    void readProfile(uint32_t frameIndex, RTProfileCounters& out) const;

    // Runtime RT gate: when every ray-path Settings toggle is off, nothing can
    // consume the acceleration structures, so buildIfNeeded and the proxy
    // repack are skipped entirely (no BLAS/TLAS work while RT is disabled).
    // Re-enabling forces one immediate rebuild + repack so scene edits made
    // while disabled are never traced from stale boxes/geometry.
    void setRuntimeEnabled(bool enabled);
    bool runtimeEnabled() const { return runtimeEnabled_; }
    bool consumeForceProxyRefresh();

    // Force one fresh scene-BLAS + TLAS rebuild on the next throttled build.
    // Must be called on the main thread (the upload-completion path does).
    // Rationale: the scene BLAS reads the merged vertex/index pools, and the
    // geometry uploads land asynchronously on a different queue, so a build
    // that overlaps a transfer bakes incomplete vertices/indices into the
    // BLAS for the racing chunks. Nothing else re-dirties the AS afterwards
    // (camera moves, LoD and draws never do), so a build that raced the last
    // uploads would keep those chunks broken for the whole session - visible
    // as reflections vanishing past a range while nearer chunks (uploaded
    // earlier) still mirror. Re-dirtying on every upload completion
    // guarantees a rebuild with the data resident; the >=30-frame throttle
    // coalesces the burst into at most one extra rebuild.
    void requestSceneBlasRefresh() {
        if (!supported_) return;
        sceneBlasDirty_ = true;
        dirty_ = true;
    }

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
    // Cross-thread: set from the main thread (proxy/geometry staging and
    // upload completions) and consumed in the async cull task by
    // wantsBuild()/buildIfNeeded()/recordSceneBlas().
    std::atomic<bool> dirty_{true};
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
    // SHADER_READ_ONLY otherwise) + sampler. reflectImage_ is retained as a
    // binding-compatibility marker only (written invalid by rgen; the inline
    // exact-triangle ray is the sole reflection source).
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
    Buffer paramsBuffers_[kParamFrames]{};
    // Per-frame RT profiling counter blocks (host-visible, atomically
    // incremented by the RT_PROFILE shader variants; contents read + reset by
    // the app after the slot's fence). Layout mirrors rt_profile.glsl.
    Buffer profileBuffers_[kParamFrames]{};

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
    // Cross-thread like dirty_: set on the main thread (setSceneGeometry,
    // upload completions), consumed in the cull task's recordSceneBlas().
    std::atomic<bool> sceneBlasDirty_{false};

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
