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
    glm::vec4 toggles = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f); // x=reflections y=refractions z=thickness w=localShadows
    glm::vec4 distances = glm::vec4(500.0f, 300.0f, 12.0f, 0.6f); // x=maxReflect y=maxRefract z=maxShadowDist w=roughnessThreshold
    glm::vec4 water = glm::vec4(1.333f, 0.0f, 0.0f, 0.0f); // x=IOR, yzw reserved
    glm::vec4 absorption = glm::vec4(0.35f, 0.12f, 0.08f, 1.0f); // rgb=Beer-Lambert coeff, a=thicknessScale
    glm::vec4 debug = glm::vec4(0.0f); // x=RT debug view, y=tlasReady, z=selfSkip, w=useWaterPipeline
    glm::mat4 invViewProj = glm::mat4(1.0f);
    glm::vec4 viewPos = glm::vec4(0.0f);
    glm::vec4 rtResolution = glm::vec4(0.0f); // xy=dispatch size, zw=1/size
    glm::vec4 clipPlanes = glm::vec4(0.1f, 8092.0f, 0.0f, 0.0f); // x=near, y=far
    glm::vec4 sunDir = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f); // xyz=direction TO sun
    glm::vec4 sunColor = glm::vec4(1.0f, 1.0f, 0.9f, 1.0f);
};

class RayTracingResources {
public:
    static constexpr uint32_t kMaxSolidProxies = 3584;
    static constexpr uint32_t kWaterProxyStart = 3584; // water boxes live in slots [3584, 4096)
    static constexpr uint32_t kMaxWaterProxies = 512;
    static constexpr uint32_t kMaxProxies = 4096; // total box slots (solids + water)
    static constexpr float kOutputScale = 0.5f; // half-res RT outputs (perf, §16)

    // TLAS instance masks: solid rays see everything, water-originated rays
    // see solids only (the water surface is never a target for its own rays —
    // no self-hits). RT shadows use both (shoreline contact).
    static constexpr uint32_t kMaskSolid = 0x01;
    static constexpr uint32_t kMaskWater = 0x02;
    static constexpr uint32_t kMaskAll = 0x03;

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
    // Handle stable; contents memcpy (coherent, no descriptor update).
    void updateParams(const RayTracingParams& p);

    // Accessors for descriptor wiring (SceneRenderer writes these into set 0;
    // handles stable after init except TLAS on capacity-grow recreate).
    VkAccelerationStructureKHR getTLAS() const { return tlas_; }
    VkBuffer getParamsBuffer() const { return paramsBuffer_.buffer; }
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

    // (Re)point the per-slot water-depth (D32) + sky equirect views. Called
    // once at init and on swapchain resize (handles stable otherwise).
    void setSceneViews(VulkanApp* app, const VkImageView waterDepthViews[3],
                       const VkImageView skyViews[3]);

private:
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
    Buffer tlasInstanceBuffer_{}; // 2x VkAccelerationStructureInstanceKHR
    VkDeviceAddress aabbAddress_ = 0;
    VkDeviceAddress tlasInstanceAddress_ = 0;

    std::vector<RTProxyBox> stagedSolids_;
    std::vector<RTProxyBox> stagedWaters_;
    bool dirty_ = true;
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
    Buffer paramsBuffer_{};

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
