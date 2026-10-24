#pragma once
#include "../Renderer.hpp"
#include <vulkan/vulkan.h>
#include <cstdint>
#include "../../core/VmaContext.hpp"
#include "../../core/TrackedHandle.hpp"
#include "../../resources/Buffer.hpp"
#include <glm/glm.hpp>
#include <array>
#include "../CommandBufferState.hpp"
#include "../../ubo/CaptureUBO.hpp"

class VulkanApp;
class VegetationRenderer;

// Captures vegetation billboard impostor views from a Fibonacci sphere grid.
// 20 evenly-distributed camera positions orbit the billboard at capture time
// for each of 3 vegetation billboard types, giving 60 layers total.
class ImpostorCapture : public Renderer {
public:
    static constexpr uint32_t NUM_VIEWS          = 20;
    static constexpr uint32_t NUM_BILLBOARD_TYPES = 3;
    static constexpr uint32_t TOTAL_LAYERS       = NUM_BILLBOARD_TYPES * NUM_VIEWS; // 60
    static constexpr uint32_t TEX_SIZE           = 256;
    static constexpr uint32_t NUM_INSTANCES      = 1; // single centred instance
    // Mip chain for the albedo/normal arrays (linear blits): minified
    // impostor quads must sample lod-appropriate footprints instead of a
    // 2x2 tent over hundreds of texels, which washed blade texels toward
    // the cleared background (blue-shifted normals, discarded alpha).
    // Captured L0 content is already premultiplied-form (opaque texels carry
    // alpha 1, background 0), so plain linear averaging stays correct and
    // the runtime divides albedo/normal by the mipped alpha. The depth
    // array stays single-level and is always fetched exactly (texelFetch).
    static constexpr uint32_t kCaptureMips = 9; // 256 -> 1

    // Allocate all GPU resources (call once after VulkanApp is ready).
    // The vegetation renderer supplies the shared set=2 wind params descriptor
    // set/layout used by the capture pipeline (see finding: duplicate wind
    // params descriptor trio was created here before).
    void init(VulkanApp* app, VegetationRenderer* vegRenderer = nullptr);

    // Destroy all GPU resources.
    void cleanup(VulkanApp* app) override;

    // H8 (perf report 26): async whole-array capture. Stages all three types
    // then issues ONE graphics-queue submit for all 60 views (type order,
    // shared type-independent UBO slots, per-type instance slots selected via
    // firstInstance); the fence signals when every render has completed and
    // the layers sit in SHADER_READ_ONLY_OPTIMAL. ImGui sets + the
    // capturedTypes bit publish in a fence callback (pumped by
    // processPendingCommandBuffers). Contract: wait the fence before the next
    // capture and before first sampling (setImpostorData / scene use) — frame
    // fences do NOT cover it. Await with VulkanApp::waitFence.
    VkFence captureAllAsync(VulkanApp* app,
                            VkImageView albedoView, VkImageView normalView,
                            VkImageView opacityView, VkSampler sampler,
                            float billboardScale);

    bool isReady() const { return capturedTypes > 0; }

    // ImGui-compatible descriptor set for view (billboardType, viewIdx).
    VkDescriptorSet getImGuiDescSet(uint32_t billboardType, uint32_t viewIdx) const;

    // World-space direction FROM the camera to the billboard center for view i.
    const glm::vec3& getViewDir(uint32_t viewIdx) const { return viewDirs[viewIdx]; }

    // Return the view index whose direction has the greatest dot-product with dir.
    uint32_t closestView(const glm::vec3& dir) const;

    // Full 60-layer array views – usable as sampler2DArray in scene shaders.
    VkImageView getCaptureArrayView()          const { return captureArrayView; }
    VkImageView getCaptureNormalArrayView()    const { return captureNormalArrayView; }
    VkImageView getCaptureDepthArrayView()     const { return captureDepthArrayView; }

    // Per-layer capture inverse VP matrices for depth reprojection in the shadow pass.
    VkBuffer         getCaptureInvVPBuffer() const { return captureInvVPBuffer; }

    // Sampler suitable for scene use (created at init).
    VkSampler   getCaptureArraySampler() const { return sceneSampler; }

    // ImGui-compatible descriptor set for normal map view (billboardType, viewIdx).
    VkDescriptorSet getImGuiNormalDescSet(uint32_t billboardType, uint32_t viewIdx) const;

    // ImGui-compatible descriptor set for depth view (billboardType, viewIdx).
    VkDescriptorSet getImGuiDepthDescSet(uint32_t billboardType, uint32_t viewIdx) const;

private:
    // Fibonacci sphere directions (unit vectors pointing FROM camera TO center).
    std::array<glm::vec3, NUM_VIEWS> viewDirs{};

    // Albedo capture texture array (TOTAL_LAYERS=60 layers, VK_FORMAT_R8G8B8A8_UNORM).
    VkImage        captureImage      = VK_NULL_HANDLE;
    VmaAllocation  captureAllocation = VK_NULL_HANDLE;
    VkDeviceMemory captureMemory     = VK_NULL_HANDLE;
    VkImageView    captureArrayView  = VK_NULL_HANDLE;
    std::array<VkImageView, TOTAL_LAYERS> captureLayerViews{};

    // Normal capture texture array (world-space normals encoded in [0,1]).
    VkImage        captureNormalImage      = VK_NULL_HANDLE;
    VmaAllocation  captureNormalAllocation = VK_NULL_HANDLE;
    VkDeviceMemory captureNormalMemory     = VK_NULL_HANDLE;
    VkImageView    captureNormalArrayView  = VK_NULL_HANDLE;
    std::array<VkImageView, TOTAL_LAYERS> captureNormalLayerViews{};

    // Depth capture texture array (device Z for shadow-map reprojection).
    VkImage        captureDepthImage      = VK_NULL_HANDLE;
    VmaAllocation  captureDepthAllocation = VK_NULL_HANDLE;
    VkDeviceMemory captureDepthMemory     = VK_NULL_HANDLE;
    VkImageView    captureDepthArrayView  = VK_NULL_HANDLE;
    std::array<VkImageView, TOTAL_LAYERS> captureDepthLayerViews{};

    // Depth image (single non-array, reused across all views in one submit).
    VkImage        depthImage  = VK_NULL_HANDLE;
    VmaAllocation  depthAllocation = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory = VK_NULL_HANDLE;
    VkImageView    depthView   = VK_NULL_HANDLE;

    // Pipeline (vegetation vert/geom/frag with simple 2-set layout).
    TrackedHandle<VkPipeline> capturePipeline;
    TrackedHandle<VkPipelineLayout> capturePipelineLayout;
    TrackedHandle<VkDescriptorSetLayout> uboDescSetLayout;
    TrackedHandle<VkDescriptorSetLayout> texDescSetLayout;

    // Per-view camera UBO (dynamic-offset uniform buffer, NUM_VIEWS slots);
    // canonical layout in vulkan/ubo/CaptureUBO.hpp.
    VkBuffer       uboBuffer = VK_NULL_HANDLE;
    void*          uboMapped = nullptr;
    // L14 (perf report 26): already the tightest legal stride — the align-up
    // of sizeof(CaptureUBO) to minUniformBufferOffsetAlignment computed in
    // init(). Packing views any tighter would break the dynamic-offset
    // alignment requirement, so this must NOT be reduced to sizeof(CaptureUBO).
    VkDeviceSize   uboStride = 256;  // aligned to minUniformBufferOffsetAlignment

    // Shared set=2 wind params descriptor set + layout, owned by the
    // VegetationRenderer (wind is disabled during capture, so the set is only
    // bound for pipeline compatibility — its content is never read).
    VegetationRenderer* sharedVegRenderer = nullptr;

    // Minimal vertex + instance buffers: 24-corner base mesh plus one vec4
    // instance slot per billboard type (draws select via firstInstance).
    VkBuffer       captureVertBuf = VK_NULL_HANDLE;
    VkBuffer       captureInstBuf = VK_NULL_HANDLE;
    void*          captureInstMapped = nullptr;
    VkBuffer       captureIdxBuf = VK_NULL_HANDLE;
    uint32_t       captureIdxCount = 0;

    // Descriptor pool + two descriptor sets (UBO dynamic + texture samplers).
    TrackedHandle<VkDescriptorPool> descriptorPool;
    TrackedHandle<VkDescriptorSet> uboDescSet;
    TrackedHandle<VkDescriptorSet> texDescSet;

    // Push constants (16 bytes) — matches the vegetation shader layout.
    // Wind parameters are in the WindParamsUBO at set=2.
    struct CapturePC {
        float     billboardScale;
        float     windEnabled;      // 0 = disabled during capture
        float     windTime;
        float     impostorDistance; // always 0 during capture
    };

    // ImGui display resources (TOTAL_LAYERS descriptor sets each for albedo and normals).
    TrackedHandle<VkSampler> imguiSampler;
    TrackedHandle<VkSampler> sceneSampler;
    std::array<TrackedHandle<VkDescriptorSet>, TOTAL_LAYERS> imguiDescSets;
    std::array<TrackedHandle<VkDescriptorSet>, TOTAL_LAYERS> imguiNormalDescSets;
    std::array<TrackedHandle<VkDescriptorSet>, TOTAL_LAYERS> imguiDepthDescSets;

    // Per-layer capture inverse VP matrices for depth reprojection.
    std::array<glm::mat4, TOTAL_LAYERS> captureInvVP{};

    // Buffer containing captureInvVP data for GPU access in depth pass.
    VkBuffer       captureInvVPBuffer = VK_NULL_HANDLE;
    void*          captureInvVPMapped = nullptr;

    // Bitmask of which billboard types have been captured.
    uint32_t capturedTypes = 0;
    bool initDone = false;

    void generateFibonacciDirs();
    void createCaptureImages(VulkanApp* app);
    void createDepth(VulkanApp* app);
    void createDescSetLayouts(VulkanApp* app);
    void createPipeline(VulkanApp* app);
    void createUBO(VulkanApp* app);
    void createCaptureBuffers(VulkanApp* app);
    void createCaptureInvVPBuffer(VulkanApp* app);
    void createSceneSampler(VulkanApp* app);
    void allocateDescSets(VulkanApp* app);
    void createImGuiDescSetsForType(VulkanApp* app, uint32_t billboardType);
    // H8 async-capture internals (see ImpostorCapture.cpp): shared camera-UBO
    // upload, per-view recorder (caller-agnostic command stream), async submit
    // over a contiguous type range, and post-fence publish.
    void writeCaptureCameraUbos(float billboardScale, uint32_t layerBase);
    void recordCaptureViews(VkCommandBuffer cb, VulkanApp* app, uint32_t layerBase,
                            uint32_t viewCount, uint32_t firstInstance, const CapturePC& pc);
    // Generate the albedo/normal mip chains for [layerBase, +viewCount) with
    // linear blits, then transition the whole touched range (all levels) to
    // SHADER_READ_ONLY. Leaves the app layout tracker untouched: all
    // transitions for these two images are recorded here explicitly, and no
    // recordTransitionImageLayoutLayer call must be added for them afterwards
    // (its per-(image,layer) tracking cannot represent per-level states).
    void generateMipmaps(VkCommandBuffer cb, VkImage image, VkFormat format,
                         uint32_t layerBase, uint32_t layerCount);
    VkFence submitCaptureTypesAsync(VulkanApp* app, const CapturePC& pc,
                                    uint32_t typeBegin, uint32_t typeCount);
    void completeCaptureType(VulkanApp* app, uint32_t billboardType, float billboardScale);
public:
    void destroyImGuiDescSets();
    void invalidateImGuiDescriptors() { destroyImGuiDescSets(); }
    void recreateAllImGuiDescSets(VulkanApp* app);
};
