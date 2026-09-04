#pragma once
// RayTracingRenderer — hardware ray-traced primary scene renderer
// (VK_KHR_ray_tracing_pipeline + VK_KHR_acceleration_structure).
//
// First-class rendering path (not a demo): primary visibility comes from TLAS
// traversal; shading reuses the engine's existing material SSBO, water params
// SSBO, sky UBO and texture arrays; output is an HDR storage image consumed
// by the existing post-processing/presentation pipeline.
//
// Frame flow (all Synchronization2, no device/queue waits in the loop):
//   geometry uploads -> BLAS build/update (dirty slots only)
//     -> TLAS build (only when instances changed) -> vkCmdTraceRaysKHR
//     -> HDR image (GENERAL -> SHADER_READ_ONLY) -> post-process -> present
//
// Ownership: every RT Vulkan object is owned here (output images, DS layout,
// pool/sets, pipeline + SBT, frame UBOs, AS manager) with explicit cleanup().

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "../Buffer.hpp"
#include "../TrackedHandle.hpp"
#include "AccelerationStructureManager.hpp"
#include "Renderer.hpp"

class VulkanApp;
class TextureArrayManager;
class MaterialManager;
struct UniformObject;
struct SkyUniform;
class IndirectRenderer;

class RayTracingRenderer : public Renderer {
public:
    // Runtime switch: RT_RENDERER_ENABLED.
    //  Legacy    = old rasterizer only (RT dispatch skipped, for comparison).
    //  RayTraced = hardware RT primary path (default once validated).
    //  Debug     = RT path with a debug visualization instead of shading.
    enum class Mode : int { Legacy = 0, RayTraced = 1, Debug = 2 };

    // RT debug views (rgen post-processes the payload, no extra rays).
    enum class DebugView : int {
        Shaded = 0,
        HitDistance = 1,
        WorldNormal = 2,
        MaterialId = 3,
        InstancePrimitive = 4,
        Fresnel = 5,
        WaterThickness = 6,
        Reflection = 7,
        Refraction = 8,
        Shadow = 9,
        BounceCount = 10,
    };

    struct Settings {
        Mode mode = Mode::RayTraced; // final default: ray tracing enabled
        DebugView debugView = DebugView::Shaded;
        int maxDepth = 2;            // bounded recursion (PRIMARY->REFL/REFR), 1..4
        bool shadowsEnabled = true;
        bool reflectionsEnabled = true;
        bool refractionEnabled = true;
        float waterIOR = 1.333f;
        glm::vec3 waterAbsorption = glm::vec3(0.35f, 0.15f, 0.08f); // Beer-Lambert sigma (RGB)
        float waterReflectionStrength = 1.0f;
    };

    struct Stats {
        uint32_t blasCount = 0;
        uint32_t tlasInstances = 0;
        uint32_t overlapColumns = 0; // selected columns holding >1 rung (want 0)
        uint32_t blasBuildsThisFrame = 0;
        float blasBuildMs = 0.0f;
        float tlasBuildMs = 0.0f;
        uint64_t totalBlasBuilds = 0;
        uint64_t totalTlasBuilds = 0;
        float dispatchCpuMs = 0.0f; // host-side record cost (GPU overlap, not GPU time)
        uint64_t primaryRays = 0;
        uint32_t outWidth = 0;
        uint32_t outHeight = 0;
    };

    // Frame constants uploaded per frame (std140 UBO, binding 2).
    struct FrameUBO {
        glm::mat4 viewProj{};
        glm::mat4 invViewProj{};
        glm::vec4 viewPos{};
        glm::vec4 lightDir{};
        glm::vec4 lightColor{};
        glm::vec4 rtParams{};        // x=time, y=debugMode, z=maxDepth, w=shadowsOn
        glm::vec4 waterAbsorption{}; // rgb=sigma, a=waterEnabled
        glm::vec4 waterMisc{};       // x=IOR, y=reflStrength, z=refractionOn, w=reflectionsOn
        glm::vec4 traceMask{};       // x=radiance cull mask (0xFF, or 0x01 to hide water)
        glm::vec4 triplanarParams{}; // x=threshold, y=exponent (mirrors ubo.triplanarSettings)
        glm::vec4 featureToggles{};  // x=normalMapping, y=roughnessOn, z=aoOn, w=tessellation
    };

    RayTracingRenderer() = default;
    ~RayTracingRenderer() override = default;
    RayTracingRenderer(const RayTracingRenderer&) = delete;
    RayTracingRenderer& operator=(const RayTracingRenderer&) = delete;

    // init() is idempotent across calls with unchanged resources; returns false
    // when the device lacks RT support (caller keeps the rasterizer).
    // Does NOT compile shaders when unavailable.
    bool init(VulkanApp* app, TextureArrayManager* textures, MaterialManager* materials,
              IndirectRenderer* solidIR, IndirectRenderer* waterIR,
              const Buffer& skyUBO, const Buffer& waterParamsBuffer, uint32_t waterParamCount);
    void cleanup(VulkanApp* app) override;
    void onSwapchainResized(VulkanApp* app, uint32_t width, uint32_t height);

    // Re-point static descriptors at new texture/material/sky resources
    // (allocation listener / resize path — never in the render loop).
    void refreshStaticDescriptors(VulkanApp* app);

    // Record AS sync/build + ray dispatch for frame `frameIndex` into cmd.
    // `time` drives animated water params on the host side only.
    // `traceCullMask` is the radiance-ray instance mask (0xFF, or 0x01 to hide
    // the water layer, e.g. when the user disables water rendering).
    // `lodBias`/`maxTargetLod` mirror the GPU cull's LoD band selection so the
    // TLAS instances exactly the rung the rasterizer would draw per region.
    void renderFrame(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                     const UniformObject& ubo, float time, uint32_t traceCullMask = 0xFF,
                     float lodBias = 8.0f, uint32_t maxTargetLod = 16);

    VkImage outputImage(uint32_t frameIndex) const;
    VkImageView outputView(uint32_t frameIndex) const;
    // NDC depth in the raster depth space (for the post-process obstacle tests).
    VkImageView depthView(uint32_t frameIndex) const;
    uint32_t outputWidth() const { return outWidth_; }
    uint32_t outputHeight() const { return outHeight_; }

    bool isAvailable() const { return available_; }
    bool isPipelineReady() const { return pipeline_ != VK_NULL_HANDLE; }
    // Primary-path predicate: RT output replaces the raster solid/water color.
    bool useAsPrimary() const { return available_ && settings.mode != Mode::Legacy && isPipelineReady(); }
    bool debugActive() const { return available_ && settings.mode == Mode::Debug; }
    int activeDebugMode() const;

    AccelerationStructureManager& asManager() { return asManager_; }
    const Settings& getSettings() const { return settings; }
    Settings& mutableSettings() { return settings; }
    Stats stats() const { return lastStats_; }

    // ImGui panel (no-op without USE_IMGUI). Called from the app's renderImGui.
    void drawUI();

private:
    bool createOutputImages(VulkanApp* app, uint32_t w, uint32_t h);
    void destroyOutputImages(VulkanApp* app);
    bool createDescriptorLayout(VulkanApp* app);
    bool createDescriptorPoolAndSets(VulkanApp* app);
    void writeStaticDescriptors(VulkanApp* app);
    void writeTlasDescriptor(VulkanApp* app, uint32_t frameIndex);
    bool createFrameUBOs(VulkanApp* app);
    bool createPipeline(VulkanApp* app);
    bool createShaderBindingTable(VulkanApp* app);
    void recordOutputBarriers(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex, bool beforeTrace);

    Settings settings{};
    bool available_ = false;
    VulkanApp* app_ = nullptr;

    IndirectRenderer* solidIR_ = nullptr;
    IndirectRenderer* waterIR_ = nullptr;
    TextureArrayManager* textures_ = nullptr;
    MaterialManager* materials_ = nullptr;
    Buffer skyUBO_{};
    Buffer waterParamsBuf_{};
    uint32_t waterParamCount_ = 0;

    AccelerationStructureManager asManager_;

    static constexpr uint32_t kFrames = 3;
    static constexpr VkFormat kOutputFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    static constexpr VkFormat kDepthFormat = VK_FORMAT_R32_SFLOAT;
    uint32_t outWidth_ = 0;
    uint32_t outHeight_ = 0;
    struct OutputTarget {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    };
    std::array<OutputTarget, kFrames> outputs_{};
    // NDC depth companion of outputs_ (same frame indexing/layout tracking).
    std::array<OutputTarget, kFrames> depthOutputs_{};

    TrackedHandle<VkDescriptorSetLayout> dsLayout_;
    TrackedHandle<VkDescriptorPool> dsPool_;
    std::array<VkDescriptorSet, kFrames> dsSets_{};
    std::array<VkAccelerationStructureKHR, kFrames> lastWrittenTlas_{};

    std::array<Buffer, kFrames> frameUBOs_{};

    TrackedHandle<VkPipeline> pipeline_;
    TrackedHandle<VkPipelineLayout> pipelineLayout_;
    Buffer sbtBuffer_{};
    VkDeviceAddress sbtRaygenAddr_ = 0;
    VkDeviceAddress sbtMissAddr_ = 0;
    VkDeviceAddress sbtHitAddr_ = 0;
    uint32_t sbtHandleSizeAligned_ = 0;
    uint32_t sbtMissStride_ = 0;
    uint32_t sbtHitStride_ = 0;

    Stats lastStats_{};
};
