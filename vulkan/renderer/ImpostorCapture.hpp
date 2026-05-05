#pragma once
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <array>

class VulkanApp;

// Captures vegetation billboard impostor views from a Fibonacci sphere grid.
// 20 evenly-distributed camera positions orbit the billboard at capture time
// for each of 3 billboard types, giving 60 layers total.
class ImpostorCapture {
public:
    static constexpr uint32_t NUM_VIEWS          = 20;
    static constexpr uint32_t NUM_BILLBOARD_TYPES = 3;
    static constexpr uint32_t TOTAL_LAYERS       = NUM_BILLBOARD_TYPES * NUM_VIEWS; // 60
    static constexpr uint32_t TEX_SIZE           = 256;
    static constexpr uint32_t NUM_INSTANCES      = 1; // single centred instance

    // Allocate all GPU resources (call once after VulkanApp is ready).
    void init(VulkanApp* app);

    // Destroy all GPU resources.
    void cleanup(VulkanApp* app);

    // Capture NUM_VIEWS impostor frames for ONE billboard type into layers
    // [billboardIndex*NUM_VIEWS .. billboardIndex*NUM_VIEWS+NUM_VIEWS-1].
    void capture(VulkanApp* app,
                 VkImageView albedoView, VkImageView normalView,
                 VkImageView opacityView, VkSampler  sampler,
                 float billboardScale, uint32_t billboardIndex = 0);

    // Convenience wrapper: capture all NUM_BILLBOARD_TYPES in one call.
    void captureAll(VulkanApp* app,
                    VkImageView albedoView, VkImageView normalView,
                    VkImageView opacityView, VkSampler  sampler,
                    float billboardScale);

    bool isReady() const { return capturedTypes > 0; }

    // ImGui-compatible descriptor set for view (billboardType, viewIdx).
    VkDescriptorSet getImGuiDescSet(uint32_t billboardType, uint32_t viewIdx) const;

    // World-space direction FROM the camera to the billboard center for view i.
    const glm::vec3& getViewDir(uint32_t viewIdx) const { return viewDirs[viewIdx]; }

    // Return the view index whose direction has the greatest dot-product with dir.
    uint32_t closestView(const glm::vec3& dir) const;

    // Full 60-layer array view – usable as sampler2DArray in scene shaders.
    VkImageView getCaptureArrayView() const { return captureArrayView; }

    // Sampler suitable for scene use (created at init).
    VkSampler   getCaptureArraySampler() const { return sceneSampler; }

private:
    // Fibonacci sphere directions (unit vectors pointing FROM camera TO center).
    std::array<glm::vec3, NUM_VIEWS> viewDirs{};

    // Capture texture array (TOTAL_LAYERS=60 layers, VK_FORMAT_R8G8B8A8_UNORM).
    VkImage        captureImage      = VK_NULL_HANDLE;
    VkDeviceMemory captureMemory     = VK_NULL_HANDLE;
    VkImageView    captureArrayView  = VK_NULL_HANDLE;          // 60-layer 2D_ARRAY view
    std::array<VkImageView, TOTAL_LAYERS> captureLayerViews{};  // per-layer 2D views

    // Depth image (single non-array, reused across all views in one submit).
    VkImage        depthImage  = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory = VK_NULL_HANDLE;
    VkImageView    depthView   = VK_NULL_HANDLE;

    // Render pass (color SHADER_READ_ONLY→STORE finalLayout=SHADER_READ_ONLY,
    //              depth  CLEAR→DONT_CARE  reused between passes).
    VkRenderPass captureRenderPass = VK_NULL_HANDLE;
    std::array<VkFramebuffer, TOTAL_LAYERS> framebuffers{};

    // Pipeline (vegetation vert/geom/frag with simple 2-set layout).
    VkPipeline            capturePipeline       = VK_NULL_HANDLE;
    VkPipelineLayout      capturePipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout uboDescSetLayout      = VK_NULL_HANDLE;
    VkDescriptorSetLayout texDescSetLayout      = VK_NULL_HANDLE;

    // Per-view camera UBO (dynamic-offset uniform buffer, NUM_VIEWS slots).
    struct alignas(16) CaptureUBO {
        glm::mat4 viewProjection;
        glm::vec4 viewPos;
        glm::vec4 lightDir;
        glm::vec4 lightColor;
    };
    VkBuffer       uboBuffer = VK_NULL_HANDLE;
    VkDeviceMemory uboMemory = VK_NULL_HANDLE;
    void*          uboMapped = nullptr;
    VkDeviceSize   uboStride = 256;  // aligned to minUniformBufferOffsetAlignment

    // Minimal vertex + instance buffers (single base vertex, one instance).
    VkBuffer       captureVertBuf = VK_NULL_HANDLE;
    VkDeviceMemory captureVertMem = VK_NULL_HANDLE;
    VkBuffer       captureInstBuf = VK_NULL_HANDLE;
    VkDeviceMemory captureInstMem = VK_NULL_HANDLE;

    // Descriptor pool + two descriptor sets (UBO dynamic + texture samplers).
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet  uboDescSet     = VK_NULL_HANDLE;
    VkDescriptorSet  texDescSet     = VK_NULL_HANDLE;

    // Push constants must match WindPushConstants layout from vegetation shaders.
    struct CapturePC {
        float     billboardScale;
        float     windEnabled;
        float     windTime;
        float     impostorDistance; // renamed from pad0; always 0 during capture
        glm::vec4 windDirAndStrength;
        glm::vec4 windNoise;
        glm::vec4 windShape;
        glm::vec4 windTurbulence;
        glm::vec4 densityParams;
        glm::vec4 cameraPosAndFalloff;
    };

    // ImGui display resources (TOTAL_LAYERS descriptor sets).
    VkSampler imguiSampler = VK_NULL_HANDLE;
    VkSampler sceneSampler = VK_NULL_HANDLE;  // for scene rendering (created at init)
    std::array<VkDescriptorSet, TOTAL_LAYERS> imguiDescSets{};

    // Bitmask of which billboard types have been captured.
    uint32_t capturedTypes = 0;
    bool initDone = false;

    void generateFibonacciDirs();
    void createCaptureImages(VulkanApp* app);
    void createDepth(VulkanApp* app);
    void createCaptureRenderPass(VulkanApp* app);
    void createFramebuffers(VulkanApp* app);
    void createDescSetLayouts(VulkanApp* app);
    void createPipeline(VulkanApp* app);
    void createUBO(VulkanApp* app);
    void createCaptureBuffers(VulkanApp* app);
    void createSceneSampler(VulkanApp* app);
    void allocateDescSets(VulkanApp* app);
    void updateTexDescSet(VkDevice device,
                          VkImageView albedo, VkImageView normal,
                          VkImageView opacity, VkSampler sampler);
    void createImGuiDescSetsForType(VulkanApp* app, uint32_t billboardType);
    void destroyImGuiDescSets();
};
