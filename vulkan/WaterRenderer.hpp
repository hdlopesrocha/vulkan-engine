#pragma once

#include "VulkanApp.hpp"
#include "IndirectRenderer.hpp"
#include <glm/glm.hpp>

// Water rendering parameters (CPU-side)
struct WaterParams {
    float time = 0.0f;
    float waveSpeed = 0.5f;
    float waveScale = 0.02f;
    float refractionStrength = 0.03f;
    float fresnelPower = 5.0f;
    float transparency = 0.7f;
    glm::vec3 shallowColor = glm::vec3(0.1f, 0.4f, 0.5f);
    glm::vec3 deepColor = glm::vec3(0.0f, 0.15f, 0.25f);
    float depthFalloff = 0.1f;
    int noiseOctaves = 4;
    float noisePersistence = 0.5f;
    float noiseScale = 8.0f;
    float waterTint = 0.3f;
    float foamDepthThreshold = 2.0f;
    float noiseTimeSpeed = 1.0f;

    // Shore/foam tuning
    float shoreStrength = 1.0f;   // multiplies shore foam contribution
    float shoreFalloff = 4.0f;    // meters over which shore foam fades
    float foamIntensity = 0.25f;  // intensity for procedural foam

    // Foam Perlin controls
    float foamNoiseScale = 4.0f;         // larger = coarser foam patterns
    int foamNoiseOctaves = 3;
    float foamNoisePersistence = 0.5f;
    glm::vec3 foamTint = glm::vec3(0.9f, 0.95f, 1.0f);
    float foamTintIntensity = 1.0f; // full control from widget by default
};

// GPU-side water params UBO (matches shader WaterParamsUBO layout)
struct WaterParamsGPU {
    glm::vec4 params1;  // x=refractionStrength, y=fresnelPower, z=transparency, w=foamDepthThreshold
    glm::vec4 params2;  // x=waterTint, y=noiseScale, z=noiseOctaves, w=noisePersistence
    glm::vec4 params3;  // x=noiseTimeSpeed, y=waterTime, z=shoreStrength, w=shoreFalloff
    glm::vec4 shallowColor;
    glm::vec4 deepColor; // w = foamIntensity
    glm::vec4 foamParams; // x=foamNoiseScale, y=foamNoiseOctaves, z=foamNoisePersistence, w=foamTintIntensity
    glm::vec4 foamTint;   // rgb foam tint, w unused
};

// GPU-side water uniform buffer
struct WaterUBO {
    glm::mat4 viewProjection;
    glm::mat4 invViewProjection;
    glm::vec4 viewPos;
    glm::vec4 waterParams1;  // time, waveSpeed, waveScale, refractionStrength
    glm::vec4 waterParams2;  // fresnelPower, transparency, depthFalloff, noiseScale
    glm::vec4 shallowColor;  // rgb + padding
    glm::vec4 deepColor;     // rgb + noiseOctaves
    glm::vec4 screenSize;    // width, height, 1/width, 1/height
    glm::vec4 noisePersistence; // noisePersistence, padding...
};

class WaterRenderer {
public:
    WaterRenderer(VulkanApp* app);
    ~WaterRenderer();

    void init();
    void cleanup();

    // Create offscreen render targets for water rendering
    void createRenderTargets(uint32_t width, uint32_t height);
    void destroyRenderTargets();

    // Get the indirect renderer for water meshes
    IndirectRenderer& getIndirectRenderer() { return waterIndirectRenderer; }

    // Begin water geometry pass (renders water depth/normals to offscreen target)
    void beginWaterGeometryPass(VkCommandBuffer cmd, uint32_t frameIndex);
    void endWaterGeometryPass(VkCommandBuffer cmd);

    // Render water post-process to swapchain (apply refraction)
    // This starts and ends its own render pass that outputs to the provided swapchain framebuffer
    void renderWaterPostProcess(VkCommandBuffer cmd, VkFramebuffer swapchainFramebuffer,
                                 VkRenderPass swapchainRenderPass,
                                 VkImageView sceneColorView, VkImageView sceneDepthView,
                                 const WaterParams& params,
                                 const glm::mat4& viewProj, const glm::mat4& invViewProj,
                                 const glm::vec3& viewPos, float time);

    // Get water framebuffer for geometry pass
    VkFramebuffer getWaterFramebuffer(uint32_t frameIndex) const { return waterFramebuffers[frameIndex]; }
    VkRenderPass getWaterRenderPass() const { return waterRenderPass; }

    // Get water depth/normal/mask images for post-process sampling
    VkImageView getWaterDepthView() const { return waterDepthImageView; }
    VkImageView getWaterNormalView() const { return waterNormalImageView; }
    VkImageView getWaterMaskView() const { return waterMaskImageView; }
    
    // Get scene offscreen target views (for rendering scene before water)
    VkImage getSceneColorImage(uint32_t frameIndex) const { return sceneColorImages[frameIndex]; }
    VkImageView getSceneColorView(uint32_t frameIndex) const { return sceneColorImageViews[frameIndex]; }
    VkImage getSceneDepthImage(uint32_t frameIndex) const { return sceneDepthImages[frameIndex]; }
    VkImageView getSceneDepthView(uint32_t frameIndex) const { return sceneDepthImageViews[frameIndex]; }
    VkFramebuffer getSceneFramebuffer(uint32_t frameIndex) const { return sceneFramebuffers[frameIndex]; }
    VkRenderPass getSceneRenderPass() const { return sceneRenderPass; }

    // Update parameters
    void setParams(const WaterParams& params) { this->params = params; }
    WaterParams& getParams() { return params; }
    
    // Check if post-process pipeline is ready
    bool isPostProcessReady() const { return waterPostProcessPipeline != VK_NULL_HANDLE; }
    
    // Get the water geometry pipeline (for rendering water to G-buffer)
    VkPipeline getWaterGeometryPipeline() const { return waterGeometryPipeline; }
    
    // Get the water geometry pipeline layout
    VkPipelineLayout getWaterGeometryPipelineLayout() const { return waterGeometryPipelineLayout; }
    
    // Get water depth descriptor set (for binding scene depth texture)
    VkDescriptorSet getWaterDepthDescriptorSet(uint32_t frameIndex) const { return waterDepthDescriptorSets[frameIndex]; }
    
    // Get water params buffer
    Buffer& getWaterParamsBuffer() { return waterParamsBuffer; }
    
    // Get sampler for ImGui texture display
    VkSampler getLinearSampler() const { return linearSampler; }
    
    // Get image views for debug display
    VkImageView getSceneColorImageView(uint32_t frameIndex) const { return sceneColorImageViews[frameIndex]; }
    VkImageView getSceneDepthImageView(uint32_t frameIndex) const { return sceneDepthImageViews[frameIndex]; }
    
    // Update the scene textures binding (color + depth) for refraction and edge foam
    void updateSceneTexturesBinding(VkImageView colorImageView, VkImageView depthImageView, uint32_t frameIndex);

private:
    void createWaterRenderPass();
    void createSceneRenderPass();
    void createWaterPipelines();
    void createPostProcessPipeline();
    void createDescriptorSets();
    void createSamplers();

    VulkanApp* app;
    WaterParams params;

    // Indirect renderer for water geometry
    IndirectRenderer waterIndirectRenderer;

    // Scene offscreen render target (render main scene here before water)
    // Per-frame offscreen render targets for main scene (color + depth) - 2 frames in flight
    std::array<VkImage, 2> sceneColorImages = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkDeviceMemory, 2> sceneColorMemories = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkImageView, 2> sceneColorImageViews = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkImage, 2> sceneDepthImages = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkDeviceMemory, 2> sceneDepthMemories = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkImageView, 2> sceneDepthImageViews = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkFramebuffer, 2> sceneFramebuffers = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkRenderPass sceneRenderPass = VK_NULL_HANDLE;

    // Offscreen render targets for water geometry pass
    VkImage waterDepthImage = VK_NULL_HANDLE;
    VkDeviceMemory waterDepthMemory = VK_NULL_HANDLE;
    VkImageView waterDepthImageView = VK_NULL_HANDLE;

    VkImage waterNormalImage = VK_NULL_HANDLE;
    VkDeviceMemory waterNormalMemory = VK_NULL_HANDLE;
    VkImageView waterNormalImageView = VK_NULL_HANDLE;

    // Water mask (where water exists)
    VkImage waterMaskImage = VK_NULL_HANDLE;
    VkDeviceMemory waterMaskMemory = VK_NULL_HANDLE;
    VkImageView waterMaskImageView = VK_NULL_HANDLE;
    
    // Water geometry pass depth buffer
    VkImage waterGeomDepthImage = VK_NULL_HANDLE;
    VkDeviceMemory waterGeomDepthMemory = VK_NULL_HANDLE;
    VkImageView waterGeomDepthImageView = VK_NULL_HANDLE;

    std::array<VkFramebuffer, 2> waterFramebuffers = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkRenderPass waterRenderPass = VK_NULL_HANDLE;

    // Pipelines
    VkPipeline waterGeometryPipeline = VK_NULL_HANDLE;
    VkPipeline waterPostProcessPipeline = VK_NULL_HANDLE;
    VkPipelineLayout waterPostProcessPipelineLayout = VK_NULL_HANDLE;
    
    // Water geometry pipeline layout (includes depth texture binding)
    VkPipelineLayout waterGeometryPipelineLayout = VK_NULL_HANDLE;

    // Descriptor sets for post-process
    VkDescriptorSetLayout postProcessDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool postProcessDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet postProcessDescriptorSet = VK_NULL_HANDLE;
    
    // Descriptor set for water geometry (scene depth texture)
    VkDescriptorSetLayout waterDepthDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool waterDepthDescriptorPool = VK_NULL_HANDLE;
    // Per-frame descriptor sets for scene textures (2 frames in flight)
    std::array<VkDescriptorSet, 2> waterDepthDescriptorSets = {VK_NULL_HANDLE, VK_NULL_HANDLE};

    // Uniform buffer for water params (post-process)
    Buffer waterUniformBuffer;
    
    // Uniform buffer for water geometry shader params
    Buffer waterParamsBuffer;

    // Samplers
    VkSampler linearSampler = VK_NULL_HANDLE;

    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
};
