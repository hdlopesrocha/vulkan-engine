#pragma once

#include "VulkanApp.hpp"
#include "IndirectRenderer.hpp"
#include <glm/glm.hpp>

// Water rendering parameters
struct WaterParams {
    float time = 0.0f;
    float waveSpeed = 0.5f;
    float waveScale = 0.02f;
    float refractionStrength = 0.1f;
    float fresnelPower = 5.0f;
    float transparency = 0.6f;
    glm::vec3 shallowColor = glm::vec3(0.1f, 0.4f, 0.5f);
    glm::vec3 deepColor = glm::vec3(0.0f, 0.1f, 0.2f);
    float depthFalloff = 0.1f;
    int noiseOctaves = 4;
    float noisePersistence = 0.5f;
    float noiseScale = 8.0f;
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
    void beginWaterGeometryPass(VkCommandBuffer cmd);
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
    VkFramebuffer getWaterFramebuffer() const { return waterFramebuffer; }
    VkRenderPass getWaterRenderPass() const { return waterRenderPass; }

    // Get water depth/normal/mask images for post-process sampling
    VkImageView getWaterDepthView() const { return waterDepthImageView; }
    VkImageView getWaterNormalView() const { return waterNormalImageView; }
    VkImageView getWaterMaskView() const { return waterMaskImageView; }
    
    // Get scene offscreen target views (for rendering scene before water)
    VkImageView getSceneColorView() const { return sceneColorImageView; }
    VkImageView getSceneDepthView() const { return sceneDepthImageView; }
    VkFramebuffer getSceneFramebuffer() const { return sceneFramebuffer; }
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
    VkDescriptorSet getWaterDepthDescriptorSet() const { return waterDepthDescriptorSet; }
    
    // Update the scene depth texture binding for edge foam effect
    void updateSceneDepthBinding(VkImageView depthImageView);
    
    // Initialize depth copy resources at the given size
    void initDepthCopyResources(uint32_t width, uint32_t height);
    
    // Copy scene depth buffer before water rendering (call between render passes)
    void copySceneDepth(VkCommandBuffer cmd, VkImage srcDepthImage, uint32_t width, uint32_t height);
    
    // Get the depth copy image view (for binding)
    VkImageView getSceneDepthCopyView() const { return sceneDepthCopyImageView; }

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
    VkImage sceneColorImage = VK_NULL_HANDLE;
    VkDeviceMemory sceneColorMemory = VK_NULL_HANDLE;
    VkImageView sceneColorImageView = VK_NULL_HANDLE;
    
    VkImage sceneDepthImage = VK_NULL_HANDLE;
    VkDeviceMemory sceneDepthMemory = VK_NULL_HANDLE;
    VkImageView sceneDepthImageView = VK_NULL_HANDLE;
    
    VkFramebuffer sceneFramebuffer = VK_NULL_HANDLE;
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

    VkFramebuffer waterFramebuffer = VK_NULL_HANDLE;
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
    VkDescriptorSet waterDepthDescriptorSet = VK_NULL_HANDLE;
    
    // Copy of scene depth for sampling during water rendering
    VkImage sceneDepthCopyImage = VK_NULL_HANDLE;
    VkDeviceMemory sceneDepthCopyMemory = VK_NULL_HANDLE;
    VkImageView sceneDepthCopyImageView = VK_NULL_HANDLE;

    // Uniform buffer for water params
    Buffer waterUniformBuffer;

    // Samplers
    VkSampler linearSampler = VK_NULL_HANDLE;

    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
};
