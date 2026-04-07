#pragma once

#include "VulkanApp.hpp"
#include "IndirectRenderer.hpp"
#include "SkyRenderer.hpp"
#include "SolidRenderer.hpp"
#include <glm/glm.hpp>
#include "../utils/Scene.hpp"
#include "../Uniforms.hpp"
#include "../widgets/SkySettings.hpp"
#include <unordered_map>
#include "Model3DVersion.hpp"

// Water rendering parameters (CPU-side)
struct WaterParams {
    float time = 0.0f;
    float waveSpeed = 0.5f;
    float waveScale = 0.03f;
    float refractionStrength = 0.03f;
    float fresnelPower = 5.0f;
    float transparency = 0.7f;
    glm::vec3 shallowColor = glm::vec3(0.1f, 0.4f, 0.5f);
    glm::vec3 deepColor = glm::vec3(0.0f, 0.15f, 0.25f);
    float depthFalloff = 0.1f;
    int noiseOctaves = 4;
    float noisePersistence = 0.5f;
    float noiseScale = 0.4f;
    float waterTint = 0.3f;
    float noiseTimeSpeed = 1.0f;

    // Reflection / specular controls
    float reflectionStrength = 0.6f;  // How much reflection mixes into the surface [0..1]
    float specularIntensity = 2.0f;   // Brightness of specular highlight
    float specularPower = 128.0f;     // Sharpness of specular highlight
    float glitterIntensity = 1.5f;    // Brightness of sun glitter sparkles

    // Vertical bump amplitude for water geometry
    float bumpAmplitude = 8.0f;

    // Depth-based wave attenuation: distance (world units) over which waves
    // transition from zero displacement (at solid surface) to full amplitude.
    // 0 = disabled (no depth-based attenuation).
    float waveDepthTransition = 20.0f;

    // Feature toggles
    bool enableReflection = true;
    bool enableRefraction = true;
    bool enableBlur = true;
    // If true, apply `reflectionStrength` uniformly across the surface
    // instead of modulating by Fresnel. Useful for debugging or stylized looks.
    bool uniformReflection = false;

    // PCF-style scene-color blur
    float blurRadius = 8.0f;    // texel radius of blur kernel
    int   blurSamples = 4;      // number of blur taps per axis (NxN kernel)

    // Volume depth-based effect transitions
    float volumeBlurRate = 0.004f;   // exponential rate: blur ramps with water thickness
    float volumeBumpRate = 0.05f;  // exponential rate: bump ramps with water thickness
};

// GPU-side water params UBO (matches shader WaterParamsUBO layout)
struct WaterParamsGPU {
    glm::vec4 params1;  // x=refractionStrength, y=fresnelPower, z=transparency, w=reflectionStrength
    glm::vec4 params2;  // x=waterTint, y=noiseScale, z=noiseOctaves, w=noisePersistence
    glm::vec4 params3;  // x=noiseTimeSpeed, y=waterTime, z=specularIntensity, w=specularPower
    glm::vec4 shallowColor; // xyz = shallowColor, w = waveDepthTransition
    glm::vec4 deepColor; // xyz = deepColor, w = glitterIntensity
    glm::vec4 waveParams; // x=unused, y=unused, z=bumpAmplitude, w=depthFalloff
    glm::vec4 reserved1;  // x=enableReflection, y=enableRefraction, z=enableBlur, w=blurRadius
    glm::vec4 reserved2;  // x=blurSamples, y=volumeBlurRate, z=volumeBumpRate, w=unused
    glm::vec4 reserved3;  // x=cube360Available(0/1), y=unused, z=unused, w=unused
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
    WaterRenderer();
    ~WaterRenderer();

    void init(VulkanApp* app, Buffer& waterParamsBuffer);
    void cleanup(VulkanApp* app);

    // Create offscreen render targets for water rendering
    void createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height);
    void destroyRenderTargets(VulkanApp* app);

    // Get the indirect renderer for water meshes
    IndirectRenderer& getIndirectRenderer() { return waterIndirectRenderer; }

    // Begin water geometry pass (renders water depth/normals to offscreen target)
    void beginWaterGeometryPass(VkCommandBuffer cmd, uint32_t frameIndex);
    void endWaterGeometryPass(VkCommandBuffer cmd);

    // Back-face depth pre-pass (reversed winding for water volume thickness)
    void renderBackFacePass(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex);
    VkImageView getBackFaceDepthView() const { return backFaceDepthImageView; }
    VkImage getBackFaceDepthImage() const { return backFaceDepthImage; }

    // Execute the water offscreen geometry pass on the provided command buffer.
    // The solid render pass must have already ended on this same command buffer.
    void render(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                VkImageView sceneColorView, VkImageView sceneDepthView,
                const WaterParams& params, float waterTime,
                VkImageView skyView = VK_NULL_HANDLE);

    // Get water depth/normal/mask images for post-process sampling
    VkImageView getWaterDepthView() const { return waterDepthImageView; }
    // View that swizzles alpha into RGB so linear depth can be displayed easily
    VkImageView getWaterDepthAlphaView() const { return waterDepthAlphaImageView; }
    VkImageView getWaterNormalView() const { return waterNormalImageView; }
    VkImageView getWaterMaskView() const { return waterMaskImageView; }
    
    // Get scene offscreen target views (for rendering scene before water)
    VkImage getSceneColorImage(uint32_t frameIndex) const { return sceneColorImages[frameIndex]; }
    VkImageView getSceneColorView(uint32_t frameIndex) const { return sceneColorImageViews[frameIndex]; }
    VkImage getSceneDepthImage(uint32_t frameIndex) const { return sceneDepthImages[frameIndex]; }
    VkImageView getSceneDepthView(uint32_t frameIndex) const { return sceneDepthImageViews[frameIndex]; }
    VkFramebuffer getSceneFramebuffer(uint32_t frameIndex) const { return sceneFramebuffers[frameIndex]; }
    VkRenderPass getSceneRenderPass() const { return sceneRenderPass; }
        void beginScenePass(VkCommandBuffer cmd, uint32_t frameIndex, VkClearValue colorClear, VkClearValue depthClear);
        void endScenePass(VkCommandBuffer cmd);

    // Update parameters
    void setParams(const WaterParams& params) { this->params = params; }
    WaterParams& getParams() { return params; }

    // Time management for water animation
    void advanceTime(float dt) { params.time += dt; }
    float getTime() const { return params.time; }
    
    // Get the water geometry pipeline (for rendering water to G-buffer)
    VkPipeline getWaterGeometryPipeline() const { return waterGeometryPipeline; }
    
    // Get the water geometry pipeline layout
    VkPipelineLayout getWaterGeometryPipelineLayout() const { return waterGeometryPipelineLayout; }

    // Get the water geometry render pass (for creating compatible wireframe pipelines)
    VkRenderPass getWaterRenderPass() const { return waterRenderPass; }

    // Get the descriptor set layout for scene textures (set 2)
    VkDescriptorSetLayout getWaterDepthDescriptorSetLayout() const { return waterDepthDescriptorSetLayout; }

    // Prepare render state (UBO upload, descriptor update, pre-barrier).
    // Call this before beginWaterGeometryPass when manually recording commands.
    void prepareRender(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                       VkImageView sceneColorView, VkImageView sceneDepthView,
                       const WaterParams& params, float waterTime,
                       VkImageView skyView = VK_NULL_HANDLE);

    // Emit post-geometry-pass barrier for fragment shader sampling.
    void postRenderBarrier(VkCommandBuffer cmd);
    
    // Get water depth descriptor set (for binding scene depth texture)
    VkDescriptorSet getWaterDepthDescriptorSet(uint32_t frameIndex) const { return waterDepthDescriptorSets[frameIndex]; }
    
    // Get water params buffer
    Buffer& getWaterParamsBuffer() { return waterParamsBuffer; }
    
    // Get sampler for ImGui texture display
    VkSampler getLinearSampler() const { return linearSampler; }
    
    // Get image views for debug display
    VkImageView getSceneColorImageView(uint32_t frameIndex) const { return sceneColorImageViews[frameIndex]; }
    VkImageView getSceneDepthImageView(uint32_t frameIndex) const { return sceneDepthImageViews[frameIndex]; }
    
    // Update the scene textures binding (color + depth + sky) for refraction and edge foam
    void updateSceneTexturesBinding(VulkanApp* app, VkImageView colorImageView, VkImageView depthImageView, uint32_t frameIndex, VkImageView skyImageView = VK_NULL_HANDLE);

    // --- Solid 360° cubemap reflection ---
    // Create offscreen cubemap + equirect conversion resources.
    // solidRenderPass must be a render pass compatible with the solid and sky pipelines.
    void createSolid360Targets(VulkanApp* app, VkRenderPass solidRenderPass);
    void destroySolid360Targets(VulkanApp* app);

    // Render the solid scene (sky + terrain) into a cubemap from the camera position,
    // then convert to equirectangular. The result can be sampled via getSolid360View().
    // Must be called outside any active render pass.
    void renderSolid360(VulkanApp* app, VkCommandBuffer cmd,
                        VkRenderPass solidRenderPass,
                        SkyRenderer* skyRenderer, SkySettings::Mode skyMode,
                        SolidRenderer* solidRenderer,
                        VkDescriptorSet mainDescriptorSet,
                        Buffer& uniformBuffer, const UniformObject& ubo);

    // Access the 360° solid equirectangular view for water reflection sampling
    VkImageView getSolid360View() const { return equirect360View; }
    // Access the cubemap view for the 360° solid reflection (cube)
    VkImageView getCube360View() const { return cube360CubeView; }
    // Access per-face 2D image views for debugging (order: +X, -X, +Y, -Y, +Z, -Z)
    VkImageView getCube360FaceView(uint32_t face) const { return (face < 6) ? cube360FaceViews[face] : VK_NULL_HANDLE; }

    // Register model version for water meshes (stored here)
    void registerModelVersion(NodeID id, const Model3DVersion& ver) { waterNodeModelVersions[id] = ver; }

    // Remove all registered water meshes from the indirect renderer and clear the map
    void removeAllRegisteredMeshes() {
        for (auto &entry : waterNodeModelVersions) {
            if (entry.second.meshId != UINT32_MAX) waterIndirectRenderer.removeMesh(entry.second.meshId);
        }
        waterNodeModelVersions.clear();
    }

private:
    void createWaterRenderPass(VulkanApp* app);
    void createSceneRenderPass(VulkanApp* app);
    void createWaterPipelines(VulkanApp* app);
    void createSamplers(VulkanApp* app);

    
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
    // Alternate view that swizzles alpha (linear depth) into RGB for debug display
    VkImageView waterDepthAlphaImageView = VK_NULL_HANDLE;

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

    // Back-face depth pass (reversed winding for water volume thickness)
    VkRenderPass backFaceRenderPass = VK_NULL_HANDLE;
    VkPipeline backFacePipeline = VK_NULL_HANDLE;
    VkImage backFaceDepthImage = VK_NULL_HANDLE;
    VkDeviceMemory backFaceDepthMemory = VK_NULL_HANDLE;
    VkImageView backFaceDepthImageView = VK_NULL_HANDLE;
    std::array<VkFramebuffer, 2> backFaceFramebuffers = {VK_NULL_HANDLE, VK_NULL_HANDLE};

    // Pipelines
    VkPipeline waterGeometryPipeline = VK_NULL_HANDLE;
    
    // Water geometry pipeline layout (includes depth texture binding)
    VkPipelineLayout waterGeometryPipelineLayout = VK_NULL_HANDLE;

    // Descriptor set for water geometry (scene depth texture)
    VkDescriptorSetLayout waterDepthDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool waterDepthDescriptorPool = VK_NULL_HANDLE;
    // Per-frame descriptor sets for scene textures (2 frames in flight)
    std::array<VkDescriptorSet, 2> waterDepthDescriptorSets = {VK_NULL_HANDLE, VK_NULL_HANDLE};

    // Uniform buffer for water geometry shader params
    Buffer waterParamsBuffer;

    // Samplers
    VkSampler linearSampler = VK_NULL_HANDLE;

    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;

    // Map of node -> model version for water geometry managed here
    std::unordered_map<NodeID, Model3DVersion> waterNodeModelVersions;

    // --- Solid 360° cubemap reflection resources ---
    static constexpr uint32_t CUBE360_FACE_SIZE = 512;
    static constexpr uint32_t EQUIRECT360_WIDTH = 1024;
    static constexpr uint32_t EQUIRECT360_HEIGHT = 512;

    // Cubemap color image (6 layers, CUBE_COMPATIBLE)
    VkImage cube360ColorImage = VK_NULL_HANDLE;
    VkDeviceMemory cube360ColorMemory = VK_NULL_HANDLE;
    std::array<VkImageView, 6> cube360FaceViews = {};   // per-face 2D views for FBO
    VkImageView cube360CubeView = VK_NULL_HANDLE;       // cubemap view for sampling

    // Shared depth image for cubemap face rendering
    VkImage cube360DepthImage = VK_NULL_HANDLE;
    VkDeviceMemory cube360DepthMemory = VK_NULL_HANDLE;
    VkImageView cube360DepthView = VK_NULL_HANDLE;

    // Per-face framebuffers (reuse solidRenderPass)
    std::array<VkFramebuffer, 6> cube360Framebuffers = {};

    // Equirectangular output (2D texture, same format as swapchain)
    VkImage equirect360Image = VK_NULL_HANDLE;
    VkDeviceMemory equirect360Memory = VK_NULL_HANDLE;
    VkImageView equirect360View = VK_NULL_HANDLE;
    VkFramebuffer equirect360Framebuffer = VK_NULL_HANDLE;

    // Cubemap→equirect conversion pipeline
    VkRenderPass equirect360RenderPass = VK_NULL_HANDLE;
    VkPipeline equirect360Pipeline = VK_NULL_HANDLE;
    VkPipelineLayout equirect360PipelineLayout = VK_NULL_HANDLE;
    VkShaderModule equirect360VertModule = VK_NULL_HANDLE;
    VkShaderModule equirect360FragModule = VK_NULL_HANDLE;

    // Descriptor set for cubemap sampling in the conversion pass
    VkDescriptorSetLayout cube360DescSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool cube360DescPool = VK_NULL_HANDLE;
    VkDescriptorSet cube360DescSet = VK_NULL_HANDLE;
};
