#include "SolidRenderer.hpp"
#include "VegetationRenderer.hpp"
#include "WaterRenderer.hpp"
#include "SkyRenderer.hpp"
#include "ShadowRenderer.hpp"
#include "DebugCubeRenderer.hpp"
#pragma once

#include <vulkan/vulkan.h>
#include "VulkanApp.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <cmath>
#include "TextureArrayManager.hpp"
#include "TextureTriple.hpp"
#include "MaterialManager.hpp"
#include "ShaderStage.hpp"
#include "../utils/FileReader.hpp"
#include "../math/Vertex.hpp"
#include <unordered_map>
#include <memory>
#include "Model3DVersion.hpp"
#include "SkyRenderer.hpp"
#include "SolidRenderer.hpp"
#include "IndirectRenderer.hpp"
#include "ShadowRenderer.hpp"
#include "WaterRenderer.hpp"

#include "PassUBO.hpp"

class SceneRenderer {
public:
    // UBOs for main, shadow, and water passes
    PassUBO<UniformObject> mainPassUBO;
    PassUBO<UniformObject> shadowPassUBO;
    PassUBO<WaterParamsGPU> waterPassUBO;

    VulkanApp* app = nullptr;

    // Texture array manager for albedo/normal/bump arrays
    TextureArrayManager textureArrayManager;
    
    // Main uniform buffer
    Buffer mainUniformBuffer;
    
    // Materials SSBO
    Buffer materialsBuffer;

    std::unique_ptr<SkyRenderer> skyRenderer;
    std::unique_ptr<ShadowRenderer> shadowMapper;
    std::unique_ptr<WaterRenderer> waterRenderer;
    std::unique_ptr<SolidRenderer> solidRenderer;
    std::unique_ptr<VegetationRenderer> vegetationRenderer;
    std::unique_ptr<DebugCubeRenderer> debugCubeRenderer;
    // Sky settings owned by this renderer
    SkySettings skySettings;
    SkySettings& getSkySettings() { return skySettings; }

    SceneRenderer(VulkanApp* app_);
    ~SceneRenderer();

    void createPipelines();
    void createDescriptorSets(MaterialManager &materialManager, TextureArrayManager &textureArrayManager, VkDescriptorSet &outDescriptorSet, VkDescriptorSet &outShadowPassDescriptorSet, size_t tripleCount);
    void shadowPass(VkCommandBuffer &commandBuffer, VkQueryPool queryPool, VkDescriptorSet shadowPassDescriptorSet, const UniformObject &uboStatic, bool shadowsEnabled, bool shadowTessellationEnabled);
    void depthPrePass(VkCommandBuffer &commandBuffer, VkQueryPool queryPool);
    void skyPass(VkCommandBuffer &commandBuffer, VkDescriptorSet perTextureDescriptorSet, Buffer &mainUniformBuffer, const UniformObject &uboStatic, const glm::mat4 &viewProj);
    void mainPass(VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &mainPassInfo, uint32_t frameIdx, bool hasWater, bool vegetationEnabled, VkDescriptorSet perTextureDescriptorSet, Buffer &mainUniformBuffer, bool wireframeEnabled, bool profilingEnabled, VkQueryPool queryPool, const glm::mat4 &viewProj,
                  const UniformObject &uboStatic, const WaterParams &waterParams, float waterTime, bool normalMappingEnabled, bool tessellationEnabled, bool shadowsEnabled, int debugMode, float triplanarThreshold, float triplanarExponent);
    void waterPass(VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &renderPassInfo, uint32_t frameIdx, VkDescriptorSet perTextureDescriptorSet, bool profilingEnabled, VkQueryPool queryPool, const WaterParams &waterParams, float waterTime);
    void init(VulkanApp* app_, VkDescriptorSet descriptorSet = VK_NULL_HANDLE);
    void cleanup();

    // Resize offscreen resources when the swapchain changes
    void onSwapchainResized(uint32_t width, uint32_t height);
};

