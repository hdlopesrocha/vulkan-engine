#pragma once

#include "VulkanApp.hpp"
#include "VertexBufferObject.hpp"
#include "SkySphere.hpp"
#include "VertexBufferObjectBuilder.hpp"
#include "../math/SphereModel.hpp"
#include "ShaderStage.hpp"
#include "../Uniforms.hpp"
#include "../widgets/SkySettings.hpp"

class SkyRenderer {
public:
    explicit SkyRenderer(VulkanApp* app);
    ~SkyRenderer();

    // Create sky pipelines; renderPassOverride lets callers supply a compatible render pass
    void init(VkRenderPass renderPassOverride = VK_NULL_HANDLE);

    // Render sky sphere using internal VBO/descriptor/uniform
    void render(VkCommandBuffer &cmd, VkDescriptorSet descriptorSet, Buffer &uniformBuffer, const UniformObject &ubo, const glm::mat4 &viewProjection, SkySettings::Mode skyMode);

    // Initialize the sky sphere and internal VBO (optional)
    void initSky(SkySettings& skySettings, VkDescriptorSet descriptorSet);

    // Update sky internals (e.g. SkySphere animation)
    void update();

    void cleanup();

private:
    VulkanApp* app = nullptr;
    VkPipeline skyPipeline = VK_NULL_HANDLE;        // Gradient sky pipeline
    VkPipelineLayout skyPipelineLayout = VK_NULL_HANDLE;
    VkPipeline skyGridPipeline = VK_NULL_HANDLE;   // Grid sky pipeline
    VkPipelineLayout skyGridPipelineLayout = VK_NULL_HANDLE;
    VkShaderModule skyVertModule = VK_NULL_HANDLE;
    VkShaderModule skyFragModule = VK_NULL_HANDLE;
    VkShaderModule skyGridFragModule = VK_NULL_HANDLE;
    // Owned sky sphere and VBO
    std::unique_ptr<SkySphere> skySphere;
    VertexBufferObject skyVBO;
};
