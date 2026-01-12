#pragma once

#include "VulkanApp.hpp"
#include "VertexBufferObject.hpp"
#include "ShaderStage.hpp"
#include "../Uniforms.hpp"
#include "../widgets/SkyWidget.hpp"

class SkyRenderer {
public:
    explicit SkyRenderer(VulkanApp* app);
    ~SkyRenderer();

    // Create sky pipeline (uses app helper to build pipeline)
    void init();

    // Render sky sphere using provided VBO/descriptor/uniform
    void render(VkCommandBuffer &cmd, const VertexBufferObject &vbo, VkDescriptorSet descriptorSet, Buffer &uniformBuffer, const UniformObject &uboStatic, const glm::mat4 &viewProjection, SkyMode skyMode);

    void cleanup();

private:
    VulkanApp* app = nullptr;
    VkPipeline skyPipeline = VK_NULL_HANDLE;        // Gradient sky pipeline
    VkPipeline skyGridPipeline = VK_NULL_HANDLE;   // Grid sky pipeline
    VkShaderModule skyVertModule = VK_NULL_HANDLE;
    VkShaderModule skyFragModule = VK_NULL_HANDLE;
    VkShaderModule skyGridFragModule = VK_NULL_HANDLE;
};
