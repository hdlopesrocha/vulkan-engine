#pragma once

#include "VulkanApp.hpp"
#include "VertexBufferObject.hpp"
#include "ShaderStage.hpp"
#include "../Uniforms.hpp"

class SkyRenderer {
public:
    explicit SkyRenderer(VulkanApp* app);
    ~SkyRenderer();

    // Create sky pipeline (uses app helper to build pipeline)
    void init();

    // Render sky sphere using provided VBO/descriptor/uniform
    void render(VkCommandBuffer &cmd, const VertexBufferObject &vbo, VkDescriptorSet descriptorSet, Buffer &uniformBuffer, const UniformObject &uboStatic, const glm::mat4 &projMat, const glm::mat4 &viewMat);

    void cleanup();

private:
    VulkanApp* app = nullptr;
    VkPipeline skyPipeline = VK_NULL_HANDLE;
};
