#pragma once
#include "VulkanApp.hpp"
#include "SkyRenderer.hpp"
#include "SolidRenderer.hpp"
#include "../Uniforms.hpp"
#include <array>

class Solid360Renderer {
public:
    Solid360Renderer();
    ~Solid360Renderer();
    void init(VulkanApp* app);
    void cleanup(VulkanApp* app);

    void createSolid360Targets(VulkanApp* app, VkRenderPass solidRenderPass, VkSampler linearSampler);
    void destroySolid360Targets(VulkanApp* app);

    void renderSolid360(VulkanApp* app, VkCommandBuffer cmd,
                        VkRenderPass solidRenderPass,
                        SkyRenderer* skyRenderer, SkySettings::Mode skyMode,
                        SolidRenderer* solidRenderer,
                        VkDescriptorSet mainDescriptorSet,
                        Buffer& uniformBuffer, const UniformObject& ubo);

    VkImageView getSolid360View() const { return equirect360View; }
    VkImageView getCube360FaceView(uint32_t face) const { return (face < 6) ? cube360FaceViews[face] : VK_NULL_HANDLE; }
    VkImageView getCube360CubeView() const { return cube360CubeView; }

private:
    static constexpr uint32_t CUBE360_FACE_SIZE = 512;
    static constexpr uint32_t EQUIRECT360_WIDTH = 1024;
    static constexpr uint32_t EQUIRECT360_HEIGHT = 512;

    VkImage cube360ColorImage = VK_NULL_HANDLE;
    VkDeviceMemory cube360ColorMemory = VK_NULL_HANDLE;
    std::array<VkImageView, 6> cube360FaceViews = {};
    VkImageView cube360CubeView = VK_NULL_HANDLE;

    VkImage cube360DepthImage = VK_NULL_HANDLE;
    VkDeviceMemory cube360DepthMemory = VK_NULL_HANDLE;
    VkImageView cube360DepthView = VK_NULL_HANDLE;

    std::array<VkFramebuffer, 6> cube360Framebuffers = {};

    VkImage equirect360Image = VK_NULL_HANDLE;
    VkDeviceMemory equirect360Memory = VK_NULL_HANDLE;
    VkImageView equirect360View = VK_NULL_HANDLE;
    VkFramebuffer equirect360Framebuffer = VK_NULL_HANDLE;

    VkRenderPass equirect360RenderPass = VK_NULL_HANDLE;
    VkPipeline equirect360Pipeline = VK_NULL_HANDLE;
    VkPipelineLayout equirect360PipelineLayout = VK_NULL_HANDLE;
    VkShaderModule equirect360VertModule = VK_NULL_HANDLE;
    VkShaderModule equirect360FragModule = VK_NULL_HANDLE;

    VkDescriptorSetLayout cube360DescSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool cube360DescPool = VK_NULL_HANDLE;
    VkDescriptorSet cube360DescSet = VK_NULL_HANDLE;
};
