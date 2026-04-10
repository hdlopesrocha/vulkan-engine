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

    // Return the cubemap view for reflection sampling
    VkImageView getSolid360View() const { return cube360CubeView; }
    VkImageView getCube360FaceView(uint32_t face) const { return (face < 6) ? cube360FaceViews[face] : VK_NULL_HANDLE; }
    VkImageView getCube360CubeView() const { return cube360CubeView; }

private:
    static constexpr uint32_t CUBE360_FACE_SIZE = 512;

    VkImage cube360ColorImage = VK_NULL_HANDLE;
    VkDeviceMemory cube360ColorMemory = VK_NULL_HANDLE;
    std::array<VkImageView, 6> cube360FaceViews = {};
    VkImageView cube360CubeView = VK_NULL_HANDLE;

    VkImage cube360DepthImage = VK_NULL_HANDLE;
    VkDeviceMemory cube360DepthMemory = VK_NULL_HANDLE;
    VkImageView cube360DepthView = VK_NULL_HANDLE;

    std::array<VkFramebuffer, 6> cube360Framebuffers = {};

    // Equirectangular conversion removed: use cubemap directly for sampling
};
