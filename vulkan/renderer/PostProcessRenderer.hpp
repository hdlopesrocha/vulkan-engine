#pragma once

#include "../VulkanApp.hpp"
#include <glm/glm.hpp>
#include <array>

// Forward-declare shared water types (defined in WaterRenderer.hpp)
struct WaterParams;
struct WaterUBO;

class PostProcessRenderer {
public:
    PostProcessRenderer();
    ~PostProcessRenderer();

    void init(VulkanApp* app);
    void cleanup(VulkanApp* app);

    /// Composite scene + water into the swapchain framebuffer.
    /// Water depth/normal/mask views come from WaterRenderer's offscreen targets.
    void render(VulkanApp* app, VkCommandBuffer cmd,
                VkFramebuffer swapchainFramebuffer,
                VkRenderPass swapchainRenderPass,
                VkImageView sceneColorView, VkImageView sceneDepthView,
                VkImageView waterDepthView,
                const glm::mat4& viewProj, const glm::mat4& invViewProj,
                const glm::vec3& viewPos,
                uint32_t frameIdx,
                bool beginRenderPass = true,
                VkImageView skyView = VK_NULL_HANDLE);

    bool isReady() const { return pipeline != VK_NULL_HANDLE; }

    VkSampler getLinearSampler() const { return linearSampler; }

    void setRenderSize(uint32_t width, uint32_t height);

private:
    void createSampler(VulkanApp* app);
    void createPipeline(VulkanApp* app);
    void createDescriptorSets(VulkanApp* app);

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    static constexpr uint32_t FRAMES_IN_FLIGHT = 2;
    std::array<VkDescriptorSet, FRAMES_IN_FLIGHT> descriptorSets = {VK_NULL_HANDLE, VK_NULL_HANDLE};

    Buffer uniformBuffer;
    VkSampler linearSampler = VK_NULL_HANDLE;

    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
};
