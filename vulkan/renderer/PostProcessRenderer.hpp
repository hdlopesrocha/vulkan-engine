#pragma once

#include "Renderer.hpp"
#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include <glm/glm.hpp>
#include <array>
#include "CommandBufferState.hpp"

// Forward-declare shared water types (defined in WaterRenderer.hpp)
struct WaterParams;
struct WaterUBO;

class PostProcessRenderer : public Renderer {
public:
    PostProcessRenderer();
    ~PostProcessRenderer();

    void init(VulkanApp* app);
    void cleanup(VulkanApp* app) override;

    /// Composite scene + water + brush into the swapchain framebuffer.
    /// Brush color/depth views come from the early brush pass offscreen targets.
    /// waterGeomDepthView is the raw water geometry depth buffer (D32).
    /// brushAlpha controls the brush overlay opacity (0.0 = invisible, 1.0 = fully opaque).
    /// brushMode: 0=overlay, 2=PAINT (replace solid texture within brush volume).
    void render(VulkanApp* app, VkCommandBuffer cmd,
                VkImageView sceneColorView, VkImageView sceneDepthView,
                VkImageView waterColorView,
                VkImageView brushColorView, VkImageView brushDepthView,
                VkImageView brushBackFaceDepthView,
                VkImageView waterGeomDepthView,
                VkImageView vegColorView, VkImageView vegDepthView,
                VkImageView sdfColorView, VkImageView sdfDepthView,
                VkImageView bboxColorView, VkImageView bboxDepthView,
                float brushAlpha, float brushMode,
                const glm::mat4& viewProj, const glm::mat4& invViewProj,
                const glm::vec3& viewPos,
                uint32_t frameIdx,
                VkImageView skyView = VK_NULL_HANDLE);

    bool isReady() const { return pipeline != VK_NULL_HANDLE; }

    VkSampler getLinearSampler() const { return linearSampler; }

    void setRenderSize(uint32_t width, uint32_t height);

private:
    void createSampler(VulkanApp* app);
    void createPipeline(VulkanApp* app);
    void createDescriptorSets(VulkanApp* app);
    // Phase 2 (VK_EXT_descriptor_buffer): allocate 3 host-visible descriptor
    // buffers (one per frame slot). No-op when !app->useDescriptorBuffer().
    void createDescriptorBuffers(VulkanApp* app);
    void destroyDescriptorBuffers(VulkanApp* app);
    // Write one frame slot's descriptor-buffer memory (bindings 0-14).
    // Returns false when the DB path cannot be used (caller falls back).
    bool writeSlotToDescriptorBuffer(VulkanApp* app, uint32_t slot,
                                     const std::array<VkDescriptorImageInfo, 15>& imageInfos,
                                     const VkDescriptorImageInfo& skyImageInfo,
                                     const VkDescriptorBufferInfo& bufferInfo);

    TrackedHandle<VkPipeline> pipeline;
    TrackedHandle<VkPipelineLayout> pipelineLayout;
    TrackedHandle<VkDescriptorSetLayout> descriptorSetLayout;
    TrackedHandle<VkDescriptorPool> descriptorPool;
    static constexpr uint32_t FRAMES_IN_FLIGHT = VulkanApp::MAX_FRAMES_IN_FLIGHT;
    std::array<TrackedHandle<VkDescriptorSet>, FRAMES_IN_FLIGHT> descriptorSets;

    // Descriptor-buffer state (live only when useDescriptorBuffer()).
    // Layout = descriptorSetLayout (15 bindings: 14 images + 1 UBO).
    std::array<Buffer, FRAMES_IN_FLIGHT> descBuffers_{};
    std::array<VkDeviceAddress, FRAMES_IN_FLIGHT> descAddresses_{};
    VkDeviceSize descSetSize_ = 0;
    std::array<VkDeviceSize, 15> descBindingOffsets_{};
    bool descReady_ = false;

    // Per-frame-slot cache of the last descriptor contents written by render().
    // The offscreen target views bound here are stable per frame slot, so the
    // descriptor writes are skipped while every input (sampler/view/layout per
    // binding + UBO) is unchanged — steady state issues 0 descriptor updates
    // (per-frame UBO contents stream via mapped memcpy into the already-bound
    // buffer, overlapped with compute on the GPU timeline).
    // `valid` starts false, guaranteeing the first frame always writes.
    // Classic path: cache miss triggers vkUpdateDescriptorSets.
    // Descriptor-buffer path (layout created with
    // VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT, bound via
    // vkCmdBindDescriptorBuffersEXT): cache miss triggers direct
    // DescriptorBufferHelper host writes (plain memcpys, no driver validation).
    struct FrameDescriptorSignature {
        std::array<VkSampler, 15> samplers{};
        std::array<VkImageView, 15> views{};
        std::array<VkImageLayout, 15> layouts{};
        VkBuffer uboBuffer = VK_NULL_HANDLE;
        VkDeviceSize uboOffset = 0;
        VkDeviceSize uboRange = 0;
        bool valid = false; // true once this slot has been written at least once
        bool matches(const FrameDescriptorSignature& o) const {
            return samplers == o.samplers && views == o.views && layouts == o.layouts &&
                   uboBuffer == o.uboBuffer && uboOffset == o.uboOffset && uboRange == o.uboRange;
        }
    };
    std::array<FrameDescriptorSignature, FRAMES_IN_FLIGHT> descriptorWriteCache;

    Buffer uniformBuffer;
    TrackedHandle<VkSampler> linearSampler;

    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
};