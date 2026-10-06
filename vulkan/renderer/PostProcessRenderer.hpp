#pragma once

#include "Renderer.hpp"
#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include <glm/glm.hpp>
#include <array>
#include "CommandBufferState.hpp"

// Forward-declare shared water types (defined in WaterRenderer.hpp)
struct WaterSettings;
struct WaterUBO;

class PostProcessRenderer : public Renderer {
public:
    PostProcessRenderer();
    ~PostProcessRenderer();

    void init(VulkanApp* app);
    void cleanup(VulkanApp* app) override;

    /// Composite scene + water + brush into the swapchain framebuffer.
    /// Brush color/depth views come from the early brush pass offscreen targets.
    /// waterBodyView is the water refraction+tint body (RGB) with the body
    /// weight in A; waterColumnView packs the measured water depth (m, R) and
    /// the per-material blur radius in pixels (G). Together they drive the
    /// depth-guided water blur performed in this final pass, which blurs only
    /// the body so reflections stay sharp.
    /// waterGeomDepthView is the raw water geometry depth buffer (D32).
    /// brushAlpha controls the brush overlay opacity (0.0 = invisible, 1.0 = fully opaque).
    /// brushMode: 0=overlay, 2=PAINT (replace solid texture within brush volume).
    /// waterBlurEnabled: pass WaterRenderer::waterBlurNeeded(). When false (no
    /// layer has enableBlur && blurRadius > 0, or water-in-main is active) the
    /// composite performs no waterBody/waterColumn fetches at all; the water
    /// pass wrote only the color target (H4, perf report 19).
    /// vegetationScaled: pass (Settings::vegetationRenderScale < 1). When true
    /// the composite resolves the vegetation depth with the closest of the
    /// 2x2 taps instead of one bilinear sample (perf report 22 M12).
    void render(VulkanApp* app, VkCommandBuffer cmd,
                VkImageView sceneColorView, VkImageView sceneDepthView,
                VkImageView waterColorView,
                VkImageView waterBodyView, VkImageView waterColumnView,
                VkImageView brushColorView, VkImageView brushDepthView,
                VkImageView brushBackFaceDepthView,
                VkImageView waterGeomDepthView,
                VkImageView vegColorView, VkImageView vegDepthView,
                VkImageView sdfColorView, VkImageView sdfDepthView,
                VkImageView bboxColorView, VkImageView bboxDepthView,
                VkImageView fireColorView, VkImageView fireDepthView,
                float brushAlpha, float brushMode,
                const glm::mat4& viewProj, const glm::mat4& invViewProj,
                const glm::vec3& viewPos,
                uint32_t frameIdx,
                VkImageView skyView = VK_NULL_HANDLE,
                bool waterBlurEnabled = true,
                bool vegetationScaled = false,
                VkBuffer windFieldBuffer = VK_NULL_HANDLE,
                float windTime = 0.0f);

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
    // Write one frame slot's descriptor-buffer memory (bindings 0-18 + 27).
    // Returns false when the DB path cannot be used (caller falls back).
    bool writeSlotToDescriptorBuffer(VulkanApp* app, uint32_t slot,
                                     const std::array<VkDescriptorImageInfo, 19>& imageInfos,
                                     const VkDescriptorImageInfo& skyImageInfo,
                                     const VkDescriptorBufferInfo& bufferInfo,
                                     const VkDescriptorBufferInfo& windBufferInfo);

    TrackedHandle<VkPipeline> pipeline;
    TrackedHandle<VkPipelineLayout> pipelineLayout;
    TrackedHandle<VkDescriptorSetLayout> descriptorSetLayout;
    TrackedHandle<VkDescriptorPool> descriptorPool;
    static constexpr uint32_t FRAMES_IN_FLIGHT = VulkanApp::MAX_FRAMES_IN_FLIGHT;
    std::array<TrackedHandle<VkDescriptorSet>, FRAMES_IN_FLIGHT> descriptorSets;

    // Descriptor-buffer state (live only when useDescriptorBuffer()).
    // Layout = descriptorSetLayout (19 bindings: 18 images + 1 UBO).
    std::array<Buffer, FRAMES_IN_FLIGHT> descBuffers_{};
    std::array<VkDeviceAddress, FRAMES_IN_FLIGHT> descAddresses_{};
    VkDeviceSize descSetSize_ = 0;
    std::array<VkDeviceSize, 28> descBindingOffsets_{};
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
        std::array<VkSampler, 19> samplers{};
        std::array<VkImageView, 19> views{};
        std::array<VkImageLayout, 19> layouts{};
        VkBuffer uboBuffer = VK_NULL_HANDLE;
        VkDeviceSize uboOffset = 0;
        VkDeviceSize uboRange = 0;
        VkBuffer windBuffer = VK_NULL_HANDLE;
        bool valid = false; // true once this slot has been written at least once
        bool matches(const FrameDescriptorSignature& o) const {
            return samplers == o.samplers && views == o.views && layouts == o.layouts &&
                   uboBuffer == o.uboBuffer && uboOffset == o.uboOffset && uboRange == o.uboRange &&
                   windBuffer == o.windBuffer;
        }
    };
    std::array<FrameDescriptorSignature, FRAMES_IN_FLIGHT> descriptorWriteCache;

    Buffer uniformBuffer;
    TrackedHandle<VkSampler> linearSampler;

    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
};