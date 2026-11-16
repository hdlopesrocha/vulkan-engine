#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <array>
#include <cstdint>

#include "../Renderer.hpp"
#include "../../core/VulkanApp.hpp"
#include "../../resources/Buffer.hpp"
#include "../../ubo/WindRaymarchUBO.hpp"
#include "../../../types/WindRaymarchSettings.hpp"

// WindRaymarchRenderer: renders a given wind velocity as a shape.
//
// Fullscreen raymarch of the SHARED 3D fBm wind velocity field (set=0
// binding 27): the rendered shape is the isosurface where |wind| equals the
// adjustable target speed. The march is clipped by the solid scene depth, so
// foreground geometry occludes the volume.
//
// Recorded inside the final swapchain rendering scope in MyApp::draw(), after
// the post-process composite (alpha-blended overlay, no depth write). One
// per-frame parameter UBO + descriptor set (the scene-depth view changes with
// the frame slot / swapchain resize; the descriptor is rewritten only when the
// bound view changes).
class WindRaymarchRenderer : public Renderer {
public:
    WindRaymarchRenderer() = default;
    ~WindRaymarchRenderer() override { cleanup(nullptr); }

    void init(VulkanApp* app);

    // Streams the parameters UBO, refreshes the per-slot descriptors and
    // records the host->fragment visibility barrier. Call from preRenderPass
    // (OUTSIDE any rendering scope: vkCmdPipelineBarrier2 with
    // HOST srcStage is illegal inside dynamic rendering).
    void prepare(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIdx,
                 bool enabled, const WindRaymarchSettings& settings,
                 const glm::mat4& invViewProjection, const glm::vec3& cameraPosition,
                 VkImageView sceneDepthView, VkBuffer windFieldBuffer, float timeSeconds);

    // Records the overlay draw. Must be inside a rendering scope whose color
    // attachment matches the swapchain format (and a D32 depth attachment).
    void render(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIdx, bool enabled);

    void cleanup(VulkanApp* app) override;

    bool ready() const { return pipeline_ != VK_NULL_HANDLE; }

private:
    void createPipeline(VulkanApp* app);
    void createDescriptors(VulkanApp* app);

    VulkanApp* app_ = nullptr;

    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkSampler depthSampler_ = VK_NULL_HANDLE;

    std::array<Buffer, VulkanApp::MAX_FRAMES_IN_FLIGHT> paramsBuffers_{};
    std::array<VkDescriptorSet, VulkanApp::MAX_FRAMES_IN_FLIGHT> descriptorSets_{};
    // Bound-resource guards: the descriptors are only rewritten when the scene
    // depth view (per frame slot) or the shared wind-field buffer changes.
    std::array<VkImageView, VulkanApp::MAX_FRAMES_IN_FLIGHT> boundDepthView_{};
    std::array<VkBuffer, VulkanApp::MAX_FRAMES_IN_FLIGHT> boundWindField_{};
    // True when prepare() staged this frame's draw (render() gates on it).
    bool active_ = false;
};
