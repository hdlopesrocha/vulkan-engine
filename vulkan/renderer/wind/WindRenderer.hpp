#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <array>
#include <cstdint>

#include "../Renderer.hpp"
#include "../../core/VulkanApp.hpp"
#include "../../resources/Buffer.hpp"
#include "../../ubo/WindParticlesUBO.hpp"
#include "../../../types/WindParticlesSettings.hpp"

// GPU wind-trail particle renderer.
//
// Presentation layer for the SHARED wind field (vulkan/ubo/WindField.hpp):
// `particleCount` particles live inside the camera frustum, are advected every
// frame by a compute shader that samples windSVF (the exact GLSL function the
// vegetation/fire/cloud shaders use), and leave a fading, time-stamped trail.
// A particle whose head leaves the view frustum is re-spawned at a random point
// inside it.
//
// Two passes, both recorded on the MAIN command buffer:
//   dispatch()  — compute; must be called OUTSIDE any rendering scope (compute
//                 cannot be recorded inside dynamic rendering). Records the
//                 particle step + a compute->vertex/fragment barrier.
//   render()    — trail draw; called inside the final swapchain rendering scope
//                 (after the composite), blended over the composed frame.
//
// The particle state + trail ring are ONE persistent device-local simulation
// (not per-frame-in-flight copies): a trail must survive the frame ring, and
// all touches happen on the main queue, serialized by the frame timeline/fence
// exactly like the water-droplet simulation (see ParticlePhysicsSim). The
// descriptor sets and the small parameters UBO are per-frame-in-flight.
class WindRenderer : public Renderer {
public:
    // Particle capacity of the fixed state/trail pools. Settings::particleCount
    // is clamped to this; the buffers are allocated once.
    static constexpr uint32_t kMaxParticles = 4096;
    // Trail ring length in samples. Together with kTrailSampleInterval this
    // bounds the maximum trail duration (191 * 1/24 s ~= 7.96 s).
    static constexpr uint32_t kTrailCapacity = 192;
    // Fixed trail sampling period (s). Sampling by TIME (not per frame) keeps
    // the trail duration in seconds identical at any frame rate.
    static constexpr float kTrailSampleInterval = 1.0f / 24.0f;

    WindRenderer() = default;
    ~WindRenderer() override { cleanup(nullptr); }

    void init(VulkanApp* app);

    // Records this frame's simulation dispatch. Call from preRenderPass (main
    // command buffer, BEFORE the dynamic-rendering scope is opened). `enabled`
    // false (or no wind-field buffer) records nothing and disables render().
    void dispatch(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIdx,
                  bool enabled, const WindParticlesSettings& settings,
                  const glm::mat4& viewProjection, const glm::mat4& invViewProjection,
                  const glm::vec3& cameraPosition, float cameraFarPlane,
                  const glm::vec2& screenSize, float timeSeconds, float deltaSeconds,
                  VkBuffer windFieldBuffer, bool windEnabled);

    // Records the trail draw. Call inside the final swapchain rendering scope
    // (after the post-process composite, before ImGui).
    void render(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIdx,
                bool enabled, const WindParticlesSettings& settings);

    void cleanup(VulkanApp* app) override;

    bool ready() const { return computePipeline_ != VK_NULL_HANDLE && trailPipeline_ != VK_NULL_HANDLE; }

    // Trail segments drawn per particle for a given requested duration. Shared
    // by the host draw-count math and (mirrored) by WindTrail.vert.
    static uint32_t trailSegments(float durationSeconds);

private:
    void createComputePipeline(VulkanApp* app);
    void createTrailPipeline(VulkanApp* app);
    void createDescriptors(VulkanApp* app);
    void createBuffers(VulkanApp* app);
    // One compute -> vertex/fragment dependency for the particle/trail buffers.
    static void barrierAfterDispatch(VkCommandBuffer cmd);
    // One host -> compute/vertex/fragment dependency for the parameters UBO.
    static void barrierBeforeDispatch(VkCommandBuffer cmd);

    VulkanApp* app_ = nullptr;

    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkPipelineLayout computePipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline computePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout trailPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline trailPipeline_ = VK_NULL_HANDLE;

    // Persistent simulation state (device-local): one record per particle plus
    // the per-particle trail ring (positions/times + the wind speed recorded at
    // each sample, used for the speed color ramp).
    Buffer particleBuffer_{};
    Buffer trailBuffer_{};
    Buffer trailSpeedBuffer_{};

    // Per-frame-in-flight parameters UBO + descriptor set (the wind-field
    // binding is rewritten only when its buffer handle changes).
    std::array<Buffer, VulkanApp::MAX_FRAMES_IN_FLIGHT> paramsBuffers_{};
    std::array<VkDescriptorSet, VulkanApp::MAX_FRAMES_IN_FLIGHT> descriptorSets_{};
    std::array<VkBuffer, VulkanApp::MAX_FRAMES_IN_FLIGHT> boundWindField_{};

    // True when dispatch() recorded a step this frame (render() gates on it).
    bool active_ = false;
    // Monotonic dispatch counter, used as the per-dispatch spawn seed.
    uint32_t seedCounter_ = 0;
};
