#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>

class VulkanApp;

// Ray-marched water-droplet renderer.
//
// One camera-facing billboard per simulated droplet (6 non-indexed vertices,
// instanced over the droplet state buffer) drawn inside the solid color pass
// after the solid geometry: depth test on, depth write off, premultiplied
// alpha blending. The fragment stage ray-marches the droplet's SDF sphere and
// shades refraction + reflection from the sky view, so the render path is
// direct SDF ray marching — no triangle mesh is generated for a droplet and
// the simulation never depends on this pass.
//
// Set 0 is the shared scene set (camera UBO + sky UBO). Set 1 is owned here:
//   binding 0 = droplet state SSBO (the simulation's device-local buffer)
//   binding 1 = sky/environment colour texture (direction -> equirect UV)
// Push constant: vec4 params = (particleCount, radiusScale, ior, debugMode).
class DropletRenderer {
public:
    void init(VulkanApp* app);
    void destroy(VulkanApp* app);

    // Records the draw. `count` is the simulated particle count, `radiusScale`
    // scales the rendered sphere (1 = the collision radius). Returns false when
    // the pass is not ready (no sky view / no state buffer) and nothing was
    // recorded.
    bool draw(VkCommandBuffer cmd, VulkanApp* app, uint32_t frameIndex,
              VkDescriptorSet mainSet, VkBuffer dropletState, uint32_t count,
              float radiusScale, float ior, float debugMode, VkImageView skyView);

    bool ready() const { return pipeline_ != VK_NULL_HANDLE; }

private:
    void writeSet(uint32_t slot, VkBuffer state, VkImageView sky);

    VulkanApp* app_ = nullptr;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
    static constexpr uint32_t kSlots = 3; // VulkanApp::MAX_FRAMES_IN_FLIGHT
    VkDescriptorSet sets_[kSlots] = {};
    VkBuffer boundState_[kSlots] = {};
    VkImageView boundSky_[kSlots] = {};
};
