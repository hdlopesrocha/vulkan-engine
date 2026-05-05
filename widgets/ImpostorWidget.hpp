#pragma once

#include "Widget.hpp"
#include "../vulkan/renderer/ImpostorCapture.hpp"
#include <vulkan/vulkan.h>
#include <cstdint>

class VulkanApp;
class VegetationRenderer;

// Dedicated widget for capturing and inspecting Fibonacci-sphere impostor views
// of billboard vegetation textures.
//
// Usage:
//   1. Call setVulkanApp() once after Vulkan init.
//   2. Call setSource() whenever the billboard array textures change (e.g. after bake).
//   3. Call setVegetationRenderer() to enable automatic wiring after capture.
//   4. cleanup() must be called before VulkanApp destruction.
class ImpostorWidget : public Widget {
public:
    ImpostorWidget();

    void setVulkanApp(VulkanApp* app);
    void setSource(VkImageView albedo, VkImageView normal, VkImageView opacity,
                   VkSampler sampler, int billboardCount);
    void setVegetationRenderer(VegetationRenderer* renderer) { vegRenderer = renderer; }
    // Re-applies setImpostorData to vegRenderer with already-captured data.
    // Call this after VegetationRenderer::init() if capture happened before init.
    void rewire();
    void cleanup();

    void render() override;

private:
    VulkanApp*          vulkanApp   = nullptr;
    VegetationRenderer* vegRenderer = nullptr;

    // Source billboard arrays (owned externally — BillboardCreator).
    VkImageView  srcAlbedo   = VK_NULL_HANDLE;
    VkImageView  srcNormal   = VK_NULL_HANDLE;
    VkImageView  srcOpacity  = VK_NULL_HANDLE;
    VkSampler    srcSampler  = VK_NULL_HANDLE;
    int          billboardCount = 0;

    // Active selection in the combo box.
    int  selectedBillboard  = 0;

    // Scale used for the capture
    float captureScale      = 10.0f;

    // Preview orbit state
    float previewYaw        = 0.0f;
    float previewPitch      = 0.3f;
    bool  previewNormals    = false; // toggle normal map preview in widget

    ImpostorCapture capture;
};
