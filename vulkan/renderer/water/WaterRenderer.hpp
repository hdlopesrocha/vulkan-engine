#pragma once

#include "../Renderer.hpp"
#include "../../core/VulkanApp.hpp"
#include "../../resources/Buffer.hpp"
#include "../../../types/WaterSettings.hpp"
#include "../../ssbo/WaterParams.hpp"
#include "../../ubo/WaterRenderUBO.hpp"

#include <cstdint>
#include <vector>

// Water is rendered by the octree SDF raymarch (OctreeRaymarchRenderer):
// there is no water geometry pass, back-face pass or water offscreen target
// pair any more. This class is the small parameter service that survives:
// it owns the per-layer WaterParams SSBO contents (the scene descriptor binds
// that buffer) and the WaterRenderUBO, and exposes a zeroed 1x1 color view the
// composite binds for its (now unused) water input.
class WaterRenderer : public Renderer {
public:
    WaterRenderer();
    ~WaterRenderer();

    void init(VulkanApp* app, Buffer& waterParamsBuffer, const std::vector<WaterSettings>& waterParams,
              uint32_t layerCount);
    void cleanup(VulkanApp* app) override;

    // Water render UBO (set 0 binding 10), bound by SceneRenderer.
    Buffer& getWaterRenderUBO() { return waterRenderUBO_; }

    // Per-layer parameter upload (WaterWidget edits).
    void updateGPUParamsForLayer(uint32_t layer, const WaterSettings& params);

    // Live music-analysis upload for MUSIC_REACTIVE_WATER.
    void updateMusicAudio(float amplitude, float bass, float mid, float high,
                          float beat, bool enabled);

    // The composite's water input collapses to the base color (zero alpha).
    VkImageView getDummyWaterColorView() const { return dummyWaterView_; }
    // Historical target accessors kept as dummies for the debug preview widget
    // and the composite signature. No water pass writes these.
    VkImageView getWaterDepthView(uint32_t) const { return dummyWaterView_; }
    VkImageView getWaterGeomDepthView(uint32_t) const { return dummyWaterView_; }
    VkImage getWaterGeomDepthImage(uint32_t) const { return dummyWaterImage_; }
    VkImageView getWaterBodyView(uint32_t) const { return dummyWaterView_; }
    VkImageView getWaterColumnView(uint32_t) const { return dummyWaterView_; }
    VkImageLayout getWaterGeomDepthLayout(uint32_t) const {
        return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    void setWaterGeomDepthLayout(uint32_t, VkImageLayout) {}

    void createRenderTargets(VulkanApp*, uint32_t, uint32_t) {}
    void destroyRenderTargets(VulkanApp*) {}
    void clearRenderTargets(VulkanApp*, VkCommandBuffer, uint32_t) {}

private:
    void ensureDummyView(VulkanApp* app);

    Buffer waterParamsBuffer;
    uint32_t waterParamsCount = 0;
    VulkanApp* appPtr = nullptr;

    Buffer waterRenderUBO_;

    VkImage dummyWaterImage_ = VK_NULL_HANDLE;
    VmaAllocation dummyWaterAlloc_ = VK_NULL_HANDLE;
    VkDeviceMemory dummyWaterMem_ = VK_NULL_HANDLE;
    VkImageView dummyWaterView_ = VK_NULL_HANDLE;
};
