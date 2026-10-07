#include <cstddef>
#include <functional>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// Convert in-place 8-bit RGBA sRGB values to linear (also 8-bit)
void convertSRGB8ToLinearInPlace(unsigned char* data, size_t pixelCount);

#ifdef __cplusplus
}
#endif
// TextureArrayManager declaration
#pragma once

#include <cstdint>
#include <array>
#include "Buffer.hpp"
#include "TextureImage.hpp"
#include <vector>
#include <backends/imgui_impl_vulkan.h>

struct TextureTriple { const char* albedo; const char* normal; const char* bump; const char* roughness = nullptr; const char* ao = nullptr; };

class TextureArrayManager {
public:
    // Number of layers in the texture arrays
    uint32_t layerAmount = 0;
    // Width and height of each layer
    uint32_t width = 0;
    uint32_t height = 0;

    // Texture arrays used by shaders (sampler2DArray)
    TextureImage albedoArray;
    TextureImage normalArray;
    TextureImage bumpArray;
    TextureImage roughnessArray;
    TextureImage aoArray;
    // Corresponding samplers for the arrays
    VkSampler albedoSampler = VK_NULL_HANDLE;
    VkSampler normalSampler = VK_NULL_HANDLE;
    VkSampler bumpSampler = VK_NULL_HANDLE;
    VkSampler roughnessSampler = VK_NULL_HANDLE;
    VkSampler aoSampler = VK_NULL_HANDLE;
    uint currentLayer = 0;
    // Back-pointer to `VulkanApp` (used for ImGui descriptor creation and legacy convenience methods).
    // Kept for backward compatibility with UI code that calls `getImTexture()` without an app.
    VulkanApp* appPtr = nullptr;

    // Per-layer 2D views and ImGui texture IDs (created on demand for UI display)
    std::vector<VkImageView> albedoLayerViews;
    std::vector<VkImageView> normalLayerViews;
    std::vector<VkImageView> bumpLayerViews;
    std::vector<VkImageView> roughnessLayerViews;
    std::vector<VkImageView> aoLayerViews;
    std::vector<ImTextureID> albedoImTextures;
    std::vector<ImTextureID> normalImTextures;
    std::vector<ImTextureID> bumpImTextures;
    std::vector<ImTextureID> roughnessImTextures;
    std::vector<ImTextureID> aoImTextures;
    // Track which layers have been initialized (contains valid data)
    std::vector<char> layerInitialized;

    // Track current per-layer image layout for each array (used by TextureMixer to pick correct oldLayout)
    std::vector<VkImageLayout> albedoLayerLayouts;
    std::vector<VkImageLayout> normalLayerLayouts;
    std::vector<VkImageLayout> bumpLayerLayouts;
    std::vector<VkImageLayout> roughnessLayerLayouts;
    std::vector<VkImageLayout> aoLayerLayouts;

    // Per-layer average albedo (linear RGB, computed from the uploaded pixels
    // at load time). Feeds the RT proxy albedo so secondary rays return
    // plausible terrain colors instead of flat gray (kills gray-vs-sky
    // rectangular tiles in grazing water reflections/refractions).
    std::vector<std::array<float, 3>> albedoAvg;
    // Linear-space average albedo of a layer; gray fallback when out of range.
    std::array<float, 3> albedoAverage(uint32_t layer) const;

    TextureArrayManager() = default;

    // Simple version counter incremented whenever GPU resources are (re)allocated
    uint32_t version = 0;
    uint32_t getVersion() const { return version; }

    // Register a callback invoked when GPU arrays are (re)allocated or destroyed.
    // Returns a listener id (>=0) that can be used to remove the listener.
    int addAllocationListener(std::function<void()> cb);
    void removeAllocationListener(int listenerId);

    // Destroy GPU resources (images, views, memory, samplers)
    void destroy(class VulkanApp* app);

    // Release the per-layer 2D views + ImGui descriptors (deferred via
    // VulkanApp::deferDestroyUntilAllPending). Split out of destroy() so a
    // resolution change can drop the views that reference the old images
    // before allocate() replaces them (perf report 23 C1). Does NOT notify
    // allocation listeners — callers reallocate afterwards.
    void releaseLayerViews(class VulkanApp* app);

    // Swap the five arrays to a new (layers, w, h) shape under a caller-held
    // device idle: drops the per-layer views/descriptors, then allocate()
    // replaces the images, bumping `version` and notifying listeners once
    // with the new views (perf report 23 C1). Content re-upload is the
    // caller's job (loadTriples).
    void recreate(class VulkanApp* app, uint32_t layers, uint32_t w, uint32_t h);

    // Log committed vs. initialized layers/bytes for the five arrays; warn
    // when fewer than 25% of the allocated layers hold data (perf report 23
    // C1, extending the report-22 pool telemetry to texture memory).
    void logMemoryUtilization(const char* name) const;

    // Variant of load/loadTriples that accepts an explicit VulkanApp instead of relying on an internal pointer
    uint load(class VulkanApp* app, const char* albedoFile, const char* normalFile, const char* bumpFile, const char* roughnessFile = nullptr, const char* aoFile = nullptr);
    size_t loadTriples(class VulkanApp* app, const std::vector<TextureTriple> &triples);

    // Invalidate all cached ImGui texture descriptors (e.g. after swapchain recreation
    // when the descriptor pool is destroyed). Next getImTexture() call re-creates them.
    void invalidateImGuiDescriptors();

    // Return an ImGui texture handle for a given array layer and map (0=albedo,1=normal,2=bump,3=roughness,4=ao)
    ImTextureID getImTexture(size_t layer, int map);
    // Variant that returns a descriptor sampling only the alpha channel of the
    // requested layer/map.  Used for visualizing noise or masks (swizzled view).
    ImTextureID getImTextureAlpha(size_t layer, int map);

    // Query/set layer initialized state
    bool isLayerInitialized(uint32_t layer) const;
    void setLayerInitialized(uint32_t layer, bool v=true);
    void allocate(uint32_t layers, uint32_t w, uint32_t h, class VulkanApp* app);

    // Query and update per-layer layouts (map: 0=albedo,1=normal,2=bump,3=roughness,4=ao)
    VkImageLayout getLayerLayout(int map, uint32_t layer) const;
    void setLayerLayout(int map, uint32_t layer, VkImageLayout layout);

private:
    // Listeners called when allocate()/destroy() change GPU resources
    std::vector<std::function<void()>> allocationListeners;

    // Notify registered listeners safely (copies callbacks and catches exceptions)
    void notifyAllocationListeners();

    // Persistent staging for layer uploads (perf report 23 C3). One buffer
    // sized to hold all five maps of one layer; overwritten in place by every
    // load(), so the bring-up path performs zero staging allocations/frees
    // and no zero-init memset. Released on allocate()/destroy() so a
    // resolution change re-sizes it on the next load.
    Buffer stagingBuffer_{};
    size_t stagingBufferSize_ = 0;
    Buffer& ensureStagingBuffer(class VulkanApp* app, size_t needBytes);
    void releaseStagingBuffer(class VulkanApp* app);

};
