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
#include "TextureImage.hpp"
#include <vector>
#include <backends/imgui_impl_vulkan.h>

struct TextureTriple { const char* albedo; const char* normal; const char* bump; };

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
    // Corresponding samplers for the arrays
    VkSampler albedoSampler = VK_NULL_HANDLE;
    VkSampler normalSampler = VK_NULL_HANDLE;
    VkSampler bumpSampler = VK_NULL_HANDLE;
    uint currentLayer = 0;
    // Back-pointer to `VulkanApp` (used for ImGui descriptor creation and legacy convenience methods).
    // Kept for backward compatibility with UI code that calls `getImTexture()` without an app.
    VulkanApp* app = nullptr;

    // Per-layer 2D views and ImGui texture IDs (created on demand for UI display)
    std::vector<VkImageView> albedoLayerViews;
    std::vector<VkImageView> normalLayerViews;
    std::vector<VkImageView> bumpLayerViews;
    std::vector<ImTextureID> albedoImTextures;
    std::vector<ImTextureID> normalImTextures;
    std::vector<ImTextureID> bumpImTextures;
    // Track which layers have been initialized (contains valid data)
    std::vector<char> layerInitialized;

    // Track current per-layer image layout for each array (used by TextureMixer to pick correct oldLayout)
    std::vector<VkImageLayout> albedoLayerLayouts;
    std::vector<VkImageLayout> normalLayerLayouts;
    std::vector<VkImageLayout> bumpLayerLayouts;

    TextureArrayManager() = default;
    TextureArrayManager(uint32_t layers, uint32_t w, uint32_t h)
        : layerAmount(layers), width(w), height(h) {}
    // Simple version counter incremented whenever GPU resources are (re)allocated
    uint32_t version = 0;
    uint32_t getVersion() const { return version; }

    // Register a callback invoked when GPU arrays are (re)allocated or destroyed.
    // Returns a listener id (>=0) that can be used to remove the listener.
    int addAllocationListener(std::function<void()> cb);
    void removeAllocationListener(int listenerId);

    // Allocate host-only metadata (no GPU resources)
    void allocate(uint32_t layers, uint32_t w, uint32_t h);

    // Public initializer that completes GPU resource allocation using an app
    void initialize(class VulkanApp* app) { allocate(layerAmount, width, height, app); }

    // Destroy GPU resources (images, views, memory, samplers)
    void destroy(class VulkanApp* app);

    // Variant of load/create/update that accepts an explicit VulkanApp instead of relying on an internal pointer
    uint load(class VulkanApp* app, const char* albedoFile, const char* normalFile, const char* bumpFile);
    size_t loadTriples(class VulkanApp* app, const std::vector<TextureTriple> &triples);
    uint create(class VulkanApp* app);
    void updateLayerFromEditable(class VulkanApp* app, uint32_t layer, const class EditableTexture& tex);
    void updateLayerFromEditableMap(class VulkanApp* app, uint32_t layer, const class EditableTexture& tex, int map);

    // Return an ImGui texture handle for a given array layer and map (0=albedo,1=normal,2=bump)
    ImTextureID getImTexture(size_t layer, int map);

    // Load a triple of images into the current layer and increment the layer counter
    uint load(const char* albedoFile, const char* normalFile, const char* bumpFile);

    // Convenience: load multiple triples (albedo, normal, bump) into consecutive layers.
    // Returns the number of successfully loaded layers.
    size_t loadTriples(const std::vector<TextureTriple> &triples);

    // Update a specific array layer from an EditableTexture (copies image -> array layer)
    // (No-app convenience overload removed — prefer caller to pass `VulkanApp*`.)


    // Query/set layer initialized state
    bool isLayerInitialized(uint32_t layer) const;
    void setLayerInitialized(uint32_t layer, bool v=true);

    // Query and update per-layer layouts (map: 0=albedo,1=normal,2=bump)
    VkImageLayout getLayerLayout(int map, uint32_t layer) const;
    void setLayerLayout(int map, uint32_t layer, VkImageLayout layout);

private:
    // Listeners called when allocate()/destroy() change GPU resources
    std::vector<std::function<void()>> allocationListeners;

    // Notify registered listeners safely (copies callbacks and catches exceptions)
    void notifyAllocationListeners();

    // Create an empty (zeroed) triple at the current layer for later editing, then increment layer counter
    uint create();

    // Allocate GPU image arrays via the provided VulkanApp
    void allocate(uint32_t layers, uint32_t w, uint32_t h, class VulkanApp* app);
};
