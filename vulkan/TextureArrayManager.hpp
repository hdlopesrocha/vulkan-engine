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
    // Back-reference to the VulkanApp used to allocate resources (optional)
    class VulkanApp* app = nullptr;
    uint currentLayer = 0;
    // Per-layer 2D views and ImGui texture IDs (created on demand for UI display)
    std::vector<VkImageView> albedoLayerViews;
    std::vector<VkImageView> normalLayerViews;
    std::vector<VkImageView> bumpLayerViews;
    std::vector<ImTextureID> albedoImTextures;
    std::vector<ImTextureID> normalImTextures;
    std::vector<ImTextureID> bumpImTextures;
    // Track which layers have been initialized (contains valid data)
    std::vector<char> layerInitialized;
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

    // Return an ImGui texture handle for a given array layer and map (0=albedo,1=normal,2=bump)
    ImTextureID getImTexture(size_t layer, int map);

    // Load a triple of images into the current layer and increment the layer counter
    uint load(const char* albedoFile, const char* normalFile, const char* bumpFile);

    // Convenience: load multiple triples (albedo, normal, bump) into consecutive layers.
    // Returns the number of successfully loaded layers.
    size_t loadTriples(const std::vector<std::tuple<const char*, const char*, const char*>> &triples);

    // Update a specific array layer from an EditableTexture (copies image -> array layer)
    void updateLayerFromEditable(uint32_t layer, const class EditableTexture& tex);

    // Update a specific array layer and map (0=albedo,1=normal,2=bump) from an EditableTexture
    void updateLayerFromEditableMap(uint32_t layer, const class EditableTexture& tex, int map);
    // Query/set layer initialized state
    bool isLayerInitialized(uint32_t layer) const;
    void setLayerInitialized(uint32_t layer, bool v=true);

private:
    // Listeners called when allocate()/destroy() change GPU resources
    std::vector<std::function<void()>> allocationListeners;

    // Notify registered listeners safely (copies callbacks and catches exceptions)
    void notifyAllocationListeners();

    // Destroy GPU resources (images, views, memory, samplers)
    void destroy(class VulkanApp* app);
    // Create an empty (zeroed) triple at the current layer for later editing, then increment layer counter
    uint create();

    // Allocate GPU image arrays via the provided VulkanApp
    void allocate(uint32_t layers, uint32_t w, uint32_t h, class VulkanApp* app);
};
