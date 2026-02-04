#pragma once

#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>
class VulkanApp;
#include "../utils/FileReader.hpp"
#include <random>
#include <cstring>
#include <functional>
#include <mutex>
#include <vector>

// Vulkan-only helper that manages compute pipelines and the EditableTexture instances.
// UI is handled by `widgets::TextureMixerWidget`.
struct MixerParameters {
    size_t targetLayer;
    uint primaryTextureIdx;
    uint secondaryTextureIdx;
    // Default Perlin parameters (kept here for compatibility)
    float perlinScale = 8.0f;
    float perlinOctaves = 4.0f;
    float perlinPersistence = 0.5f;
    float perlinLacunarity = 2.0f;
    float perlinBrightness = 0.0f;  // -1.0 to 1.0
    float perlinContrast = 5.0f;    // 0.0 to 5.0
    uint32_t perlinSeed = 12345;    // Fixed seed for consistent generation
    float perlinTime = 0.0f;        // Time parameter for noise evolution
};


class TextureMixer {
public:
    TextureMixer();
    
    // Pass optional TextureArrayManager so compute can sample from arrays
    // Backwards-compatible init: old callers that don't pass a TextureArrayManager
    void init(VulkanApp* app, uint32_t width, uint32_t height);

    // New init that accepts an optional TextureArrayManager so compute can sample from arrays
    void init(VulkanApp* app, uint32_t width, uint32_t height, class TextureArrayManager* textureArrayManager);

    // Set callback to be called after texture generation
    void setOnTextureGenerated(std::function<void()> callback);

    // Generate all textures initially
    void generateInitialTextures(std::vector<MixerParameters> &mixerParams);

    // Queue a generation request from UI thread; flushed synchronously from main update
    void enqueueGenerate(const MixerParameters &params, int map = -1);
    // Flush pending generation requests synchronously (call from main update loop)
    void flushPendingRequests();

    // Poll for completed async generation fences and invoke callbacks (called from update()/preRender)
    void pollPendingGenerations();

    // Diagnostics: number of pending async generations and a small log buffer
    size_t getPendingGenerationCount();
    std::vector<std::string> consumeLogs();

    void cleanup();

    // Query array layer dimensions (0 if none)
    uint32_t getLayerWidth() const;
    uint32_t getLayerHeight() const;
    // Bytes per pixel for array textures (RGBA8 -> 4)
    int getBytesPerPixel() const;

    // Generate Perlin noise for a texture using explicit parameters (used by UI widget)
    // map: -1 = all maps, 0 = albedo, 1 = normal, 2 = bump
    void generatePerlinNoise(MixerParameters &params, int map = -1);

private:
    VulkanApp* app = nullptr;
    VkSampler computeSampler = VK_NULL_HANDLE;
    // EditableTexture instances removed; use TextureArrayManager arrays instead
    uint32_t width = 0, height = 0;
    // Optional reference to global texture arrays so compute can sample from them
    class TextureArrayManager* textureArrayManager = nullptr;

    // Compute pipeline for Perlin noise generation
    VkPipeline computePipeline = VK_NULL_HANDLE;
    VkPipelineLayout computePipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout computeDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool computeDescriptorPool = VK_NULL_HANDLE;
    // Single descriptor set that binds the three storage images (albedo, normal, bump)
    VkDescriptorSet tripleComputeDescSet = VK_NULL_HANDLE;
    // Per-map descriptor sets (allow generating a single map)
    VkDescriptorSet albedoComputeDescSet = VK_NULL_HANDLE;
    VkDescriptorSet normalComputeDescSet = VK_NULL_HANDLE;
    VkDescriptorSet bumpComputeDescSet = VK_NULL_HANDLE;


    // Callback function to notify when textures are generated
    std::function<void()> onTextureGeneratedCallback;

    // Pending generation requests (thread-safe queue)
    std::mutex pendingRequestsMutex;
    std::vector<std::pair<MixerParameters,int>> pendingRequests;

    // Pending async fences (fence, layer) for in-flight generation submissions
    std::mutex pendingFencesMutex;
    std::vector<std::tuple<VkFence, uint32_t>> pendingFences;

    // Diagnostics: small textual log buffer for UI and a mutex to protect it
    std::mutex logsMutex;
    std::vector<std::string> logs;

public:
    // Query whether a layer currently has an in-flight generation
    bool isLayerGenerationPending(uint32_t layer);
    // Block until generation for a specific layer completes (returns true if waited)
    // This uses Vulkan fences and will block until the generation fence signals.
    bool waitForLayerGeneration(uint32_t layer, uint64_t timeoutNs = UINT64_MAX);

    // Global instance accessor (set on init) so external systems can wait for generations
    static TextureMixer* getGlobalInstance();

    // If editable textures are represented inside a TextureArrayManager, store the layer index
public:
    // If editable textures are represented inside a TextureArrayManager, store the layer index
    uint32_t editableLayer = UINT32_MAX;

    void setEditableLayer(uint32_t layer) { editableLayer = layer; }
    // Return an ImGui descriptor for previewing the requested map (0=albedo,1=normal,2=bump)
    VkDescriptorSet getPreviewDescriptor(int map);
    // Return an ImGui descriptor for previewing the requested map at a specific array layer
    VkDescriptorSet getPreviewDescriptor(int map, uint32_t layer);
    // Return number of layers in the attached TextureArrayManager (0 if none)
    uint32_t getArrayLayerCount() const;

    // Re-write compute descriptor sets after a TextureArrayManager is available
    void updateComputeDescriptorSets();

    // Attach/replace the TextureArrayManager used and refresh descriptors
    void attachTextureArrayManager(TextureArrayManager* tam);


private:


    void createComputePipeline();
    void createTripleComputeDescriptorSet();
    // Helper to create a descriptor set targeting a single map (map index: 0=albedo,1=normal,2=bump)
    void createComputeDescriptorSet(int map, VkDescriptorSet& descSet);

};

