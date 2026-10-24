#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>
class VulkanApp;
#include "../utils/FileReader.hpp"
#include "MixerParameters.hpp"
#include <random>
#include <cstring>
#include <mutex>
#include <vector>

// Vulkan-only helper that manages the Perlin generation compute pipeline and
// its descriptor set. UI is handled by `widgets::TextureMixerWidget`.


class TextureMixer {
public:
    TextureMixer();

    // New init that accepts an optional TextureArrayManager so compute can sample from arrays
    void init(VulkanApp* app, class TextureArrayManager* textureArrayManager);

    // Generate all textures initially
    void generateInitialTextures(std::vector<MixerParameters> &mixerParams);

    // Queue a generation request from UI thread; flushed synchronously from main update
    void enqueueGenerate(const MixerParameters &params, int map = -1);
    // Flush pending generation requests synchronously (call from main update loop)
    void flushPendingRequests(VulkanApp* app);

    // Poll for completed async generation fences and invoke callbacks (called from update()/preRender)
    void pollPendingGenerations(VulkanApp* app);

    // Diagnostics: number of pending async generations
    size_t getPendingGenerationCount();

    // Query array layer dimensions (0 if none)
    uint32_t getLayerWidth() const;
    uint32_t getLayerHeight() const;
    // Generate Perlin noise for a texture using explicit parameters (used by UI widget)
    // map: -1 = all maps, 0 = albedo, 1 = normal, 2 = bump
    void generatePerlinNoise(VulkanApp* app, MixerParameters &params, int map = -1);

private:
    // No stored VulkanApp*; callers pass `VulkanApp*` to methods that need it
    // EditableTexture instances removed; use TextureArrayManager arrays instead
    uint32_t width = 0, height = 0;
    // Optional reference to global texture arrays so compute can sample from them
    class TextureArrayManager* textureArrayManager = nullptr;

    // Compute pipeline for Perlin noise generation.
    // Perf report 23 C2: one descriptor set is rewritten per generation with
    // single-layer views for the target / primary / secondary layers, so a
    // generation only touches those three layers (the old triple / per-map /
    // per-layer persistent-set machinery was removed with the array-view
    // bindings).
    VkPipeline computePipeline = VK_NULL_HANDLE;
    VkPipelineLayout computePipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout computeDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool computeDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet generationDescSet = VK_NULL_HANDLE;

    // Ensure the per-layer 2D view for (layer, map) exists and return it
    // (map: 0=albedo, 1=normal, 2=bump, 3=roughness, 4=ao).
    VkImageView layerViewFor(uint32_t layer, int map);


    // Pending generation requests (thread-safe queue)
    std::mutex pendingRequestsMutex;
    std::vector<std::pair<MixerParameters,int>> pendingRequests;

    // Pending async fences (fence, layer) for in-flight generation submissions
    std::mutex pendingFencesMutex;
    std::vector<std::tuple<VkFence, uint32_t>> pendingFences;
    std::vector<std::tuple<VkFence, uint32_t>> completed;

public:
    // Return an ImGui descriptor for previewing the requested map at a specific array layer
    VkDescriptorSet getPreviewDescriptor(int map, uint32_t layer);
    // Return a descriptor which samples only the alpha channel of the specified layer
    VkDescriptorSet getNoiseDescriptor(uint32_t layer);
    // Return number of layers in the attached TextureArrayManager (0 if none)
    uint32_t getArrayLayerCount() const;

    // Attach/replace the TextureArrayManager used by generations. With the
    // C2 per-generation bindings there are no persistent sets to refresh:
    // the next generatePerlinNoise call reads the current views.
    void attachTextureArrayManager(TextureArrayManager* tam);


private:


    void createComputePipeline(VulkanApp* app);

};

