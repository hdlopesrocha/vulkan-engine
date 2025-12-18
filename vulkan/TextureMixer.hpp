#pragma once

#pragma once

#include "EditableTexture.hpp"
#include "../utils/FileReader.hpp"
#include <random>
#include <cstring>
#include <functional>

// Vulkan-only helper that manages compute pipelines and the EditableTexture instances.
// UI is handled by `widgets::AnimatedTextureWidget`.
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
    void generateInitialTextures();

    void cleanup();

    // Getters for individual textures
    EditableTexture& getAlbedo();
    EditableTexture& getNormal();
    EditableTexture& getBump();

    const EditableTexture& getAlbedo() const;
    const EditableTexture& getNormal() const;
    const EditableTexture& getBump() const;

    // Generate Perlin noise for a texture using explicit parameters (used by UI widget)
    void generatePerlinNoiseWithParams(EditableTexture& texture, float scale, float octaves, float persistence, float lacunarity, float brightness, float contrast, float time, uint32_t seed);

private:
    VulkanApp* app = nullptr;
    VkSampler computeSampler = VK_NULL_HANDLE;
    EditableTexture albedo;
    EditableTexture normal;
    EditableTexture bump;

    // Optional reference to global texture arrays so compute can sample from them
    class TextureArrayManager* textureArrayManager = nullptr;

    // Compute pipeline for Perlin noise generation
    VkPipeline computePipeline = VK_NULL_HANDLE;
    VkPipelineLayout computePipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout computeDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool computeDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet albedoComputeDescSet = VK_NULL_HANDLE;
    VkDescriptorSet normalComputeDescSet = VK_NULL_HANDLE;
    VkDescriptorSet bumpComputeDescSet = VK_NULL_HANDLE;
    // Single descriptor set that binds the three storage images (albedo, normal, bump)
    VkDescriptorSet tripleComputeDescSet = VK_NULL_HANDLE;

    // Default Perlin parameters (kept here for compatibility)
    float perlinScale = 8.0f;
    float perlinOctaves = 4.0f;
    float perlinPersistence = 0.5f;
    float perlinLacunarity = 2.0f;
    float perlinBrightness = 0.0f;  // -1.0 to 1.0
    float perlinContrast = 5.0f;    // 0.0 to 5.0
    uint32_t perlinSeed = 12345;    // Fixed seed for consistent generation
    float perlinTime = 0.0f;        // Time parameter for noise evolution

    // Texture selection indices (grass=3, sand=8)
    int primaryTextureIdx = 3;      // Grass texture
    int secondaryTextureIdx = 8;    // Sand texture

    // Callback function to notify when textures are generated
    std::function<void()> onTextureGeneratedCallback;
    // If editable textures are represented inside a TextureArrayManager, store the layer index
public:
    // If editable textures are represented inside a TextureArrayManager, store the layer index
    uint32_t editableLayer = UINT32_MAX;

    void setEditableLayer(uint32_t layer) { editableLayer = layer; }
    // Return an ImGui descriptor for previewing the requested map (0=albedo,1=normal,2=bump)
    VkDescriptorSet getPreviewDescriptor(int map);
private:


    void createComputePipeline();
    void createComputeDescriptorSet(EditableTexture& texture, VkDescriptorSet& descSet);
    void createTripleComputeDescriptorSet();
    void generatePerlinNoise(EditableTexture& texture);
};

