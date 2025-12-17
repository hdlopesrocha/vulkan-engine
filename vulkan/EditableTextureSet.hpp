#pragma once

#include "EditableTexture.hpp"
#include "TextureManager.hpp"
#include "FileReader.hpp"
#include "../widgets/Widget.hpp"
#include <imgui.h>
#include <random>
#include <cstring>
#include <functional>

class EditableTextureSet : public Widget {
public:
    EditableTextureSet();
    
    void init(VulkanApp* app, uint32_t width, uint32_t height, const char* windowName = "Editable Textures");
    
    void setTextureManager(TextureManager* texMgr);
    
    // Set callback to be called after texture generation
    void setOnTextureGenerated(std::function<void()> callback);
    
    // Generate all textures initially
    void generateInitialTextures();
    
    void cleanup();
    
    // Render a single ImGui window with tabs
    void render() override;
    
    // Getters for individual textures
    EditableTexture& getAlbedo();
    EditableTexture& getNormal();
    EditableTexture& getBump();

    const EditableTexture& getAlbedo() const;
    const EditableTexture& getNormal() const;
    const EditableTexture& getBump() const;
    
private:
    VulkanApp* app = nullptr;
    TextureManager* textureMgr = nullptr;
    EditableTexture albedo;
    EditableTexture normal;
    EditableTexture bump;
    
    // Compute pipeline for Perlin noise generation
    VkPipeline computePipeline = VK_NULL_HANDLE;
    VkPipelineLayout computePipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout computeDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool computeDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet albedoComputeDescSet = VK_NULL_HANDLE;
    VkDescriptorSet normalComputeDescSet = VK_NULL_HANDLE;
    VkDescriptorSet bumpComputeDescSet = VK_NULL_HANDLE;
    
    // Perlin noise parameters
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

    // Index of the editable triple in TextureManager (set by caller)
    size_t editableLayerIndex = SIZE_MAX;

public:
    // Set the layer index corresponding to the editable triple in TextureManager
    void setEditableLayerIndex(size_t idx) { editableLayerIndex = idx; }
    
    // Previous parameter values for change detection
    float prevPerlinScale = 8.0f;
    float prevPerlinOctaves = 4.0f;
    float prevPerlinPersistence = 0.5f;
    float prevPerlinLacunarity = 2.0f;
    float prevPerlinBrightness = 0.0f;
    float prevPerlinContrast = 5.0f;
    float prevPerlinTime = 0.0f;
    int prevPrimaryTextureIdx = 3;      // Match initial primary (grass)
    int prevSecondaryTextureIdx = 8;    // Match initial secondary (sand)
    
    void renderTextureTab(EditableTexture& texture);
    
    void createComputePipeline();
    
    void createComputeDescriptorSet(EditableTexture& texture, VkDescriptorSet& descSet);
    
    void generatePerlinNoise(EditableTexture& texture);
};
