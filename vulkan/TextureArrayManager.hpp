// TextureArrayManager declaration
#pragma once

#include <cstdint>
#include "TextureImage.hpp"

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
    TextureArrayManager() = default;
    TextureArrayManager(uint32_t layers, uint32_t w, uint32_t h)
        : layerAmount(layers), width(w), height(h) {}
    // Allocate host-only metadata (no GPU resources)
    void allocate(uint32_t layers, uint32_t w, uint32_t h);

    // Allocate GPU image arrays via the provided VulkanApp
    void allocate(uint32_t layers, uint32_t w, uint32_t h, class VulkanApp* app);

    // Destroy GPU resources (images, views, memory, samplers)
    void destroy(class VulkanApp* app);
    // Load a triple of images into the current layer and increment the layer counter
    uint load(char* albedoFile, char* normalFile, char* bumpFile);
    // Create an empty (zeroed) triple at the current layer for later editing, then increment layer counter
    uint create();
};
