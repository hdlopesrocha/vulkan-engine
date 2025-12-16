#pragma once

#include "vulkan.hpp"
#include <imgui.h>
#include <vector>
#include <string>
#include "TextureImage.hpp"
// Material properties per texture (defined in separate header)
#include "../utils/MaterialProperties.hpp"

class TextureManager {
public:
    struct Triple {
        int layerIndex = -1; // Index into the global texture arrays
        TextureImage albedo; // Kept for ImGui display
        VkSampler albedoSampler = VK_NULL_HANDLE;
        TextureImage normal;
        VkSampler normalSampler = VK_NULL_HANDLE;
        TextureImage height;
        VkSampler heightSampler = VK_NULL_HANDLE;
        // cached ImGui texture IDs for quick display
        ImTextureID albedoTexID = nullptr;
        ImTextureID normalTexID = nullptr;
        ImTextureID heightTexID = nullptr;
        
        // Material properties for this texture
        MaterialProperties material;
        
        // Ownership flag - if false, resources are managed externally (e.g., EditableTextureSet)
        bool ownsResources = true;
    };

    TextureManager() = default;
    void init(VulkanApp* app);

    // Loads a triple (albedo, normal, height). Returns index of the stored triple.
    size_t loadTriple(const std::string &albedoFile, const std::string &normalFile, const std::string &heightFile);

    // Add an existing triple (e.g., from EditableTextureSet). Returns index of the stored triple.
    size_t addTriple(const TextureImage& albedo, VkSampler albedoSampler,
                     const TextureImage& normal, VkSampler normalSampler,
                     const TextureImage& height, VkSampler heightSampler);

    // Create global texture arrays from collected filenames
    void createGlobalArrays();

    // Return an ImTextureID for ImGui rendering of the requested map (0=albedo,1=normal,2=height)
    ImTextureID getImTexture(size_t idx, int map);

    // Access material properties
    MaterialProperties& getMaterial(size_t idx);
    const MaterialProperties& getMaterial(size_t idx) const;

    // Access global arrays
    const TextureImage& getGlobalAlbedoArray() const { return globalAlbedoArray; }
    VkSampler getGlobalAlbedoSampler() const { return globalAlbedoSampler; }
    const TextureImage& getGlobalNormalArray() const { return globalNormalArray; }
    VkSampler getGlobalNormalSampler() const { return globalNormalSampler; }
    const TextureImage& getGlobalHeightArray() const { return globalHeightArray; }
    VkSampler getGlobalHeightSampler() const { return globalHeightSampler; }

    // number of loaded triples
    size_t count() const;

    const Triple& getTriple(size_t idx) const;

    void destroyAll();

    ~TextureManager();

private:
    VulkanApp* app = nullptr;
    std::vector<Triple> triples;

    // Collected filenames for global arrays
    std::vector<std::string> albedoFiles;
    std::vector<std::string> normalFiles;
    std::vector<std::string> heightFiles;

    // Global texture arrays
    TextureImage globalAlbedoArray;
    VkSampler globalAlbedoSampler = VK_NULL_HANDLE;
    TextureImage globalNormalArray;
    VkSampler globalNormalSampler = VK_NULL_HANDLE;
    TextureImage globalHeightArray;
    VkSampler globalHeightSampler = VK_NULL_HANDLE;
};

