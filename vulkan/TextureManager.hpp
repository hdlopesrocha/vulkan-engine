#pragma once

#include "vulkan.hpp"
#include <imgui.h>
#include <vector>
#include <string>

// Material properties per texture (defined in separate header)
#include "../utils/MaterialProperties.hpp"

class TextureManager {
public:
    struct Triple {
        TextureImage albedo;
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
    void init(VulkanApp* app) { this->app = app; }

    // Loads a triple (albedo, normal, height). Returns index of the stored triple.
    size_t loadTriple(const std::string &albedoFile, const std::string &normalFile, const std::string &heightFile);

    // Add an existing triple (e.g., from EditableTextureSet). Returns index of the stored triple.
    size_t addTriple(const TextureImage& albedo, VkSampler albedoSampler,
                     const TextureImage& normal, VkSampler normalSampler,
                     const TextureImage& height, VkSampler heightSampler);

    // Return an ImTextureID for ImGui rendering of the requested map (0=albedo,1=normal,2=height)
    ImTextureID getImTexture(size_t idx, int map);

    // Access material properties
    MaterialProperties& getMaterial(size_t idx) { return triples.at(idx).material; }
    const MaterialProperties& getMaterial(size_t idx) const { return triples.at(idx).material; }

    // number of loaded triples
    size_t count() const { return triples.size(); }

    const Triple& getTriple(size_t idx) const { return triples.at(idx); }

    void destroyAll();

    ~TextureManager() { destroyAll(); }

private:
    VulkanApp* app = nullptr;
    std::vector<Triple> triples;
};

