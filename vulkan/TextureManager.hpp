#pragma once

#include "vulkan.hpp"
#include <imgui.h>
#include <vector>
#include <string>

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
    };

    TextureManager() = default;
    void init(VulkanApp* app) { this->app = app; }

    // Loads a triple (albedo, normal, height). Returns index of the stored triple.
    size_t loadTriple(const std::string &albedoFile, const std::string &normalFile, const std::string &heightFile);

    // Return an ImTextureID for ImGui rendering of the requested map (0=albedo,1=normal,2=height)
    ImTextureID getImTexture(size_t idx, int map);

    // number of loaded triples
    size_t count() const { return triples.size(); }

    const Triple& getTriple(size_t idx) const { return triples.at(idx); }

    void destroyAll();

    ~TextureManager() { destroyAll(); }

private:
    VulkanApp* app = nullptr;
    std::vector<Triple> triples;
};
