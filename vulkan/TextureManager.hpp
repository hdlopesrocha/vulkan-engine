#pragma once

#include "vulkan.hpp"
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
    };

    TextureManager() = default;
    void init(VulkanApp* app) { this->app = app; }

    // Loads a triple (albedo, normal, height). Returns index of the stored triple.
    size_t loadTriple(const std::string &albedoFile, const std::string &normalFile, const std::string &heightFile);

    const Triple& getTriple(size_t idx) const { return triples.at(idx); }

    void destroyAll();

    ~TextureManager() { destroyAll(); }

private:
    VulkanApp* app = nullptr;
    std::vector<Triple> triples;
};
