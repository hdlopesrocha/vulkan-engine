#include "TextureManager.hpp"
#include <stdexcept>

size_t TextureManager::loadTriple(const std::string &albedoFile, const std::string &normalFile, const std::string &heightFile) {
    if (!app) throw std::runtime_error("TextureManager not initialized with VulkanApp pointer");

    Triple t;
    // create array textures with a single layer each (keeps current behavior but centralizes resource management)
    t.albedo = app->createTextureImageArray({ albedoFile });
    t.albedoSampler = app->createTextureSampler(t.albedo.mipLevels);

    t.normal = app->createTextureImageArray({ normalFile });
    t.normalSampler = app->createTextureSampler(t.normal.mipLevels);

    t.height = app->createTextureImageArray({ heightFile });
    t.heightSampler = app->createTextureSampler(t.height.mipLevels);

    triples.push_back(t);
    return triples.size() - 1;
}

void TextureManager::destroyAll() {
    if (!app) return;
    VkDevice device = app->getDevice();
    for (auto &t : triples) {
        if (t.albedoSampler != VK_NULL_HANDLE) vkDestroySampler(device, t.albedoSampler, nullptr);
        if (t.albedo.view != VK_NULL_HANDLE) vkDestroyImageView(device, t.albedo.view, nullptr);
        if (t.albedo.image != VK_NULL_HANDLE) vkDestroyImage(device, t.albedo.image, nullptr);
        if (t.albedo.memory != VK_NULL_HANDLE) vkFreeMemory(device, t.albedo.memory, nullptr);

        if (t.normalSampler != VK_NULL_HANDLE) vkDestroySampler(device, t.normalSampler, nullptr);
        if (t.normal.view != VK_NULL_HANDLE) vkDestroyImageView(device, t.normal.view, nullptr);
        if (t.normal.image != VK_NULL_HANDLE) vkDestroyImage(device, t.normal.image, nullptr);
        if (t.normal.memory != VK_NULL_HANDLE) vkFreeMemory(device, t.normal.memory, nullptr);

        if (t.heightSampler != VK_NULL_HANDLE) vkDestroySampler(device, t.heightSampler, nullptr);
        if (t.height.view != VK_NULL_HANDLE) vkDestroyImageView(device, t.height.view, nullptr);
        if (t.height.image != VK_NULL_HANDLE) vkDestroyImage(device, t.height.image, nullptr);
        if (t.height.memory != VK_NULL_HANDLE) vkFreeMemory(device, t.height.memory, nullptr);
    }
    triples.clear();
}
