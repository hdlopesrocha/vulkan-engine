#include "TextureManager.hpp"
#include <stdexcept>
#include <backends/imgui_impl_vulkan.h>

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

    // Initialize material properties with reasonable defaults
    t.material.pomHeightScale = 0.06f;
    t.material.pomMinLayers = 8.0f;
    t.material.pomMaxLayers = 32.0f;
    t.material.pomEnabled = 1.0f;
    t.material.flipNormalY = 0.0f;
    t.material.flipTangentHandedness = 0.0f;
    t.material.ambientFactor = 0.15f;
    t.material.flipParallaxDirection = 0.0f;
    t.material.specularStrength = 0.5f;
    t.material.shininess = 32.0f;

    triples.push_back(t);
    return triples.size() - 1;
}

size_t TextureManager::addTriple(const TextureImage& albedo, VkSampler albedoSampler,
                                  const TextureImage& normal, VkSampler normalSampler,
                                  const TextureImage& height, VkSampler heightSampler) {
    Triple t;
    t.albedo = albedo;
    t.albedoSampler = albedoSampler;
    t.normal = normal;
    t.normalSampler = normalSampler;
    t.height = height;
    t.heightSampler = heightSampler;
    t.ownsResources = false;  // Externally managed resources
    
    // Initialize material properties with reasonable defaults
    t.material.pomHeightScale = 0.06f;
    t.material.pomMinLayers = 8.0f;
    t.material.pomMaxLayers = 32.0f;
    t.material.pomEnabled = 1.0f;
    t.material.flipNormalY = 0.0f;
    t.material.flipTangentHandedness = 0.0f;
    t.material.ambientFactor = 0.15f;
    t.material.flipParallaxDirection = 0.0f;
    t.material.specularStrength = 0.5f;
    t.material.shininess = 32.0f;
    
    triples.push_back(t);
    return triples.size() - 1;
}

void TextureManager::destroyAll() {
    if (!app) return;
    VkDevice device = app->getDevice();
    printf("[TextureManager] destroyAll start: triples=%zu device=%p\n", triples.size(), (void*)device);
    if (device == VK_NULL_HANDLE) {
        // Device already destroyed; can't call Vulkan destroy functions or ImGui removal safely.
        // Just clear our local bookkeeping to avoid double-destruction from destructors later.
        printf("[TextureManager] device is VK_NULL_HANDLE - skipping Vulkan destroys and ImGui removals\n");
        triples.clear();
        return;
    }
    for (size_t i = 0; i < triples.size(); ++i) {
        auto &t = triples[i];
        printf("[TextureManager] triple %zu: ownsResources=%d albedoSampler=%p albedoView=%p albedoImage=%p albedoTexID=%p\n",
               i, t.ownsResources, (void*)t.albedoSampler, (void*)t.albedo.view, (void*)t.albedo.image, (void*)t.albedoTexID);
        // remove ImGui texture handles if created (check that descriptor set is valid)
        if (t.albedoTexID && (VkDescriptorSet)t.albedoTexID != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture((VkDescriptorSet)t.albedoTexID);
            t.albedoTexID = nullptr;
        }
        if (t.normalTexID && (VkDescriptorSet)t.normalTexID != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture((VkDescriptorSet)t.normalTexID);
            t.normalTexID = nullptr;
        }
        if (t.heightTexID && (VkDescriptorSet)t.heightTexID != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture((VkDescriptorSet)t.heightTexID);
            t.heightTexID = nullptr;
        }

        // Only destroy resources we own
        if (t.ownsResources) {
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
    }
    triples.clear();
    printf("[TextureManager] destroyAll done\n");
}

ImTextureID TextureManager::getImTexture(size_t idx, int map) {
    if (idx >= triples.size()) return nullptr;
    Triple &t = triples[idx];
    switch (map) {
        case 0:
            if (!t.albedoTexID) t.albedoTexID = ImGui_ImplVulkan_AddTexture(t.albedoSampler, t.albedo.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            return t.albedoTexID;
        case 1:
            if (!t.normalTexID) t.normalTexID = ImGui_ImplVulkan_AddTexture(t.normalSampler, t.normal.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            return t.normalTexID;
        case 2:
            if (!t.heightTexID) t.heightTexID = ImGui_ImplVulkan_AddTexture(t.heightSampler, t.height.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            return t.heightTexID;
        default:
            return nullptr;
    }
}
