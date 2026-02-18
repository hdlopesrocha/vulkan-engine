#pragma once

#include "Widget.hpp"
#include "../vulkan/VulkanResourceManager.hpp"
#include <memory>

class VulkanApp;

class VulkanResourcesManagerWidget : public Widget {
public:
    VulkanResourcesManagerWidget(VulkanResourceManager* mgr);
    // Per-frame refresh (does NOT store VulkanApp* persistently)
    void updateWithApp(class VulkanApp* app);
    void render() override;

private:
    VulkanResourceManager* mgr;
    // Cached app-provided values (no stored VulkanApp*)
    VkInstance cachedInstance = VK_NULL_HANDLE;
    VkPhysicalDevice cachedPhysicalDevice = VK_NULL_HANDLE;
    VkDevice cachedDevice = VK_NULL_HANDLE;
    VkQueue cachedGraphicsQueue = VK_NULL_HANDLE;
    VkQueue cachedPresentQueue = VK_NULL_HANDLE;
    VkSwapchainKHR cachedSwapchain = VK_NULL_HANDLE;
    VkFormat cachedSwapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D cachedSwapchainExtent = {0,0};
    std::vector<VkDescriptorSet> cachedRegisteredDescriptorSets;
    std::vector<VkPipeline> cachedRegisteredPipelines;
    VkDescriptorPool cachedDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorPool cachedImGuiDescriptorPool = VK_NULL_HANDLE;
    bool hasAppCache = false;
    bool showHex = true;
};
