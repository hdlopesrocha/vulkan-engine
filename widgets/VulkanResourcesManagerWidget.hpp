#pragma once

#include "Widget.hpp"
#include "../vulkan/VulkanResourceManager.hpp"
#include <memory>
#include <array>
#include <cstdint>

class VulkanApp;

class VulkanResourcesManagerWidget : public Widget {
public:
    VulkanResourcesManagerWidget(VulkanResourceManager* mgr_);
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

    // --- Per-queue activity chart ---
    // One rolling history of in-flight (pending) command buffers per logical queue.
    // Keyed by logical slot; if two slots alias the same VkQueue handle (e.g. only
    // one graphics queue exposed), they share the same underlying hardware queue and
    // will show identical values.
    static constexpr int QUEUE_HISTORY = 240;
    enum QueueId : int { Q_GRAPHICS = 0, Q_PRESENT, Q_VEGETATION, Q_SDF, Q_BBOX, Q_GEOMETRY, Q_TRANSFER, Q_COUNT };
    VkQueue cachedQueue[Q_COUNT] = {};
    std::array<float, QUEUE_HISTORY> queueHistory[Q_COUNT];
    bool queueActive[Q_COUNT] = {};   // handle non-null this frame
    int queuePendingNow[Q_COUNT] = {};
    uint64_t queueSubmittedTotal[Q_COUNT] = {};
    uint64_t queueCompletedTotal[Q_COUNT] = {};
};
