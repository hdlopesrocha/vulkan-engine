#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>
#include <glm/glm.hpp>
#include <mutex>
#include "OctreeMeshTypes.hpp"

class VulkanApp;

class OctreeMeshCompute {
public:
    static OctreeMeshCompute* globalInstance __attribute__((weak));
    static OctreeMeshCompute* getGlobal() __attribute__((weak)) { return globalInstance; }
    OctreeMeshCompute();
    void init(VulkanApp* appInstance);
    void destroy(VulkanApp* appInstance);
    VulkanApp* app = nullptr;
    std::mutex mutex;
    // Synchronously compute vertices for the given cells. Returns positions/normals.
    OctreeMeshResult compute(const std::vector<OctreeMeshCell>& cells, float isoValue = 0.0f) __attribute__((weak));
private:
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;
};