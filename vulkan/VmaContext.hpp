#pragma once
#include <vulkan/vulkan.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "../third_party/VulkanMemoryAllocator/include/vk_mem_alloc.h"
#pragma GCC diagnostic pop

class VmaContext {
public:
    VmaAllocator allocator = VK_NULL_HANDLE;

    void init(VkInstance instance, VkPhysicalDevice physDev, VkDevice dev) {
        VmaVulkanFunctions vf{};
        vf.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
        vf.vkGetDeviceProcAddr = &vkGetDeviceProcAddr;
        vf.vkGetPhysicalDeviceProperties = &vkGetPhysicalDeviceProperties;
        vf.vkGetPhysicalDeviceMemoryProperties = &vkGetPhysicalDeviceMemoryProperties;
        vf.vkAllocateMemory = &vkAllocateMemory;
        vf.vkFreeMemory = &vkFreeMemory;
        vf.vkMapMemory = &vkMapMemory;
        vf.vkUnmapMemory = &vkUnmapMemory;
        vf.vkFlushMappedMemoryRanges = &vkFlushMappedMemoryRanges;
        vf.vkInvalidateMappedMemoryRanges = &vkInvalidateMappedMemoryRanges;
        vf.vkBindBufferMemory = &vkBindBufferMemory;
        vf.vkBindImageMemory = &vkBindImageMemory;
        vf.vkGetBufferMemoryRequirements = &vkGetBufferMemoryRequirements;
        vf.vkGetImageMemoryRequirements = &vkGetImageMemoryRequirements;
        vf.vkCreateBuffer = &vkCreateBuffer;
        vf.vkDestroyBuffer = &vkDestroyBuffer;
        vf.vkCreateImage = &vkCreateImage;
        vf.vkDestroyImage = &vkDestroyImage;
        vf.vkCmdCopyBuffer = &vkCmdCopyBuffer;

        VmaAllocatorCreateInfo ci{};
        ci.vulkanApiVersion = VK_API_VERSION_1_4;
        ci.physicalDevice = physDev;
        ci.device = dev;
        ci.instance = instance;
        ci.pVulkanFunctions = &vf;
        // Allow creating buffers with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
        // (required for VK_EXT_descriptor_buffer source buffers and GPU-assisted
        // validation). Harmless when the device feature is disabled — only the
        // buffers that request the usage bit depend on it.
        ci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

        vmaCreateAllocator(&ci, &allocator);
    }

    void destroy() {
        if (allocator) { vmaDestroyAllocator(allocator); allocator = VK_NULL_HANDLE; }
    }
};
