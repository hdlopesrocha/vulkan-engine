#pragma once

#include "../VulkanApp.hpp"
#include "../Buffer.hpp"
#include "RtDispatch.hpp"
#include <vulkan/vulkan.h>
#include <vector>

// Thin accessor around the ray-tracing-capable VulkanApp. Centralizes the device
// handle, the VMA allocator, and the resource registry so the acceleration-
// structure and SBT helpers don't each reach into VulkanApp internals. The RT
// feature flags live on VulkanApp::rtSupport; query them before building AS/RT
// pipelines so the renderer can gracefully fall back to rasterization.
class RayTracingContext {
public:
    VulkanApp* app = nullptr;
    RtDispatch dispatch;

    void init(VulkanApp* app_) {
        app = app_;
        dispatch = app_->rtDispatch;
    }

    bool supported() const { return app && app->rtSupport.any() && dispatch.ready(); }

    VkDevice device() const { return app->getDevice(); }
    VmaAllocator allocator() const { return app->getVmaAllocator(); }
    VulkanApp::RayTracingSupport rt() const { return app->rtSupport; }

    // Allocate a device-local, device-addressable buffer for AS storage, scratch,
    // or SBT. Registered with the app's resource manager so it is destroyed in
    // the correct order at shutdown. zeroInit=false avoids an extra GPU fill —
    // callers fully overwrite these buffers before first use.
    Buffer createRtBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
        return app->createBuffer(size,
            usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, /*zeroInit=*/false);
    }

    void destroyRtBuffer(Buffer& buf) { app->destroyBuffer(buf); }

    // Query acceleration-structure build sizes for a given build info.
    // `maxPrimitiveCounts` must have one entry per geometry in buildInfo.
    void getBuildSizes(const VkAccelerationStructureBuildGeometryInfoKHR& buildInfo,
                       const uint32_t* maxPrimitiveCounts,
                       VkAccelerationStructureBuildSizesInfoKHR& outSizes) const {
        outSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        outSizes.pNext = nullptr;
        dispatch.vkGetAccelerationStructureBuildSizesKHR(device(),
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &buildInfo, maxPrimitiveCounts, &outSizes);
    }
};
