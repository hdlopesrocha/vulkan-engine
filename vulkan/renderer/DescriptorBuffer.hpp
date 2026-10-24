#pragma once

#include <vulkan/vulkan.h>
#include <cstddef>

#include "DescriptorBufferHelper.hpp"

// Non-owning view of one host-visible descriptor buffer (set 0).
//
// Owns no Vulkan objects: the backing VkBuffer is created by the caller with
// VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
// VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT and host-visible + coherent memory.
// This wrapper only computes aligned binding offsets (via
// vkGetDescriptorSetLayoutBindingOffsetEXT when available) and forwards writes
// to DescriptorBufferHelper, so per-frame updates are plain host memcpys.
class DescriptorBuffer {
public:
    DescriptorBuffer() = default;
    DescriptorBuffer(VkDevice device,
                     PFN_vkGetDescriptorEXT fpGetDescriptor,
                     PFN_vkGetDescriptorSetLayoutBindingOffsetEXT fpBindingOffset,
                     void* mapped, size_t size, size_t alignment)
        : device_(device), fpGet_(fpGetDescriptor), fpOffset_(fpBindingOffset),
          mapped_(static_cast<char*>(mapped)), size_(size), alignment_(alignment) {}

    bool valid() const { return device_ != VK_NULL_HANDLE && mapped_ != nullptr && fpGet_ != nullptr; }
    void* at(size_t byteOffset) const {
        if (!mapped_ || byteOffset >= size_) return nullptr;
        return mapped_ + byteOffset;
    }
    size_t size() const { return size_; }

    // Binding byte offset inside a descriptor-buffer layout. Falls back to a
    // packed running offset (caller-provided) when the offset query entry point
    // is unavailable — keeps the fallback path functional on old drivers.
    VkDeviceSize bindingOffset(VkDescriptorSetLayout layout, uint32_t binding,
                               VkDeviceSize fallback) const {
        if (fpOffset_ && layout != VK_NULL_HANDLE) {
            VkDeviceSize off = 0;
            fpOffset_(device_, layout, binding, &off);
            return off;
        }
        return fallback;
    }

    bool writeBuffer(VkDeviceSize dstOffset, size_t dataSize, VkDescriptorType type,
                     VkBuffer srcBuffer, VkDeviceSize srcOffset, VkDeviceSize range) {
        void* dst = at(static_cast<size_t>(dstOffset));
        if (!dst) return false;
        const VkDeviceAddress base = DescriptorBufferHelper::bufferAddress(device_, srcBuffer);
        return DescriptorBufferHelper::writeBufferDescriptor(device_, fpGet_, dst, dataSize, type,
                                                             base, srcOffset, range);
    }

    bool writeImage(VkDeviceSize dstOffset, size_t dataSize, VkDescriptorType type,
                    VkSampler sampler, VkImageView view, VkImageLayout layout) {
        void* dst = at(static_cast<size_t>(dstOffset));
        if (!dst) return false;
        return DescriptorBufferHelper::writeImageDescriptor(device_, fpGet_, dst, dataSize, type,
                                                            sampler, view, layout);
    }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    PFN_vkGetDescriptorEXT fpGet_ = nullptr;
    PFN_vkGetDescriptorSetLayoutBindingOffsetEXT fpOffset_ = nullptr;
    char* mapped_ = nullptr;
    size_t size_ = 0;
    size_t alignment_ = 1;
};
