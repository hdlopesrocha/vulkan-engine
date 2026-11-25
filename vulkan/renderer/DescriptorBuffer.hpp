#pragma once

#include <vulkan/vulkan.h>
#include <cstddef>

#include "DescriptorBufferHelper.hpp"

// Non-owning view of one host-visible descriptor buffer (set 0).
//
// Owns no Vulkan objects: the backing VkBuffer is created by the caller with
// VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
// VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT and host-visible + coherent memory.
// Binding offsets are resolved by the caller via
// vkGetDescriptorSetLayoutBindingOffsetEXT; this wrapper only forwards the
// writes to DescriptorBufferHelper, so per-frame updates are plain host
// memcpys.
class DescriptorBuffer {
public:
    DescriptorBuffer() = default;
    DescriptorBuffer(VkDevice device,
                     PFN_vkGetDescriptorEXT fpGetDescriptor,
                     void* mapped, size_t size)
        : device_(device), fpGet_(fpGetDescriptor),
          mapped_(static_cast<char*>(mapped)), size_(size) {}

    bool valid() const { return device_ != VK_NULL_HANDLE && mapped_ != nullptr && fpGet_ != nullptr; }
    void* at(size_t byteOffset) const {
        if (!mapped_ || byteOffset >= size_) return nullptr;
        return mapped_ + byteOffset;
    }
    size_t size() const { return size_; }

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
    char* mapped_ = nullptr;
    size_t size_ = 0;
};
