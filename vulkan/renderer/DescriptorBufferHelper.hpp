#pragma once

#include <vulkan/vulkan.h>
#include <cstddef>

// Stateless helpers for host-side descriptor-buffer writes.
struct DescriptorBufferHelper {
    static size_t alignUp(size_t v, size_t a) {
        if (a == 0) return v;
        return (v + a - 1) & ~(a - 1);
    }

    // Device address of a buffer (0 when the buffer lacks
    // VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT).
    static VkDeviceAddress bufferAddress(VkDevice device, VkBuffer buffer) {
        VkBufferDeviceAddressInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        info.buffer = buffer;
        return vkGetBufferDeviceAddress(device, &info);
    }

    // Write one buffer descriptor (uniform / storage) into dst[0, dataSize).
    static bool writeBufferDescriptor(VkDevice device, PFN_vkGetDescriptorEXT fpGetDescriptor,
                                      void* dst, size_t dataSize,
                                      VkDescriptorType type,
                                      VkDeviceAddress baseAddress,
                                      VkDeviceSize offset, VkDeviceSize range) {
        if (!fpGetDescriptor || !dst || device == VK_NULL_HANDLE || baseAddress == 0) return false;
        if (type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
            type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) return false;
        VkDescriptorAddressInfoEXT addr{};
        addr.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT;
        addr.address = baseAddress + offset;
        addr.range = range;
        addr.format = VK_FORMAT_UNDEFINED;
        VkDescriptorGetInfoEXT get{};
        get.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
        get.type = type;
        if (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) get.data.pUniformBuffer = &addr;
        else get.data.pStorageBuffer = &addr;
        fpGetDescriptor(device, &get, dataSize, dst);
        return true;
    }

    // Write one image descriptor (combined / sampled / storage / input) into
    // dst[0, dataSize).
    static bool writeImageDescriptor(VkDevice device, PFN_vkGetDescriptorEXT fpGetDescriptor,
                                     void* dst, size_t dataSize,
                                     VkDescriptorType type,
                                     VkSampler sampler, VkImageView view,
                                     VkImageLayout layout) {
        if (!fpGetDescriptor || !dst || device == VK_NULL_HANDLE) return false;
        if (view == VK_NULL_HANDLE) return false;
        if (type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
            type != VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE &&
            type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE &&
            type != VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT) return false;
        VkDescriptorImageInfo img{};
        img.sampler = sampler;
        img.imageView = view;
        img.imageLayout = layout;
        VkDescriptorGetInfoEXT get{};
        get.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
        get.type = type;
        if (type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) get.data.pCombinedImageSampler = &img;
        else if (type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) get.data.pSampledImage = &img;
        else if (type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) get.data.pStorageImage = &img;
        else get.data.pInputAttachmentImage = &img;
        fpGetDescriptor(device, &get, dataSize, dst);
        return true;
    }
};
