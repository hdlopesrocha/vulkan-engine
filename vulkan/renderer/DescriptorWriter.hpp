#pragma once

#include <vulkan/vulkan.h>
#include <deque>
#include <vector>
#include <cstdint>

// Fluent builder for VkWriteDescriptorSet updates.
// Accumulates writes and descriptor infos, then flushes them in one
// vkUpdateDescriptorSets call.  Internal storage uses std::deque so
// pointers to infos remain valid after subsequent calls.
//
// Usage:
//
//   DescriptorWriter(device)
//       .writeBuffer(descriptorSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
//                    buffer, 0, VK_WHOLE_SIZE)
//       .writeImage(descriptorSet, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
//                   sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
//       .flush();

class DescriptorWriter {
    VkDevice device_;
    std::vector<VkWriteDescriptorSet> writes_;
    std::deque<VkDescriptorBufferInfo> bufferInfos_;
    std::deque<VkDescriptorImageInfo> imageInfos_;

public:
    explicit DescriptorWriter(VkDevice device) : device_(device) {}

    DescriptorWriter& writeBuffer(VkDescriptorSet dst, uint32_t binding,
                                  VkDescriptorType type,
                                  VkBuffer buffer, VkDeviceSize offset,
                                  VkDeviceSize range, uint32_t count = 1)
    {
        bufferInfos_.push_back({buffer, offset, range});
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = dst;
        w.dstBinding = binding;
        w.descriptorType = type;
        w.descriptorCount = count;
        w.pBufferInfo = &bufferInfos_.back();
        writes_.push_back(w);
        return *this;
    }

    DescriptorWriter& writeImage(VkDescriptorSet dst, uint32_t binding,
                                 VkDescriptorType type,
                                 VkSampler sampler, VkImageView view,
                                 VkImageLayout layout, uint32_t count = 1)
    {
        imageInfos_.push_back({sampler, view, layout});
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = dst;
        w.dstBinding = binding;
        w.descriptorType = type;
        w.descriptorCount = count;
        w.pImageInfo = &imageInfos_.back();
        writes_.push_back(w);
        return *this;
    }

    void flush()
    {
        if (writes_.empty()) return;
        vkUpdateDescriptorSets(device_,
                               static_cast<uint32_t>(writes_.size()),
                               writes_.data(), 0, nullptr);
        clear();
    }

    // Access accumulated writes for custom processing before flush.
    const std::vector<VkWriteDescriptorSet>& writes() const { return writes_; }
    std::vector<VkWriteDescriptorSet>& writes() { return writes_; }

    void clear()
    {
        writes_.clear();
        bufferInfos_.clear();
        imageInfos_.clear();
    }
};
