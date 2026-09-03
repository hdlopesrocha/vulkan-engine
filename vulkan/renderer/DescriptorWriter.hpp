#pragma once

#include <vulkan/vulkan.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

// Render-loop vkUpdateDescriptorSets accounting.
//
// Validation for the descriptor-buffer migration: the render loop must issue
// zero vkUpdateDescriptorSets calls in steady state (all static bindings are
// written once at init / on resource change; per-frame UBO contents are
// streamed via memcpy / vkCmdCopyBuffer into the already-bound buffers).
// Every DescriptorWriter::flush() records its write count here so a CPU
// profile (or a debug snapshot around drawFrame) can assert the delta is 0.
struct DescriptorUpdateStats {
    static inline std::atomic<uint64_t> s_updateCalls{0};
    static void noteUpdate(uint64_t n = 1) {
        s_updateCalls.fetch_add(n, std::memory_order_relaxed);
    }
    static uint64_t updateCalls() {
        return s_updateCalls.load(std::memory_order_relaxed);
    }
    static void reset() { s_updateCalls.store(0, std::memory_order_relaxed); }
};

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
        DescriptorUpdateStats::noteUpdate(writes_.size());
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

// ── VK_EXT_descriptor_buffer write path ─────────────────────────────────────
// GPU-side alternative to DescriptorWriter::flush(): writes a single descriptor
// directly into host-visible descriptor-buffer memory via vkGetDescriptorEXT —
// no vkUpdateDescriptorSets call, so no CPU-side descriptor-set validation or
// pool bookkeeping on the update path. Per-frame resources (UBOs, dynamic
// images) are then streamed with a host memcpy (host-visible descriptor buffer)
// or vkCmdCopyBuffer (device-local staging -> descriptor buffer), which the GPU
// timeline can overlap with compute.
//
// Usage requires:
//   * VulkanApp::useDescriptorBuffer() == true (extension + feature enabled),
//   * the source VkBuffer created with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
//   * dst pointing at descriptor-buffer memory with at least dataSize bytes free
//     (sizes come from VkPhysicalDeviceDescriptorBufferPropertiesEXT or
//     vkGetDescriptorSetLayoutBindingOffsetEXT strides).
// Every helper returns false when the fast path cannot be used; callers must
// fall back to DescriptorWriter so devices without the extension keep working.
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

    static VkDescriptorBufferBindingInfoEXT bindingInfo(VkDeviceAddress address,
                                                        VkBufferUsageFlags usage) {
        VkDescriptorBufferBindingInfoEXT info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
        info.address = address;
        info.usage = usage;
        return info;
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
