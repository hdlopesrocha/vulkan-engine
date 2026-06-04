#include "StagingRingBuffer.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>

void StagingRingBuffer::init(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size) {
    if (ringBuffer_ != VK_NULL_HANDLE) return;
    device_ = device;
    physicalDevice_ = physicalDevice;
    capacity_ = size;
    head_ = 0;
    tail_ = 0;
    bytesInUse_ = 0;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = capacity_;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device_, &bufferInfo, nullptr, &ringBuffer_) != VK_SUCCESS) {
        throw std::runtime_error("StagingRingBuffer: failed to create ring buffer");
    }

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device_, ringBuffer_, &memReq);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);
    int32_t memTypeIndex = -1;
    VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & required) == required) {
            memTypeIndex = static_cast<int32_t>(i);
            break;
        }
    }
    if (memTypeIndex < 0) {
        vkDestroyBuffer(device_, ringBuffer_, nullptr);
        ringBuffer_ = VK_NULL_HANDLE;
        throw std::runtime_error("StagingRingBuffer: no suitable host-visible coherent memory type");
    }

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = static_cast<uint32_t>(memTypeIndex);
    if (vkAllocateMemory(device_, &allocInfo, nullptr, &ringMemory_) != VK_SUCCESS) {
        vkDestroyBuffer(device_, ringBuffer_, nullptr);
        ringBuffer_ = VK_NULL_HANDLE;
        throw std::runtime_error("StagingRingBuffer: failed to allocate ring memory");
    }
    vkBindBufferMemory(device_, ringBuffer_, ringMemory_, 0);

    if (vkMapMemory(device_, ringMemory_, 0, capacity_, 0, &mappedPtr_) != VK_SUCCESS) {
        vkFreeMemory(device_, ringMemory_, nullptr);
        vkDestroyBuffer(device_, ringBuffer_, nullptr);
        ringMemory_ = VK_NULL_HANDLE;
        ringBuffer_ = VK_NULL_HANDLE;
        throw std::runtime_error("StagingRingBuffer: failed to map ring memory");
    }
    std::cout << "[StagingRingBuffer] initialized with " << (capacity_ / (1024*1024)) << " MiB" << std::endl;
}

void StagingRingBuffer::cleanup(VkDevice device) {
    std::lock_guard<std::mutex> lock(mutex_);
    // vkFreeMemory implicitly unmaps — no explicit vkUnmapMemory needed.
    mappedPtr_ = nullptr;
    if (ringMemory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device, ringMemory_, nullptr);
        ringMemory_ = VK_NULL_HANDLE;
    }
    if (ringBuffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, ringBuffer_, nullptr);
        ringBuffer_ = VK_NULL_HANDLE;
    }
    head_ = 0;
    tail_ = 0;
    bytesInUse_ = 0;
    capacity_ = 0;
    device_ = VK_NULL_HANDLE;
}

StagingRingBuffer::Allocation StagingRingBuffer::allocate(VkDeviceSize size) {
    if (size == 0) return {};
    const VkDeviceSize alignment = 16;
    VkDeviceSize alignedSize = (size + alignment - 1) & ~(alignment - 1);

    std::unique_lock<std::mutex> lock(mutex_);

    // If not enough space, return empty allocation. The caller must retry
    // after pending transfers complete and release their ring buffer regions.
    // Blocking via cv_.wait() would deadlock because release() callbacks are
    // processed in drawFrame(), which requires the main thread to return from
    // update() — but allocate() is called from update().
    if (bytesInUse_ + alignedSize > capacity_) {
        return {};
    }

    // If ring is completely free, reset to beginning
    if (bytesInUse_ == 0) {
        head_ = 0;
        tail_ = 0;
    }

    // Try to fit after head (no wrap)
    if (head_ + alignedSize <= capacity_) {
        Allocation alloc;
        alloc.offset = head_;
        alloc.mappedPtr = static_cast<char*>(mappedPtr_) + head_;
        head_ += alignedSize;
        bytesInUse_ += alignedSize;
        return alloc;
    }

    // Must wrap: only safe if the beginning is free (tail_ has advanced past it).
    // Out-of-order releases can leave tail_ behind, so we must check that the
    // space at the beginning is actually available. If tail_ < alignedSize,
    // the beginning still has in-use data and wrapping would cause corruption.
    if (tail_ >= alignedSize) {
        Allocation alloc;
        alloc.offset = 0;
        alloc.mappedPtr = mappedPtr_;
        head_ = alignedSize;
        bytesInUse_ += alignedSize;
        return alloc;
    }

    // Cannot fit — ring is fragmented. Return empty allocation.
    // Caller must handle this by falling back to a dedicated staging buffer.
    std::cerr << "[StagingRingBuffer] allocation of " << alignedSize
              << " bytes failed: head=" << head_ << " tail=" << tail_
              << " capacity=" << capacity_ << " inUse=" << bytesInUse_ << std::endl;
    return {};
}

void StagingRingBuffer::release(VkDeviceSize offset, VkDeviceSize size) {
    if (size == 0) return;
    const VkDeviceSize alignment = 16;
    VkDeviceSize alignedSize = (size + alignment - 1) & ~(alignment - 1);
    std::lock_guard<std::mutex> lock(mutex_);
    if (alignedSize <= bytesInUse_) {
        bytesInUse_ -= alignedSize;
        // Advance tail past contiguous freed regions at the front
        if (offset == tail_) {
            tail_ += alignedSize;
        }
        if (bytesInUse_ == 0) {
            head_ = 0;
            tail_ = 0;
        }
    }
    cv_.notify_one();
}
