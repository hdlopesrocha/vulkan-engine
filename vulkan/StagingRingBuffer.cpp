#include "StagingRingBuffer.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <algorithm>

void StagingRingBuffer::init(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size) {
    if (ringBuffer_ != VK_NULL_HANDLE) return;
    device_ = device;
    physicalDevice_ = physicalDevice;
    capacity_ = size;
    bytesInUse_ = 0;
    freeList_.clear();
    freeList_.push_back({0, capacity_});

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
    mappedPtr_ = nullptr;
    if (ringMemory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device, ringMemory_, nullptr);
        ringMemory_ = VK_NULL_HANDLE;
    }
    if (ringBuffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, ringBuffer_, nullptr);
        ringBuffer_ = VK_NULL_HANDLE;
    }
    bytesInUse_ = 0;
    capacity_ = 0;
    freeList_.clear();
    device_ = VK_NULL_HANDLE;
}

StagingRingBuffer::Allocation StagingRingBuffer::allocate(VkDeviceSize size) {
    if (size == 0) return {};
    const VkDeviceSize alignment = 16;
    VkDeviceSize alignedSize = (size + alignment - 1) & ~(alignment - 1);

    std::unique_lock<std::mutex> lock(mutex_);

    if (bytesInUse_ + alignedSize > capacity_) {
        return {};
    }

    // First-fit: find the first free block large enough
    for (auto it = freeList_.begin(); it != freeList_.end(); ++it) {
        if (it->size >= alignedSize) {
            Allocation alloc;
            alloc.offset = it->offset;
            alloc.mappedPtr = static_cast<char*>(mappedPtr_) + it->offset;

            if (it->size == alignedSize) {
                freeList_.erase(it);
            } else {
                it->offset += alignedSize;
                it->size -= alignedSize;
            }
            bytesInUse_ += alignedSize;
            return alloc;
        }
    }

    // No contiguous free region large enough
    std::cerr << "[StagingRingBuffer] allocation of " << alignedSize
              << " bytes failed: capacity=" << capacity_ << " inUse=" << bytesInUse_
              << " freeBlocks=" << freeList_.size() << std::endl;
    return {};
}

void StagingRingBuffer::release(VkDeviceSize offset, VkDeviceSize size) {
    if (size == 0) return;
    const VkDeviceSize alignment = 16;
    VkDeviceSize alignedSize = (size + alignment - 1) & ~(alignment - 1);
    std::lock_guard<std::mutex> lock(mutex_);

    if (alignedSize > bytesInUse_) {
        std::cerr << "[StagingRingBuffer] release underflow: offset=" << offset
                  << " alignedSize=" << alignedSize << " bytesInUse=" << bytesInUse_ << std::endl;
        return;
    }
    bytesInUse_ -= alignedSize;

    // Insert into free list, merge with adjacent blocks, keep sorted by offset.
    auto it = freeList_.begin();
    while (it != freeList_.end() && it->offset < offset) ++it;

    // Check merge with previous block
    bool mergedPrev = false;
    if (it != freeList_.begin()) {
        auto prev = it - 1;
        if (prev->offset + prev->size == offset) {
            prev->size += alignedSize;
            offset = prev->offset;
            alignedSize = prev->size;
            mergedPrev = true;
        }
    }

    // Check merge with next block
    bool mergedNext = false;
    if (it != freeList_.end() && offset + alignedSize == it->offset) {
        if (mergedPrev) {
            auto prev = it - 1;
            prev->size += it->size;
            freeList_.erase(it);
        } else {
            it->offset = offset;
            it->size += alignedSize;
            mergedNext = true;
        }
    }

    if (!mergedPrev && !mergedNext) {
        freeList_.insert(it, {offset, alignedSize});
    }

    // If completely empty, coalesce into a single block for cleanliness.
    if (bytesInUse_ == 0) {
        freeList_.clear();
        freeList_.push_back({0, capacity_});
    }

    cv_.notify_one();
}