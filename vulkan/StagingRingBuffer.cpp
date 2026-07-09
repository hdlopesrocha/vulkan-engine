#include "StagingRingBuffer.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <algorithm>

void StagingRingBuffer::init(VmaAllocator vmaAllocator, VkDeviceSize size) {
    if (ringBuffer_ != VK_NULL_HANDLE) return;
    vmaAlloc_ = vmaAllocator;
    capacity_ = size;
    bytesInUse_ = 0;
    freeList_.clear();
    freeList_.push_back({0, capacity_});

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = capacity_;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO;
    allocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                  | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo allocInfo;
    if (vmaCreateBuffer(vmaAlloc_, &bufferInfo, &allocCI, &ringBuffer_, &ringAllocation_, &allocInfo) != VK_SUCCESS)
        throw std::runtime_error("StagingRingBuffer: failed to create ring buffer with VMA");

    mappedPtr_ = allocInfo.pMappedData;
    std::cout << "[StagingRingBuffer] initialized with " << (capacity_ / (1024*1024)) << " MiB (VMA)" << std::endl;
}

void StagingRingBuffer::cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    mappedPtr_ = nullptr;
    if (ringBuffer_ != VK_NULL_HANDLE && vmaAlloc_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(vmaAlloc_, ringBuffer_, ringAllocation_);
        ringBuffer_ = VK_NULL_HANDLE;
        ringAllocation_ = VK_NULL_HANDLE;
    }
    bytesInUse_ = 0;
    capacity_ = 0;
    freeList_.clear();
}

StagingRingBuffer::Allocation StagingRingBuffer::allocate(VkDeviceSize size) {
    if (size == 0) return {};
    const VkDeviceSize alignment = 16;
    VkDeviceSize alignedSize = (size + alignment - 1) & ~(alignment - 1);

    std::unique_lock<std::mutex> lock(mutex_);

    if (bytesInUse_ + alignedSize > capacity_) {
        return {};
    }

    for (auto it = freeList_.begin(); it != freeList_.end(); ++it) {
        if (it->size >= alignedSize) {
            Allocation alloc;
            alloc.offset = it->offset;
            alloc.size   = alignedSize;
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

    std::cerr << "[StagingRingBuffer] allocation of " << alignedSize
              << " bytes failed: capacity=" << capacity_ << " inUse=" << bytesInUse_
              << " freeBlocks=" << freeList_.size() << std::endl;
    return {};
}

void StagingRingBuffer::release(Allocation& alloc) {
    if (alloc.released) return;
    if (alloc.size == 0) return;
    VkDeviceSize offset = alloc.offset;
    VkDeviceSize alignedSize = alloc.size;
    std::lock_guard<std::mutex> lock(mutex_);

    if (alignedSize > bytesInUse_) {
        std::cerr << "[StagingRingBuffer] release underflow: offset=" << offset
                  << " alignedSize=" << alignedSize << " bytesInUse=" << bytesInUse_ << std::endl;
        return;
    }
    bytesInUse_ -= alignedSize;

    auto it = freeList_.begin();
    while (it != freeList_.end() && it->offset < offset) ++it;

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

    if (bytesInUse_ == 0) {
        freeList_.clear();
        freeList_.push_back({0, capacity_});
    }

    alloc.released = true;
    cv_.notify_one();
}
