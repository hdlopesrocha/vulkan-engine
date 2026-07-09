#pragma once

#include <vulkan/vulkan.h>
#include "../third_party/VulkanMemoryAllocator/include/vk_mem_alloc.h"
#include <mutex>
#include <condition_variable>
#include <vector>
#include <cstddef>

class StagingRingBuffer {
public:
    struct Allocation {
        VkDeviceSize offset = 0;
        VkDeviceSize size   = 0;
        void*        mappedPtr = nullptr;
        bool         released = false;
    };

    StagingRingBuffer() = default;
    ~StagingRingBuffer() = default;

    void init(VmaAllocator vmaAllocator, VkDeviceSize size = 4 * 1024 * 1024);
    void cleanup();
    Allocation allocate(VkDeviceSize size);
    void release(Allocation& alloc);

    VkBuffer buffer() const { return ringBuffer_; }

private:
    struct FreeBlock {
        VkDeviceSize offset;
        VkDeviceSize size;
    };

    VmaAllocator     vmaAlloc_ = VK_NULL_HANDLE;
    VkBuffer         ringBuffer_ = VK_NULL_HANDLE;
    VmaAllocation    ringAllocation_ = VK_NULL_HANDLE;
    void*            mappedPtr_ = nullptr;
    VkDeviceSize     capacity_ = 0;
    VkDeviceSize     bytesInUse_ = 0;
    std::vector<FreeBlock> freeList_;
    std::mutex mutex_;
    std::condition_variable cv_;
};
