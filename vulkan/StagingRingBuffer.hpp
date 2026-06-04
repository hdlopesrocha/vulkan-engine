#pragma once

#include <vulkan/vulkan.h>
#include <mutex>
#include <condition_variable>
#include <cstddef>

class StagingRingBuffer {
public:
    struct Allocation {
        VkDeviceSize offset = 0;
        void*        mappedPtr = nullptr;
    };

    StagingRingBuffer() = default;
    ~StagingRingBuffer() = default;

    void init(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size = 4 * 1024 * 1024);
    void cleanup(VkDevice device);
    Allocation allocate(VkDeviceSize size);
    void release(VkDeviceSize offset, VkDeviceSize size);

    VkBuffer buffer() const { return ringBuffer_; }

private:
    VkDevice         device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkBuffer         ringBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory   ringMemory_ = VK_NULL_HANDLE;
    void*            mappedPtr_ = nullptr;
    VkDeviceSize     capacity_ = 0;
    VkDeviceSize     head_ = 0;
    VkDeviceSize     tail_ = 0;
    VkDeviceSize     bytesInUse_ = 0;
    std::mutex mutex_;
    std::condition_variable cv_;
};
