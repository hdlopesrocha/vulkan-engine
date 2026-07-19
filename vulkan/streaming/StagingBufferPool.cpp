#include "StagingBufferPool.hpp"
#include <stdexcept>
#include <iostream>

namespace streaming {

void StagingBufferPool::createSlot(StagingSlot& s, VkDevice device,
                                  uint32_t queueFamily, VkDeviceSize slotSize) {
    // Host-visible, sequentially-written, persistently-mapped staging buffer.
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size  = slotSize;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode           = VK_SHARING_MODE_EXCLUSIVE;
    bi.queueFamilyIndexCount = 1;
    bi.pQueueFamilyIndices   = &queueFamily;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
              | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo info{};
    if (vmaCreateBuffer(vma_, &bi, &aci, &s.buffer, &s.alloc, &info) != VK_SUCCESS)
        throw std::runtime_error("StagingBufferPool: failed to create staging buffer");
    s.mapped = info.pMappedData;
    s.size   = slotSize;

    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = cmdPool_;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &ai, &s.cmd) != VK_SUCCESS)
        throw std::runtime_error("StagingBufferPool: vkAllocateCommandBuffers failed");

    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(device, &fi, nullptr, &s.fence) != VK_SUCCESS)
        throw std::runtime_error("StagingBufferPool: vkCreateFence failed");

    // NOTE: signalSem is intentionally NOT created here. In the timeline path the
    // manager signals one shared timeline (safe to reuse); in the fallback path a
    // FRESH binary semaphore is created per submission and destroyed on recycle
    // (reusing one binary semaphore across submissions would stay permanently
    // signaled and break cross-frame hazard tracking).
}

void StagingBufferPool::init(VmaAllocator vma, VkDevice device,
                             uint32_t queueFamily, uint32_t slotCount,
                             VkDeviceSize slotSize) {
    vma_ = vma;
    device_ = device;

    VkCommandPoolCreateInfo pi{};
    pi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pi.queueFamilyIndex = queueFamily;
    // RESET_COMMAND_BUFFER_BIT: we reset+reuse the same command buffer every
    // submission instead of allocating a new one (avoids per-upload churn).
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device, &pi, nullptr, &cmdPool_) != VK_SUCCESS)
        throw std::runtime_error("StagingBufferPool: vkCreateCommandPool failed");

    slots_.resize(slotCount);
    for (auto& s : slots_) {
        createSlot(s, device, queueFamily, slotSize);
    }
    std::cout << "[StagingBufferPool] " << slotCount << " slots x "
              << (slotSize / (1024 * 1024)) << " MiB staging (" << (slotCount * slotSize / (1024 * 1024))
              << " MiB total)\n";
}

StagingSlot* StagingBufferPool::acquireFree() {
    for (auto& s : slots_) {
        if (!s.busy) return &s;
    }
    return nullptr;
}

void StagingBufferPool::reset(StagingSlot& s) {
    vkResetFences(device_, 1, &s.fence);
    vkResetCommandBuffer(s.cmd, 0);
    s.busy         = false;
    s.waitRegistered = false;
    s.timelineValue = 0;
    s.onComplete   = nullptr;
    s.chunkSlot    = nullptr;
    // signalSem lifecycle is owned by the manager (fresh per submission in the
    // fallback path), so we only clear the handle here.
    s.signalSem    = VK_NULL_HANDLE;
}

void StagingBufferPool::destroy() {
    for (auto& s : slots_) {
        if (s.fence)    vkDestroyFence(device_, s.fence, nullptr);
        if (s.signalSem) vkDestroySemaphore(device_, s.signalSem, nullptr);
        if (s.cmd)      vkFreeCommandBuffers(device_, cmdPool_, 1, &s.cmd);
        if (s.alloc)    vmaDestroyBuffer(vma_, s.buffer, s.alloc);
    }
    slots_.clear();
    if (cmdPool_) vkDestroyCommandPool(device_, cmdPool_, nullptr);
    cmdPool_ = VK_NULL_HANDLE;
}

} // namespace streaming
