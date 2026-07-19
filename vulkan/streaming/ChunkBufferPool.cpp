#include "ChunkBufferPool.hpp"
#include "../VulkanApp.hpp"
#include <stdexcept>
#include <iostream>

namespace streaming {

void ChunkBufferPool::init(VulkanApp* app,
                           VkDeviceSize chunkVertexBytes,
                           VkDeviceSize chunkIndexBytes,
                           uint32_t initialSlots) {
    app_ = app;
    vma_ = app->getVmaAllocator();
    vCap_ = chunkVertexBytes;
    iCap_ = chunkIndexBytes;

    auto qfi = app->findQueueFamilies(app->getPhysicalDevice());
    gfxFamily_ = qfi.graphicsFamily.value();

    slots_.reserve(initialSlots);
    free_.reserve(initialSlots);
    for (uint32_t i = 0; i < initialSlots; ++i) grow();
}

void ChunkBufferPool::grow() {
    auto makeBuf = [&](VkDeviceSize size, VkBufferUsageFlags usage) -> Buffer {
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size  = size;
        bi.usage = usage;
        bi.sharingMode           = VK_SHARING_MODE_EXCLUSIVE;
        bi.queueFamilyIndexCount = 1;
        bi.pQueueFamilyIndices   = &gfxFamily_;

        VmaAllocationCreateInfo aci{};
        aci.usage          = VMA_MEMORY_USAGE_AUTO;
        aci.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        Buffer b{};
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(vma_, &bi, &aci, &b.buffer, &b.allocation, &info) != VK_SUCCESS)
            throw std::runtime_error("ChunkBufferPool: vmaCreateBuffer failed");
        b.mappedData = info.pMappedData;
        return b;
    };

    auto slot = std::make_unique<ChunkGPUBuffers>();
    slot->vertexBuffer   = makeBuf(vCap_, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    slot->indexBuffer    = makeBuf(iCap_, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    slot->vertexCapacity = vCap_;
    slot->indexCapacity  = iCap_;
    free_.push_back(slot.get());
    slots_.push_back(std::move(slot));
}

ChunkGPUBuffers* ChunkBufferPool::acquire(uint64_t chunkId) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (free_.empty()) {
        // Grow on demand. This is a main-thread Vulkan allocation, never reached
        // from a worker, so it stays off the async generation path.
        grow();
    }
    ChunkGPUBuffers* s = free_.back();
    free_.pop_back();
    s->inUse      = true;
    s->ownerChunk = chunkId;
    return s;
}

void ChunkBufferPool::release(ChunkGPUBuffers* slot) {
    if (!slot) return;
    std::lock_guard<std::mutex> lk(mtx_);
    slot->inUse = false;
    free_.push_back(slot);
}

void ChunkBufferPool::destroy() {
    // The manager must have released every slot first. Buffers are owned by the
    // unique_ptr vector; free the VMA allocations here.
    for (auto& s : slots_) {
        if (s->vertexBuffer.allocation)
            vmaDestroyBuffer(vma_, s->vertexBuffer.buffer, s->vertexBuffer.allocation);
        if (s->indexBuffer.allocation)
            vmaDestroyBuffer(vma_, s->indexBuffer.buffer, s->indexBuffer.allocation);
    }
    slots_.clear();
    free_.clear();
}

} // namespace streaming
