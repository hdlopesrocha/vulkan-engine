#include "ShaderBindingTable.hpp"

void ShaderBindingTable::build() {
    if (raygenGroups_.empty())
        throw std::runtime_error("ShaderBindingTable::build: no raygen group set");

    const uint32_t stride = entryStride();
    const uint32_t handleStorage = handleSize_; // raw handle size (<= stride)

    // Compute region offsets (concatenated; raygen first so its base address is
    // the buffer base — which VMA aligns to the device's base alignment).
    raygenOffset_   = 0;
    missOffset_     = raygenOffset_   + stride * (uint32_t)raygenGroups_.size();
    hitOffset_      = missOffset_     + stride * (uint32_t)missGroups_.size();
    callableOffset_ = hitOffset_      + stride * (uint32_t)hitGroups_.size();
    VkDeviceSize totalSize = callableOffset_ + stride * (uint32_t)callableGroups_.size();
    if (totalSize == 0) totalSize = stride;

    // Recreate the buffer only when it is too small (reuse across rebuilds).
    if (buffer_.buffer == VK_NULL_HANDLE || bufferSize_ < totalSize) {
        if (buffer_.buffer != VK_NULL_HANDLE) ctx_->destroyRtBuffer(buffer_);
        buffer_ = ctx_->createRtBuffer(totalSize, VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR
            | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        bufferSize_ = totalSize;
    }

    // Gather shader-group handles for every group used.
    uint32_t totalGroups = (uint32_t)(raygenGroups_.size() + missGroups_.size()
                                     + hitGroups_.size() + callableGroups_.size());
    std::vector<uint8_t> allHandles(totalGroups * handleSize_);
    if (ctx_->dispatch.vkGetRayTracingShaderGroupHandlesKHR(ctx_->device(), pipeline_, 0, totalGroups,
            allHandles.size(), allHandles.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to get ray tracing shader group handles");

    // Stage the SBT contents in a host buffer, then copy into the device buffer.
    std::vector<uint8_t> staging((size_t)totalSize, 0);
    auto writeRegion = [&](const std::vector<uint32_t>& groups, VkDeviceSize offset) {
        for (uint32_t i = 0; i < (uint32_t)groups.size(); ++i) {
            uint32_t g = groups[i];
            VkDeviceSize dst = offset + (VkDeviceSize)i * stride;
            std::memcpy(staging.data() + dst, allHandles.data() + (size_t)g * handleSize_, handleStorage);
        }
    };
    writeRegion(raygenGroups_,   raygenOffset_);
    writeRegion(missGroups_,     missOffset_);
    writeRegion(hitGroups_,      hitOffset_);
    writeRegion(callableGroups_, callableOffset_);

    // Upload via a temporary staging buffer (device-local SBT).
    Buffer stagingBuf = ctx_->app->createBuffer(totalSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, false);
    void* ptr = stagingBuf.map(0);
    std::memcpy(ptr, staging.data(), (size_t)totalSize);
    stagingBuf.unmap();

    // Record the copy into a one-off command buffer and submit synchronously.
    ctx_->app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
        VkBufferCopy c{};
        c.srcOffset = 0; c.dstOffset = 0; c.size = totalSize;
        vkCmdCopyBuffer(cmd, stagingBuf.buffer, buffer_.buffer, 1, &c);
    });
    ctx_->app->deferDestroyUntilAllPending([s = stagingBuf, app = ctx_->app]() mutable {
        app->destroyBuffer(s);
    });
}
