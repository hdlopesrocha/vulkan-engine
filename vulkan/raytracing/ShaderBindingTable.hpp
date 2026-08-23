#pragma once

#include "RayTracingContext.hpp"
#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>

// Builds a Shader Binding Table for a ray-tracing pipeline. The SBT is a single
// device-addressable buffer laid out as [raygen][miss...][hit...][callable...].
// Each group entry holds its shader-group handle (padded to the device's
// required stride). Region descriptors expose the addresses/strides consumed by
// vkCmdTraceRaysKHR. Different geometry/material types are represented as
// separate HIT GROUPS (see RayTracingRenderer) rather than separate SBT buffers,
// so the layout stays compact and extensible.
class ShaderBindingTable {
public:
    ShaderBindingTable() = default;

    void init(RayTracingContext& ctx, VkPipeline pipeline,
              uint32_t handleSize, uint32_t baseAlignment, uint32_t handleAlignment) {
        ctx_ = &ctx;
        pipeline_ = pipeline;
        handleSize_ = handleSize;
        baseAlign_ = baseAlignment;
        handleAlign_ = handleAlignment;
        raygenGroups_.clear(); missGroups_.clear(); hitGroups_.clear(); callableGroups_.clear();
        buffer_ = {};
    }

    void setRaygen(uint32_t groupIndex) { raygenGroups_ = { groupIndex }; }
    void addMiss(uint32_t groupIndex) { missGroups_.push_back(groupIndex); }
    void addHitGroup(uint32_t groupIndex) { hitGroups_.push_back(groupIndex); }
    void addCallable(uint32_t groupIndex) { callableGroups_.push_back(groupIndex); }

    // Allocate the SBT buffer and write the shader-group handles into it.
    void build();

    VkStridedDeviceAddressRegionKHR raygenRegion()   const { return region(raygenOffset_, 1); }
    VkStridedDeviceAddressRegionKHR missRegion()     const { return region(missOffset_,    (uint32_t)missGroups_.size()); }
    VkStridedDeviceAddressRegionKHR hitGroupRegion() const { return region(hitOffset_,     (uint32_t)hitGroups_.size()); }
    VkStridedDeviceAddressRegionKHR callableRegion() const { return region(callableOffset_,(uint32_t)callableGroups_.size()); }

    void destroy() { if (buffer_.buffer != VK_NULL_HANDLE) { ctx_->destroyRtBuffer(buffer_); buffer_ = {}; } }

private:
    RayTracingContext* ctx_ = nullptr;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    uint32_t handleSize_ = 0, baseAlign_ = 64, handleAlign_ = 64;

    std::vector<uint32_t> raygenGroups_, missGroups_, hitGroups_, callableGroups_;
    Buffer buffer_;
    VkDeviceSize bufferSize_ = 0;
    VkDeviceSize raygenOffset_ = 0, missOffset_ = 0, hitOffset_ = 0, callableOffset_ = 0;

    uint32_t entryStride() const {
        uint32_t s = handleSize_;
        if (handleAlign_ > 1) s = (s + handleAlign_ - 1) & ~(handleAlign_ - 1);
        return s;
    }
    VkStridedDeviceAddressRegionKHR region(VkDeviceSize offset, uint32_t count) const {
        VkStridedDeviceAddressRegionKHR r{};
        if (buffer_.buffer) {
            VkBufferDeviceAddressInfo addrInfo{};
            addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            addrInfo.buffer = buffer_.buffer;
            r.deviceAddress = vkGetBufferDeviceAddress(ctx_->device(), &addrInfo) + offset;
        } else {
            r.deviceAddress = 0;
        }
        r.stride = entryStride();
        r.size = count > 0 ? entryStride() * count : 0;
        return r;
    }
};
