#include "AccelerationStructure.hpp"

void AccelerationStructure::ensureStorage(RayTracingContext& ctx, VkDeviceSize size) {
    if (storage_.buffer != VK_NULL_HANDLE && storageSize_ >= size) return;
    if (storage_.buffer != VK_NULL_HANDLE) { ctx.destroyRtBuffer(storage_); storage_ = {}; storageSize_ = 0; }
    storage_ = ctx.createRtBuffer(size, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
    storageSize_ = size;
}

void AccelerationStructure::ensureScratch(RayTracingContext& ctx, VkDeviceSize size) {
    if (scratch_.buffer != VK_NULL_HANDLE && scratchSize_ >= size) return;
    if (scratch_.buffer != VK_NULL_HANDLE) { ctx.destroyRtBuffer(scratch_); scratch_ = {}; scratchSize_ = 0; }
    scratch_ = ctx.createRtBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    scratchSize_ = size;
}

// Instance buffer is HOST_VISIBLE + device-addressable so the CPU can write the
// instance descriptors directly (no staging copy), while still being readable
// by the TLAS build via device address.
void AccelerationStructure::ensureInstanceBuffer(RayTracingContext& ctx, VkDeviceSize size) {
    if (instanceBuffer_.buffer != VK_NULL_HANDLE && instanceSize_ >= size) return;
    if (instanceBuffer_.buffer != VK_NULL_HANDLE) { ctx.destroyRtBuffer(instanceBuffer_); instanceBuffer_ = {}; instanceSize_ = 0; }
    instanceBuffer_ = ctx.app->createBuffer(size,
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        /*zeroInit=*/false);
    instanceSize_ = size;
}

void AccelerationStructure::buildBlas(VulkanApp* app, VkCommandBuffer cmd, RayTracingContext& ctx,
                                      const std::vector<BlasGeometryInput>& geometries,
                                      VkBuildAccelerationStructureFlagsKHR buildFlags) {
    kind_ = AsKind::Blas;
    const uint32_t gc = static_cast<uint32_t>(geometries.size());

    std::vector<VkAccelerationStructureGeometryKHR> geoms(gc);
    std::vector<uint32_t> primCounts(gc);
    for (uint32_t i = 0; i < gc; ++i) {
        const auto& g = geometries[i];
        VkAccelerationStructureGeometryTrianglesDataKHR tris{};
        tris.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        tris.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        tris.vertexData.deviceAddress = g.vertexAddress;
        tris.vertexStride = g.vertexStride;
        tris.maxVertex = g.vertexCount > 0 ? (g.vertexCount - 1) : 0;
        tris.indexType = g.indexType;
        tris.indexData.deviceAddress = g.indexAddress;

        geoms[i].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geoms[i].pNext = nullptr;
        geoms[i].geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geoms[i].flags = g.geometryFlags;
        geoms[i].geometry.triangles = tris;

        primCounts[i] = (g.indexCount / 3);
    }

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.pNext = nullptr;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.flags = buildFlags;
    buildInfo.geometryCount = gc;
    buildInfo.pGeometries = geoms.data();

    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    ctx.getBuildSizes(buildInfo, primCounts.data(), sizes);

    ensureStorage(ctx, sizes.accelerationStructureSize);
    ensureScratch(ctx, sizes.buildScratchSize);

    if (as_ != VK_NULL_HANDLE) {
        ctx.dispatch.vkDestroyAccelerationStructureKHR(ctx.device(), as_, nullptr);
        as_ = VK_NULL_HANDLE;
    }
    VkAccelerationStructureCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    ci.pNext = nullptr;
    ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    ci.size = sizes.accelerationStructureSize;
    ci.buffer = storage_.buffer;
    ci.offset = 0;
    ci.deviceAddress = 0;
    if (ctx.dispatch.vkCreateAccelerationStructureKHR(ctx.device(), &ci, nullptr, &as_) != VK_SUCCESS)
        throw std::runtime_error("failed to create BLAS");

    buildInfo.dstAccelerationStructure = as_;
    {
        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = scratch_.buffer;
        buildInfo.scratchData.deviceAddress = vkGetBufferDeviceAddress(ctx.device(), &addrInfo);
    }

    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges(gc);
    for (uint32_t i = 0; i < gc; ++i) {
        ranges[i].primitiveCount = primCounts[i];
        ranges[i].primitiveOffset = 0;
        ranges[i].firstVertex = 0;
        ranges[i].transformOffset = 0;
    }
    const VkAccelerationStructureBuildRangeInfoKHR* pRanges = ranges.data();

    // Ensure geometry uploaded by earlier commands/submissions is visible to the
    // build before it reads the vertex/index buffers.
    VkMemoryBarrier2KHR mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT_KHR;
    mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT_KHR;
    VkDependencyInfoKHR dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &dep);

    ctx.dispatch.vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRanges);

    {
        VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addrInfo.accelerationStructure = as_;
        address_ = ctx.dispatch.vkGetAccelerationStructureDeviceAddressKHR(ctx.device(), &addrInfo);
    }
}

void AccelerationStructure::buildTlas(VulkanApp* app, VkCommandBuffer cmd, RayTracingContext& ctx,
                                      const std::vector<VkAccelerationStructureInstanceKHR>& instances,
                                      bool update) {
    kind_ = AsKind::Tlas;
    const uint32_t instanceCount = static_cast<uint32_t>(instances.size());

    ensureInstanceBuffer(ctx, std::max<VkDeviceSize>(
        sizeof(VkAccelerationStructureInstanceKHR) * instanceCount, 1));
    // Write instance descriptors directly (instance buffer is host-visible).
    void* ptr = instanceBuffer_.map(0);
    if (ptr) {
        std::memcpy(ptr, instances.data(), sizeof(VkAccelerationStructureInstanceKHR) * instanceCount);
        instanceBuffer_.unmap();
    }

    VkAccelerationStructureGeometryInstancesDataKHR instData{};
    instData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instData.pNext = nullptr;
    instData.arrayOfPointers = VK_FALSE;
    {
        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = instanceBuffer_.buffer;
        instData.data.deviceAddress = vkGetBufferDeviceAddress(ctx.device(), &addrInfo);
    }

    VkAccelerationStructureGeometryKHR geom{};
    geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.pNext = nullptr;
    geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geom.geometry.instances = instData;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.pNext = nullptr;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
                             : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                      | (update ? VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR : 0);
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geom;

    uint32_t maxPrims = instanceCount;
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    ctx.getBuildSizes(buildInfo, &maxPrims, sizes);

    ensureStorage(ctx, sizes.accelerationStructureSize);
    ensureScratch(ctx, sizes.buildScratchSize);

    if (!update || as_ == VK_NULL_HANDLE) {
        if (as_ != VK_NULL_HANDLE) {
            ctx.dispatch.vkDestroyAccelerationStructureKHR(ctx.device(), as_, nullptr);
            as_ = VK_NULL_HANDLE;
        }
        VkAccelerationStructureCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        ci.pNext = nullptr;
        ci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        ci.size = sizes.accelerationStructureSize;
        ci.buffer = storage_.buffer;
        ci.offset = 0;
        ci.deviceAddress = 0;
        if (ctx.dispatch.vkCreateAccelerationStructureKHR(ctx.device(), &ci, nullptr, &as_) != VK_SUCCESS)
            throw std::runtime_error("failed to create TLAS");
    }

    buildInfo.dstAccelerationStructure = as_;
    {
        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = scratch_.buffer;
        buildInfo.scratchData.deviceAddress = vkGetBufferDeviceAddress(ctx.device(), &addrInfo);
    }

    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = instanceCount;
    range.primitiveOffset = 0;
    range.firstVertex = 0;
    range.transformOffset = 0;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

    ctx.dispatch.vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRange);

    {
        VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addrInfo.accelerationStructure = as_;
        address_ = ctx.dispatch.vkGetAccelerationStructureDeviceAddressKHR(ctx.device(), &addrInfo);
    }
}

void AccelerationStructure::teardown(VulkanApp* app) {
    if (as_ != VK_NULL_HANDLE) {
        app->rtDispatch.vkDestroyAccelerationStructureKHR(app->getDevice(), as_, nullptr);
        as_ = VK_NULL_HANDLE;
    }
    if (storage_.buffer != VK_NULL_HANDLE) { app->destroyBuffer(storage_); storage_ = {}; }
    if (scratch_.buffer != VK_NULL_HANDLE) { app->destroyBuffer(scratch_); scratch_ = {}; }
    if (instanceBuffer_.buffer != VK_NULL_HANDLE) { app->destroyBuffer(instanceBuffer_); instanceBuffer_ = {}; }
}
