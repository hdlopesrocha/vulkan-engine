#pragma once

#include "RayTracingContext.hpp"
#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>

// One geometry contribution to a BLAS, built from an existing GPU vertex/index
// buffer range (e.g. a chunk's span inside the slotted IndirectRenderer pools).
// Addresses are absolute device addresses of the FIRST vertex/index of the span.
struct BlasGeometryInput {
    VkDeviceAddress vertexAddress = 0;   // first vertex
    uint32_t        vertexCount   = 0;   // maxVertex (indexed draws use indexCount)
    uint32_t        vertexStride  = 64;  // sizeof(Vertex)
    VkDeviceAddress indexAddress  = 0;   // first index
    uint32_t        indexCount    = 0;
    VkIndexType     indexType     = VK_INDEX_TYPE_UINT32;
    VkGeometryFlagsKHR geometryFlags = VK_GEOMETRY_OPAQUE_BIT_KHR; // opaque=solid; non-opaque=water/vegetation (any-hit)
    VkPrimitiveTopology topology  = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
};

enum class AsKind { Blas, Tlas };

// Wraps a single VkAccelerationStructureKHR plus its backing storage and scratch.
// Designed to be owned by a std::unique_ptr in the chunk map; on replacement the
// old object is moved into a deferred-destroy callback (see teardown()) so GPU
// resources still referenced by in-flight frames are never freed early.
class AccelerationStructure {
public:
    AccelerationStructure() = default;
    AccelerationStructure(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;
    AccelerationStructure(AccelerationStructure&&) = default;
    AccelerationStructure& operator=(AccelerationStructure&&) = default;
    ~AccelerationStructure() { /* teardown is explicit; destructor only used at shutdown-after-device-lost */ }

    // Build (or rebuild) a BLAS from one or more geometry inputs. Records the
    // build into `cmd`. The caller submits `cmd` and is responsible for ensuring
    // the source geometry is resident and for deferring destruction of the
    // PREVIOUS BLAS (the one this object replaced) once the submit fence signals.
    void buildBlas(VulkanApp* app, VkCommandBuffer cmd, RayTracingContext& ctx,
                   const std::vector<BlasGeometryInput>& geometries,
                   VkBuildAccelerationStructureFlagsKHR buildFlags =
                       VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

    // Build (or UPDATE in place) the TLAS from instance descriptors. `instances`
    // is copied into a GPU buffer owned by this object; on UPDATE the same buffer
    // is rewritten and the existing TLAS storage reused (cheap path used when only
    // instance transforms/masks change — which for static terrain never happens,
    // so the TLAS is only rebuilt on chunk add/remove).
    void buildTlas(VulkanApp* app, VkCommandBuffer cmd, RayTracingContext& ctx,
                   const std::vector<VkAccelerationStructureInstanceKHR>& instances,
                   bool update = false);

    VkDeviceAddress deviceAddress() const { return address_; }
    VkAccelerationStructureKHR handle() const { return as_; }
    bool valid() const { return as_ != VK_NULL_HANDLE; }
    AsKind kind() const { return kind_; }

    // GPU-safe teardown. Destroys the acceleration structure, its storage buffer,
    // the (TLAS) instance buffer, and the scratch buffer through the app's
    // resource manager. Intended to be called from a deferred-destroy callback
    // after the submitting fence has signaled.
    void teardown(VulkanApp* app);

private:
    AsKind kind_ = AsKind::Blas;
    VkAccelerationStructureKHR as_ = VK_NULL_HANDLE;
    Buffer storage_;        // AS backing storage
    Buffer scratch_;        // reused across rebuilds of this AS
    Buffer instanceBuffer_; // TLAS only: VkAccelerationStructureInstanceKHR[]
    VkDeviceAddress address_ = 0;
    VkDeviceSize storageSize_ = 0;
    VkDeviceSize scratchSize_ = 0;
    VkDeviceSize instanceSize_ = 0;

    void ensureStorage(RayTracingContext& ctx, VkDeviceSize size);
    void ensureScratch(RayTracingContext& ctx, VkDeviceSize size);
    void ensureInstanceBuffer(RayTracingContext& ctx, VkDeviceSize size);
};
