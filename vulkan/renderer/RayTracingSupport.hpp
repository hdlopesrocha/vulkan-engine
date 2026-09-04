#pragma once
// RayTracingSupport — device capability detection, feature enabling helpers,
// entry-point loader and shared RT constants for the hardware ray-tracing
// rendering path (VK_KHR_acceleration_structure + VK_KHR_ray_tracing_pipeline).
//
// Feature detection is preferred over version checks everywhere: every flag
// below is queried from the physical device before any optional functionality
// is enabled, and the renderer falls back to the legacy rasterizer when the
// required features/extensions are unavailable.

#include <vulkan/vulkan.h>
#include <cstring>
#include <string>
#include <vector>

namespace rt {

// Required device extensions for the full ray-tracing pipeline path.
// VK_KHR_deferred_host_operations is required by the acceleration-structure
// spec for host builds (we build on-device, but the loader still expects the
// extension present on most drivers when AS is enabled).
inline const char* kRequiredDeviceExtensions[] = {
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
};

// Optional but preferred: buffer device address is core in 1.2 but must be
// explicitly enabled; ray query enables shadow-ray evaluation in any shader
// stage (used as a fallback/complement to pipeline trace calls).
inline const char* kOptionalDeviceExtensions[] = {
    VK_KHR_RAY_QUERY_EXTENSION_NAME,
};

// ── Instance → mesh → primitive → material routing ─────────────────────────
// TLAS VkAccelerationStructureInstanceKHR::instanceCustomIndexEXT packing.
// NOTE: instanceCustomIndex is a 24-bit bitfield (mask is the other 8 bits of
// the same word), so all flags must live in bits 0..23:
// bit 23     : water flag (MATERIAL_WATER) — hit belongs to the water layer.
// bit 22     : vegetation/alpha-tested flag (foliage uses any-hit discard).
// bits 0..21 : BLAS/slot id within its layer (matches IndirectRenderer slot).
constexpr uint32_t kWaterInstanceBit = 0x00800000u;
constexpr uint32_t kVegetationInstanceBit = 0x00400000u;
constexpr uint32_t kSlotIdMask = 0x003FFFFFu;
constexpr uint32_t kWaterFlagSlotBit = 30; // slot-meta layer bit (solid=0, water=1)

inline uint32_t makeCustomIndex(uint32_t slotId, bool water, bool vegetation = false) {
    uint32_t v = slotId & kSlotIdMask;
    if (water) v |= kWaterInstanceBit;
    if (vegetation) v |= kVegetationInstanceBit;
    return v;
}
inline bool customIndexIsWater(uint32_t ci) { return (ci & kWaterInstanceBit) != 0; }
inline bool customIndexIsVegetation(uint32_t ci) { return (ci & kVegetationInstanceBit) != 0; }
inline uint32_t customIndexSlot(uint32_t ci) { return ci & kSlotIdMask; }

// Ray payload / miss / hit-group indices for the single RT pipeline.
constexpr uint32_t kRayTypeRadiance = 0;
constexpr uint32_t kRayTypeShadow = 1;
constexpr uint32_t kRayTypeCount = 2;

// Trace flags shared by all rays.
constexpr float kRayTMin = 0.001f;
constexpr float kRayTMax = 1e32f;
// Robust origin offset along the geometric normal to avoid self-intersection.
constexpr float kRayOriginEpsilon = 1e-3f;

// Bounded recursion: PRIMARY -> REFLECTION(1) -> REFRACTION(1); max depth 2-4.
constexpr uint32_t kMaxRecursionDepth = 2;
constexpr uint32_t kMaxBounceDepthConfigurableMax = 4;

// Water optics.
constexpr float kWaterIOR = 1.333f;
constexpr float kWaterAirIOR = 1.0f;

// Result of probing the physical device for RT capability.
struct DeviceSupport {
    bool accelerationStructure = false;
    bool rayTracingPipeline = false;
    bool deferredHostOps = false;
    bool bufferDeviceAddress = false;
    bool scalarBlockLayout = false;
    bool rayQuery = false; // optional
    bool usable() const { return accelerationStructure && rayTracingPipeline && bufferDeviceAddress; }
    std::string missingReason() const {
        std::string s;
        if (!accelerationStructure) s += "VK_KHR_acceleration_structure ";
        if (!rayTracingPipeline) s += "VK_KHR_ray_tracing_pipeline ";
        if (!bufferDeviceAddress) s += "bufferDeviceAddress ";
        return s.empty() ? std::string("none") : s;
    }
};

// Query extension availability + feature support. Pure feature detection:
// never assumes support from the Vulkan version alone.
inline DeviceSupport queryDeviceSupport(VkPhysicalDevice phys) {
    DeviceSupport out;
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    if (extCount) vkEnumerateDeviceExtensionProperties(phys, nullptr, &extCount, exts.data());
    auto hasExt = [&](const char* name) {
        for (auto& e : exts) {
            if (std::strcmp(e.extensionName, name) == 0) return true;
        }
        return false;
    };
    const bool hasAS = hasExt(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    const bool hasRTP = hasExt(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    const bool hasDHO = hasExt(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    const bool hasRQ = hasExt(VK_KHR_RAY_QUERY_EXTENSION_NAME);

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeat{};
    asFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtpFeat{};
    rtpFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rtpFeat.pNext = &asFeat;
    VkPhysicalDeviceRayQueryFeaturesKHR rqFeat{};
    rqFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rqFeat.pNext = &rtpFeat;
    VkPhysicalDeviceVulkan12Features f12{};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f12.pNext = &rqFeat;
    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &f12;
    vkGetPhysicalDeviceFeatures2(phys, &f2);

    out.accelerationStructure = hasAS && asFeat.accelerationStructure == VK_TRUE;
    out.rayTracingPipeline = hasRTP && rtpFeat.rayTracingPipeline == VK_TRUE;
    out.deferredHostOps = hasDHO;
    out.bufferDeviceAddress = f12.bufferDeviceAddress == VK_TRUE;
    out.scalarBlockLayout = f12.scalarBlockLayout == VK_TRUE;
    out.rayQuery = hasRQ && rqFeat.rayQuery == VK_TRUE;
    return out;
}

// Per-device KHR entry points. Loaded once after vkCreateDevice when RT is
// enabled; all null otherwise so callers can branch safely.
struct Functions {
    PFN_vkGetAccelerationStructureBuildSizesKHR getBuildSizes = nullptr;
    PFN_vkCreateAccelerationStructureKHR createAS = nullptr;
    PFN_vkDestroyAccelerationStructureKHR destroyAS = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR cmdBuild = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR getAddress = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR createRTPipelines = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR getGroupHandles = nullptr;
    PFN_vkCmdTraceRaysKHR cmdTraceRays = nullptr;
    bool loaded() const {
        return getBuildSizes && createAS && destroyAS && cmdBuild && getAddress &&
               createRTPipelines && getGroupHandles && cmdTraceRays;
    }
    void load(VkDevice dev) {
        getBuildSizes = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
            vkGetDeviceProcAddr(dev, "vkGetAccelerationStructureBuildSizesKHR"));
        createAS = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
            vkGetDeviceProcAddr(dev, "vkCreateAccelerationStructureKHR"));
        destroyAS = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
            vkGetDeviceProcAddr(dev, "vkDestroyAccelerationStructureKHR"));
        cmdBuild = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
            vkGetDeviceProcAddr(dev, "vkCmdBuildAccelerationStructuresKHR"));
        getAddress = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
            vkGetDeviceProcAddr(dev, "vkGetAccelerationStructureDeviceAddressKHR"));
        createRTPipelines = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(
            vkGetDeviceProcAddr(dev, "vkCreateRayTracingPipelinesKHR"));
        getGroupHandles = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
            vkGetDeviceProcAddr(dev, "vkGetRayTracingShaderGroupHandlesKHR"));
        cmdTraceRays = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(
            vkGetDeviceProcAddr(dev, "vkCmdTraceRaysKHR"));
    }
};

inline VkDeviceAddress getBufferAddress(VkDevice dev, VkBuffer buf) {
    VkBufferDeviceAddressInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buf;
    return vkGetBufferDeviceAddress(dev, &info);
}

inline VkDeviceAddress getASAddress(VkDevice dev,
                                    PFN_vkGetAccelerationStructureDeviceAddressKHR fn,
                                    VkAccelerationStructureKHR as) {
    VkAccelerationStructureDeviceAddressInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    info.accelerationStructure = as;
    return fn(dev, &info);
}

} // namespace rt
