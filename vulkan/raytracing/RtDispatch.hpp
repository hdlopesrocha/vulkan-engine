#pragma once

#include <vulkan/vulkan.h>

// Runtime dispatch for the ray-tracing KHR extensions. The Vulkan loader does
// not statically export these symbols (they are resolved dynamically from the
// instance), so we fetch their function pointers once and call through them.
// Per AGENTS.md this keeps validation-clean, version-agnostic behavior: we only
// ever call them when ray tracing is supported AND the pointers resolved.
struct RtDispatch {
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR = nullptr;
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR = nullptr;
    PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR = nullptr;

    void load(VkInstance instance) {
        vkCreateAccelerationStructureKHR =
            (PFN_vkCreateAccelerationStructureKHR)vkGetInstanceProcAddr(instance, "vkCreateAccelerationStructureKHR");
        vkDestroyAccelerationStructureKHR =
            (PFN_vkDestroyAccelerationStructureKHR)vkGetInstanceProcAddr(instance, "vkDestroyAccelerationStructureKHR");
        vkGetAccelerationStructureBuildSizesKHR =
            (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetInstanceProcAddr(instance, "vkGetAccelerationStructureBuildSizesKHR");
        vkCmdBuildAccelerationStructuresKHR =
            (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetInstanceProcAddr(instance, "vkCmdBuildAccelerationStructuresKHR");
        vkGetAccelerationStructureDeviceAddressKHR =
            (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetInstanceProcAddr(instance, "vkGetAccelerationStructureDeviceAddressKHR");
        vkCreateRayTracingPipelinesKHR =
            (PFN_vkCreateRayTracingPipelinesKHR)vkGetInstanceProcAddr(instance, "vkCreateRayTracingPipelinesKHR");
        vkGetRayTracingShaderGroupHandlesKHR =
            (PFN_vkGetRayTracingShaderGroupHandlesKHR)vkGetInstanceProcAddr(instance, "vkGetRayTracingShaderGroupHandlesKHR");
        vkCmdTraceRaysKHR =
            (PFN_vkCmdTraceRaysKHR)vkGetInstanceProcAddr(instance, "vkCmdTraceRaysKHR");
    }

    bool ready() const { return vkCmdTraceRaysKHR != nullptr; }
};
