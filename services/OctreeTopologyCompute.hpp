#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <mutex>
#include "OctreeTopologyTypes.hpp"
#include "../vulkan/resources/Buffer.hpp"

class VulkanApp;

// Rolling stats for the GPU topology pass (dispatch + readback + bucketing for
// one chunk is one "call"). Logged periodically so the GPU path's cost can be
// compared against VULKAN_CPU_TOPOLOGY=1 without any UI wiring.
struct OctreeTopologyStats {
    uint64_t calls = 0;
    double avgMs = 0.0;
    double avgMsLast = 0.0;
    double avgCells = 0.0;
    double avgNodes = 0.0;
    double avgTriangles = 0.0;
};

// GPU triangle-topology service: dispatches
// shaders/renderer/octree/OctreeTriangles.comp (the exact port of
// Octree::iterateTriangles) over a flattened octree neighborhood and reads the
// emitted triangle vertices back for the existing Tesselator/Geometry path.
class OctreeTopologyCompute {
public:
    static OctreeTopologyCompute* globalInstance __attribute__((weak));
    static OctreeTopologyCompute* getGlobal() __attribute__((weak)) { return globalInstance; }

    OctreeTopologyCompute();
    void init(VulkanApp* app);
    void destroy(VulkanApp* app);

    // Synchronous, worker-thread safe (serialized). Returns the emitted
    // triangles; when the fixed output arena overflows, `truncated` is set and
    // the caller must fall back to the CPU walker.
    OctreeTopologyResult compute(const OctreeTopologyInput& input) __attribute__((weak));

    OctreeTopologyStats stats() __attribute__((weak));

    VulkanApp* app = nullptr;
    std::mutex mutex;

private:
    void ensureCapacity(size_t nodeCount, size_t cellCount, size_t vertexCount);

    uint64_t statCalls = 0;
    double statMs = 0.0;
    double statMsLast = 0.0;
    double statCells = 0.0;
    double statNodes = 0.0;
    double statTriangles = 0.0;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;

    size_t nodeCap = 0;
    size_t cellCap = 0;
    size_t vertexCap = 0;
    Buffer bSdf0{}, bSdf1{}, bCube{}, bMeta{}, bHsv{}, bChild03{}, bChild47{};
    Buffer bCells{}, bPos{}, bNrm{}, bOutHsv{}, bCounter{};
};
