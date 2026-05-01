#pragma once

#include "vulkan.hpp"
#include "Buffer.hpp"
#include "../utils/MaterialProperties.hpp"
#include "ubo/MaterialGPU.hpp"
#include <vector>

class VulkanApp;

class MaterialManager {
public:
    MaterialManager() = default;
    ~MaterialManager() = default;

    // Allocate GPU storage for `count` materials using provided VulkanApp.
    void allocate(size_t count, VulkanApp* app);

    // Update a single material at `index` from CPU-side MaterialProperties.
    void update(size_t index, const MaterialProperties& mat, VulkanApp* app);

    // Release GPU resources.
    void destroy(VulkanApp* app);

    size_t count() const { return materialCount; }

    // Access underlying storage buffer (for descriptor binding)
    const Buffer& getBuffer() const { return materialBuffer; }
    Buffer& getBuffer() { return materialBuffer; }

private:
    Buffer materialBuffer{};
    size_t materialCount = 0;
    size_t materialBufferSize = 0;
    std::vector<MaterialGPU> cpuCache;
};
