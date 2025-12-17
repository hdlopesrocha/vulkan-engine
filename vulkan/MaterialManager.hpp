#pragma once

#include "vulkan/vulkan.hpp"
#include "Buffer.hpp"
#include "../utils/MaterialProperties.hpp"
#include <vector>

struct MaterialGPU {
    glm::vec4 materialFlags;   // .z = ambientFactor
    glm::vec4 mappingParams;   // x = mappingEnabled (0/1), y = tessLevel, z = invertHeight (0/1), w = tessHeightScale
    glm::vec4 specularParams;  // x = specularStrength, y = shininess
    glm::vec4 triplanarParams; // x = scaleU, y = scaleV, z = triplanarEnabled (0/1)
};

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
