#pragma once

#include "../core/vulkan.hpp"
#include "Buffer.hpp"
#include "types/MaterialProperties.hpp"
#include "types/ImageType.hpp"
#include "../ssbo/Material.hpp"
#include <vector>
#include <glm/glm.hpp>

class VulkanApp;

class MaterialManager {
public:
    MaterialManager() = default;
    ~MaterialManager() = default;

    // Allocate GPU storage for `count` materials using provided VulkanApp.
    void allocate(size_t count, VulkanApp* app);

    // Update a single material at `index` from CPU-side MaterialProperties.
    void update(size_t index, const MaterialProperties& mat, VulkanApp* app);

    // ── Brush map ───────────────────────────────────────────────────────────
    // Per-brush image source table (types/ImageType.hpp). One uvec2 {type,index}
    // per brush, read by the shaders to resolve which array/layer a brush samples.
    void allocateBrushMap(size_t count, VulkanApp* app);
    void updateBrushMap(size_t index, const BrushImage& image, VulkanApp* app);
    Buffer& getBrushMapBuffer() { return brushMapBuffer; }
    size_t brushMapEntryCount() const { return brushMapCount; }

    // Release GPU resources.
    void destroy(VulkanApp* app);

    // CPU-side mirror strength of a material [0..1]. Used when packing the RT
    // scene-geometry meta so reflection rays can decide whether a hit surface
    // is itself reflective (multi-bounce reflections).
    float reflectionStrength(size_t index) const {
        return (index < cpuCache.size()) ? cpuCache[index].tessLevelParams.z : 0.0f;
    }

    // Access underlying storage buffer (for descriptor binding)
    Buffer& getBuffer() { return materialBuffer; }
    // Byte size of the storage buffer (for vkGetDescriptorEXT, which forbids
    // VK_WHOLE_SIZE ranges). 0 when not allocated.
    size_t bufferSize() const { return materialBufferSize; }

private:
    Buffer materialBuffer{};
    size_t materialCount = 0;
    size_t materialBufferSize = 0;
    std::vector<Material> cpuCache;
    Buffer brushMapBuffer{};
    size_t brushMapCount = 0;
    size_t brushMapByteSize = 0;
};
