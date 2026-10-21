#include "MaterialManager.hpp"
#include "../core/VulkanApp.hpp"
#include <stdexcept>
#include <cstring>

static Material toGPU(const MaterialProperties &m) {
    Material out;
    out.materialFlags = glm::vec4(0.0f, 0.0f, m.ambientFactor, 0.0f);
    out.mappingParams = glm::vec4(m.mappingMode ? 1.0f : 0.0f, m.tessLevel, m.invertHeight ? 1.0f : 0.0f, m.tessHeightScale);
    out.specularParams = glm::vec4(m.specularStrength, m.shininess, 0.0f, 0.0f);
    out.triplanarParams = glm::vec4(m.triplanarScaleU, m.triplanarScaleV, 0.0f, 0.0f);
    out.normalParams = glm::vec4(m.normalFlipY ? 1.0f : 0.0f, m.normalSwapXZ ? 1.0f : 0.0f, m.invertWidth ? 1.0f : 0.0f, 0.0f);
    out.tessLevelParams = glm::vec4(m.tessMinLevel, m.tessMaxLevel, m.reflectionStrength, 0.0f);
    out.roughnessAOParams = glm::vec4(m.roughnessFactor, m.aoFactor, m.useAO ? 1.0f : 0.0f, 0.0f);
    // layerParams.z (coordinate technique) is reserved: the technique is a
    // per-vertex value now (Vertex::textureType).
    out.layerParams = glm::vec4(static_cast<float>(m.kind), static_cast<float>(m.textureIndex),
                                0.0f, static_cast<float>(m.blendMode));
    out.mappingOrigin = m.mappingOrigin;
    return out;
}

void MaterialManager::allocate(size_t count, VulkanApp* app) {
    if (!app) throw std::runtime_error("MaterialManager::allocate: app is null");
    if (count == 0) return;

    size_t newSize = sizeof(Material) * count;

    // Destroy previous buffer if size differs
    if (materialBuffer.buffer != VK_NULL_HANDLE && materialBufferSize != newSize) {
        // Defer actual destruction to VulkanResourceManager; clear local handle
        materialBuffer.buffer = VK_NULL_HANDLE;
    }
    if (materialBuffer.memory != VK_NULL_HANDLE && materialBuffer.buffer == VK_NULL_HANDLE) {
        // Defer actual freeing to VulkanResourceManager; clear local handle
        materialBuffer.memory = VK_NULL_HANDLE;
    }

    if (materialBuffer.buffer == VK_NULL_HANDLE) {
        // Descriptor-buffer sources need a device address for vkGetDescriptorEXT,
        // which requires SHADER_DEVICE_ADDRESS_BIT at buffer creation. The flag
        // is harmless on the classic vkUpdateDescriptorSets path.
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (app->useDescriptorBuffer())
            usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        materialBuffer = app->createBuffer(static_cast<VkDeviceSize>(newSize), usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        materialBufferSize = newSize;
    }

    materialCount = count;
    cpuCache.assign(materialCount, Material{});

    // Initialize to zeros on GPU
    memset(materialBuffer.mappedData, 0, materialBufferSize);
}

void MaterialManager::update(size_t index, const MaterialProperties& mat, VulkanApp* app) {
    if (!app) throw std::runtime_error("MaterialManager::update: app is null");
    if (index >= materialCount) throw std::out_of_range("MaterialManager::update: index out of range");
    Material gpu = toGPU(mat);
    cpuCache[index] = gpu;

    VkDeviceSize offset = static_cast<VkDeviceSize>(index) * sizeof(Material);
    memcpy(materialBuffer.map(offset), &gpu, sizeof(Material));
}

void MaterialManager::allocateBrushMap(size_t count, VulkanApp* app) {
    if (!app) throw std::runtime_error("MaterialManager::allocateBrushMap: app is null");
    if (count == 0) return;
    const size_t newSize = sizeof(glm::uvec2) * count;
    if (brushMapBuffer.buffer != VK_NULL_HANDLE && brushMapByteSize != newSize) {
        brushMapBuffer.buffer = VK_NULL_HANDLE;
    }
    if (brushMapBuffer.memory != VK_NULL_HANDLE && brushMapBuffer.buffer == VK_NULL_HANDLE) {
        brushMapBuffer.memory = VK_NULL_HANDLE;
    }
    if (brushMapBuffer.buffer == VK_NULL_HANDLE) {
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (app->useDescriptorBuffer()) usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        brushMapBuffer = app->createBuffer(static_cast<VkDeviceSize>(newSize), usage,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        brushMapByteSize = newSize;
    }
    brushMapCount = count;
    memset(brushMapBuffer.mappedData, 0, brushMapByteSize);
}

void MaterialManager::updateBrushMap(size_t index, const BrushImage& image, VulkanApp* app) {
    if (!app) throw std::runtime_error("MaterialManager::updateBrushMap: app is null");
    if (index >= brushMapCount) throw std::out_of_range("MaterialManager::updateBrushMap: index out of range");
    glm::uvec2 entry(static_cast<uint32_t>(image.type), image.index);
    VkDeviceSize offset = static_cast<VkDeviceSize>(index) * sizeof(glm::uvec2);
    memcpy(brushMapBuffer.map(offset), &entry, sizeof(glm::uvec2));
}

void MaterialManager::destroy(VulkanApp* app) {
    if (!app) return;
    // Defer destruction/freeing to VulkanResourceManager; clear local handles
    materialBuffer.buffer = VK_NULL_HANDLE;
    materialBuffer.memory = VK_NULL_HANDLE;
    materialCount = 0;
    materialBufferSize = 0;
    cpuCache.clear();
    brushMapBuffer.buffer = VK_NULL_HANDLE;
    brushMapBuffer.memory = VK_NULL_HANDLE;
    brushMapCount = 0;
    brushMapByteSize = 0;
}
