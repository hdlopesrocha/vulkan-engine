#include "MaterialManager.hpp"
#include "VulkanApp.hpp"
#include <stdexcept>
#include <cstring>

static MaterialGPU toGPU(const MaterialProperties &m) {
    MaterialGPU out;
    out.materialFlags = glm::vec4(0.0f, 0.0f, m.ambientFactor, 0.0f);
    out.mappingParams = glm::vec4(m.mappingMode ? 1.0f : 0.0f, m.tessLevel, m.invertHeight ? 1.0f : 0.0f, m.tessHeightScale);
    out.specularParams = glm::vec4(m.specularStrength, m.shininess, 0.0f, 0.0f);
    out.triplanarParams = glm::vec4(m.triplanarScaleU, m.triplanarScaleV, m.triplanar ? 1.0f : 0.0f, 0.0f);
    out.normalParams = glm::vec4(m.normalFlipY ? 1.0f : 0.0f, m.normalSwapXZ ? 1.0f : 0.0f, 0.0f, 0.0f);
    return out;
}

void MaterialManager::allocate(size_t count, VulkanApp* app) {
    if (!app) throw std::runtime_error("MaterialManager::allocate: app is null");
    if (count == 0) return;

    size_t newSize = sizeof(MaterialGPU) * count;

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
        materialBuffer = app->createBuffer(static_cast<VkDeviceSize>(newSize), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        materialBufferSize = newSize;
    }

    materialCount = count;
    cpuCache.assign(materialCount, MaterialGPU{});

    // Initialize to zeros on GPU
    void* mapped = nullptr;
    if (vkMapMemory(app->getDevice(), materialBuffer.memory, 0, materialBufferSize, 0, &mapped) == VK_SUCCESS) {
        memset(mapped, 0, materialBufferSize);
        vkUnmapMemory(app->getDevice(), materialBuffer.memory);
    }
}

void MaterialManager::update(size_t index, const MaterialProperties& mat, VulkanApp* app) {
    if (!app) throw std::runtime_error("MaterialManager::update: app is null");
    if (index >= materialCount) throw std::out_of_range("MaterialManager::update: index out of range");
    MaterialGPU gpu = toGPU(mat);
    cpuCache[index] = gpu;

    VkDevice device = app->getDevice();
    VkDeviceSize offset = static_cast<VkDeviceSize>(index) * sizeof(MaterialGPU);
    void* mapped = nullptr;
    if (vkMapMemory(device, materialBuffer.memory, offset, sizeof(MaterialGPU), 0, &mapped) != VK_SUCCESS) {
        throw std::runtime_error("MaterialManager::update: vkMapMemory failed");
    }
    memcpy(mapped, &gpu, sizeof(MaterialGPU));
    vkUnmapMemory(device, materialBuffer.memory);
}

void MaterialManager::destroy(VulkanApp* app) {
    if (!app) return;
    // Defer destruction/freeing to VulkanResourceManager; clear local handles
    materialBuffer.buffer = VK_NULL_HANDLE;
    materialBuffer.memory = VK_NULL_HANDLE;
    materialCount = 0;
    materialBufferSize = 0;
    cpuCache.clear();
}
