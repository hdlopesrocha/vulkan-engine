#pragma once

#define GLM_FORCE_DEPTH_ZERO_TO_ONE  // Use Vulkan depth range [0,1] instead of OpenGL [-1,1]

#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <optional>
#include <set>
#include <fstream>
#include <array>
#include <algorithm>
#include <cmath>
// stb_image for texture loading
#include <stb/stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/integer.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/norm.hpp> 
#include <glm/matrix.hpp>

#include "FileReader.hpp"

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// Forward declarations for types defined in implementation headers
class VulkanApp;
struct MaterialProperties;

// Vertex definition from utils
#include "../utils/Vertex.hpp"

// Small helper types used across the Vulkan module
struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

struct TextureImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    uint32_t mipLevels = 1;
};

struct VertexBufferObject {
    Buffer vertexBuffer;
    Buffer indexBuffer;
    uint32_t indexCount = 0;
    void destroy(VkDevice device) {
        if (vertexBuffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, vertexBuffer.buffer, nullptr);
        if (vertexBuffer.memory != VK_NULL_HANDLE) vkFreeMemory(device, vertexBuffer.memory, nullptr);
        if (indexBuffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, indexBuffer.buffer, nullptr);
        if (indexBuffer.memory != VK_NULL_HANDLE) vkFreeMemory(device, indexBuffer.memory, nullptr);
    }
};

// Small helper to create shader stage infos
struct ShaderStage {
    VkPipelineShaderStageCreateInfo info{};
    ShaderStage() = default;
    ShaderStage(VkShaderModule module, VkShaderStageFlagBits stage) {
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage = stage;
        info.module = module;
        info.pName = "main";
    }
};

// Queue family indices helper
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    bool isComplete() const {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

// Include the full VulkanApp declaration so translation units including
// `vulkan.hpp` get the concrete `VulkanApp` type as well.
#include "VulkanApp.hpp"