#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>
#include <cstddef>

#include "locations.hpp"
#include "../../math/Vertex.hpp"

namespace vk_layouts {

inline std::vector<VkVertexInputBindingDescription> defaultBindings() {
    return { VkVertexInputBindingDescription{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } };
}

inline std::vector<VkVertexInputBindingDescription> vertexAndInstanceBindings(uint32_t instanceStride) {
    return {
        VkVertexInputBindingDescription{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX },
        VkVertexInputBindingDescription{ 1, instanceStride, VK_VERTEX_INPUT_RATE_INSTANCE }
    };
}

inline std::vector<VkVertexInputAttributeDescription> defaultAttributes() {
    return {
        VkVertexInputAttributeDescription{ ATTR_POS, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
        VkVertexInputAttributeDescription{ ATTR_COLOR, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
        VkVertexInputAttributeDescription{ ATTR_UV, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) },
        VkVertexInputAttributeDescription{ ATTR_NORMAL, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
        VkVertexInputAttributeDescription{ ATTR_BRUSH_INDEX, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, brushIndex) }
    };
}

inline std::vector<VkVertexInputAttributeDescription> defaultAttributesWithInstance() {
    auto attrs = defaultAttributes();
    attrs.push_back(VkVertexInputAttributeDescription{ ATTR_INSTANCE, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0 });
    return attrs;
}

} // namespace vk_layouts
