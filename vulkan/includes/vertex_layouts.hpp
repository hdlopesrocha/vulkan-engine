#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <initializer_list>

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
        VkVertexInputAttributeDescription{ ATTR_BRUSH_INDEX, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, brushIndex) },
        VkVertexInputAttributeDescription{ ATTR_HSV, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, hsv) }
    };
}

inline std::vector<VkVertexInputAttributeDescription> defaultAttributesWithInstance() {
    auto attrs = defaultAttributes();
    attrs.push_back(VkVertexInputAttributeDescription{ ATTR_INSTANCE, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0 });
    return attrs;
}

// Subset of defaultAttributes() keeping only the given attribute locations.
// Pipeline variants whose vertex shader does not consume every default
// attribute must not declare the unused ones: VVL reports the performance
// warning "Vertex attribute at location N not consumed by vertex shader"
// (e.g. the WATER_NO_TESS water vertex path, which prunes COLOR/UV under -O).
inline std::vector<VkVertexInputAttributeDescription> defaultAttributesFiltered(
        std::initializer_list<uint32_t> keepLocations) {
    std::vector<VkVertexInputAttributeDescription> attrs;
    for (const VkVertexInputAttributeDescription& attr : defaultAttributes()) {
        for (uint32_t location : keepLocations) {
            if (attr.location == location) {
                attrs.push_back(attr);
                break;
            }
        }
    }
    return attrs;
}

} // namespace vk_layouts
