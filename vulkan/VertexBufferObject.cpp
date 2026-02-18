// Translation unit for VertexBufferObject wrapper header
#include "VertexBufferObject.hpp"

void VertexBufferObject::destroy(VkDevice device) {
    // Rely on VulkanResourceManager to destroy tracked buffers/memory later
    vertexBuffer.buffer = VK_NULL_HANDLE;
    vertexBuffer.memory = VK_NULL_HANDLE;
    indexBuffer.buffer = VK_NULL_HANDLE;
    indexBuffer.memory = VK_NULL_HANDLE;
}
