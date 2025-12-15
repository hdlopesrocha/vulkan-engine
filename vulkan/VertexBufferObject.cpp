// Translation unit for VertexBufferObject wrapper header
#include "VertexBufferObject.hpp"

void VertexBufferObject::destroy(VkDevice device) {
    if (vertexBuffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, vertexBuffer.buffer, nullptr);
    if (vertexBuffer.memory != VK_NULL_HANDLE) vkFreeMemory(device, vertexBuffer.memory, nullptr);
    if (indexBuffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, indexBuffer.buffer, nullptr);
    if (indexBuffer.memory != VK_NULL_HANDLE) vkFreeMemory(device, indexBuffer.memory, nullptr);
}
