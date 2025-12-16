#include "ModelManager.hpp"

void ModelManager::addInstance(Mesh3D* model, const VertexBufferObject& vbo, const glm::mat4& transform, 
                               VkDescriptorSet descriptorSet, Buffer* uniformBuffer,
                               VkDescriptorSet shadowDescriptorSet, Buffer* shadowUniformBuffer) {
    instances.emplace_back(model, vbo, transform, descriptorSet, uniformBuffer, shadowDescriptorSet, shadowUniformBuffer);
}

void ModelManager::clear() {
    instances.clear();
}


