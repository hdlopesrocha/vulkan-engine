#include "ModelManager.hpp"

void ModelManager::addInstance(Model3D* model, const VertexBufferObject& vbo, const glm::mat4& transform, 
                               VkDescriptorSet descriptorSet, Buffer* uniformBuffer,
                               VkDescriptorSet shadowDescriptorSet, Buffer* shadowUniformBuffer,
                               const MaterialProperties* material) {
    instances.emplace_back(model, vbo, transform, descriptorSet, uniformBuffer, shadowDescriptorSet, shadowUniformBuffer, material);
}

void ModelManager::clear() {
    instances.clear();
}

