#include "ModelManager.hpp"

void ModelManager::addInstance(Mesh3D* model, const VertexBufferObject& vbo, const glm::mat4& transform, 
                               VkDescriptorSet descriptorSet,
                               VkDescriptorSet shadowDescriptorSet) {
    instances.emplace_back(model, vbo, transform, descriptorSet, shadowDescriptorSet);
}

void ModelManager::clear() {
    instances.clear();
}


