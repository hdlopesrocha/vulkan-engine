#include "ModelManager.hpp"

void ModelManager::addInstance(Model3D* model, const glm::mat4& transform, 
                               VkDescriptorSet descriptorSet, Buffer* uniformBuffer,
                               const MaterialProperties* material) {
    instances.emplace_back(model, transform, descriptorSet, uniformBuffer, material);
}

void ModelManager::clear() {
    instances.clear();
}

