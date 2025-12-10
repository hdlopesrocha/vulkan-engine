#include "ModelManager.hpp"

void ModelManager::addInstance(Model3D* model, const glm::mat4& transform, 
                               VkDescriptorSet descriptorSet, Buffer* uniformBuffer) {
    instances.emplace_back(model, transform, descriptorSet, uniformBuffer);
}

void ModelManager::clear() {
    instances.clear();
}
