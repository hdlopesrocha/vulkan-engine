#include "ModelManager.hpp"

void ModelManager::addInstance(const VertexBufferObject& vbo, const glm::mat4& transform) {
    instances.emplace_back(vbo, transform);
}

void ModelManager::clear() {
    instances.clear();
}


