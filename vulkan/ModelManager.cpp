#include "ModelManager.hpp"

void ModelManager::addInstance(Mesh3D* model, const VertexBufferObject& vbo, const glm::mat4& transform) {
    instances.emplace_back(model, vbo, transform);
}

void ModelManager::clear() {
    instances.clear();
}


