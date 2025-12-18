// Translation unit for ModelInstance wrapper header
#include "ModelInstance.hpp"

ModelInstance::ModelInstance(const VertexBufferObject& v, const glm::mat4& t)
    : vbo(v), transform(t) {}
