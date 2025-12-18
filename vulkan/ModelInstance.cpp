// Translation unit for ModelInstance wrapper header
#include "ModelInstance.hpp"

ModelInstance::ModelInstance(Mesh3D* m, const VertexBufferObject& v, const glm::mat4& t)
    : model(m), vbo(v), transform(t) {}
