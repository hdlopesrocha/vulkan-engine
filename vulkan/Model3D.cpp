
#include "Model3D.hpp"

Model3D::Model3D(VertexBufferObject &vboRef, const glm::mat4 &modelMat)
    : vbo(vboRef), model(modelMat) {}

