
#include "Model3D.hpp"

Model3D::Model3D(Mesh3D *meshPtr, VertexBufferObject &vboRef, const glm::mat4 &modelMat)
    : mesh(meshPtr), vbo(vboRef), model(modelMat) {}

