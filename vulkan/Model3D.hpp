#pragma once

#include "VertexBufferObject.hpp"
#include <glm/glm.hpp>

class Mesh3D; // forward declare to avoid include cycles

class Model3D {
public:
    Mesh3D *mesh;                    // pointer to the source mesh
    VertexBufferObject &vbo;         // reference to the GPU buffers
    glm::mat4 model;                 // model transform

    Model3D(Mesh3D *meshPtr, VertexBufferObject &vboRef, const glm::mat4 &modelMat = glm::mat4(1.0f));
    ~Model3D() = default;

    void setModel(const glm::mat4 &m) { model = m; }
    const glm::mat4 &getModel() const { return model; }

    VertexBufferObject &getVBO() { return vbo; }
    const VertexBufferObject &getVBO() const { return vbo; }
};
