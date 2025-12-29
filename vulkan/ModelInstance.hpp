// Self-contained header for ModelInstance
#pragma once

#include <glm/glm.hpp>
#include "VertexBufferObject.hpp"   
#include "Buffer.hpp"

struct MaterialProperties;

// Represents an instance of a model with its transform
struct ModelInstance {
    VertexBufferObject vbo;
    glm::mat4 transform;
    
    ModelInstance(const VertexBufferObject& v, const glm::mat4& t);
};
