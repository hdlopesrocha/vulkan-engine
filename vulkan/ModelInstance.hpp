// Self-contained header for ModelInstance
#pragma once

#include <glm/glm.hpp>
#include "VertexBufferObject.hpp"   
#include "Buffer.hpp"

struct Mesh3D;
struct MaterialProperties;

// Represents an instance of a model with its transform
struct ModelInstance {
    Mesh3D* model;
    VertexBufferObject vbo;
    glm::mat4 transform;
    VkDescriptorSet descriptorSet;
    Buffer* uniformBuffer;
    VkDescriptorSet shadowDescriptorSet;
    Buffer* shadowUniformBuffer;
    
    ModelInstance(Mesh3D* m, const VertexBufferObject& v, const glm::mat4& t, VkDescriptorSet ds, Buffer* ub, VkDescriptorSet sds, Buffer* sub);
};
