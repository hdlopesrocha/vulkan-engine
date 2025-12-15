// Self-contained header for ModelInstance
#pragma once

#include <glm/glm.hpp>
#include "VertexBufferObject.hpp"   
#include "Buffer.hpp"

struct Model3D;
struct MaterialProperties;

// Represents an instance of a model with its transform
struct ModelInstance {
    Model3D* model;
    VertexBufferObject vbo;
    glm::mat4 transform;
    VkDescriptorSet descriptorSet;
    Buffer* uniformBuffer;
    VkDescriptorSet shadowDescriptorSet;
    Buffer* shadowUniformBuffer;
    const MaterialProperties* material;

    ModelInstance(Model3D* m, const VertexBufferObject& v, const glm::mat4& t, VkDescriptorSet ds, Buffer* ub, VkDescriptorSet sds, Buffer* sub, const MaterialProperties* mat = nullptr);
};
