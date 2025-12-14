#pragma once

#include "../utils/Model3D.hpp"
#include "vulkan.hpp"
#include <vector>
#include <glm/glm.hpp>

// Forward declaration
struct MaterialProperties;

// Represents an instance of a model with its transform
struct ModelInstance {
    Model3D* model;           // Pointer to the model geometry
    VertexBufferObject vbo;   // GPU buffers associated with the model instance
    glm::mat4 transform;      // Transform matrix for this instance
    VkDescriptorSet descriptorSet;  // Descriptor set for textures/uniforms
    Buffer* uniformBuffer;    // Pointer to uniform buffer for this instance
    VkDescriptorSet shadowDescriptorSet;  // Descriptor set for shadow pass
    Buffer* shadowUniformBuffer;    // Pointer to uniform buffer for shadow pass
    const MaterialProperties* material;  // Pointer to material properties
    
    ModelInstance(Model3D* m, const VertexBufferObject& v, const glm::mat4& t, VkDescriptorSet ds, Buffer* ub, VkDescriptorSet sds, Buffer* sub, const MaterialProperties* mat = nullptr)
        : model(m), vbo(v), transform(t), descriptorSet(ds), uniformBuffer(ub), shadowDescriptorSet(sds), shadowUniformBuffer(sub), material(mat) {}
};

class ModelManager {
public:
    ModelManager() = default;
    
    // Add a model instance to be rendered
    void addInstance(Model3D* model, const VertexBufferObject& vbo, const glm::mat4& transform, 
                    VkDescriptorSet descriptorSet, Buffer* uniformBuffer,
                    VkDescriptorSet shadowDescriptorSet, Buffer* shadowUniformBuffer,
                    const MaterialProperties* material = nullptr);
    
    // Clear all instances
    void clear();
    
    // Get all instances for rendering
    const std::vector<ModelInstance>& getInstances() const { return instances; }
    
    // Get number of instances
    size_t getInstanceCount() const { return instances.size(); }
    
private:
    std::vector<ModelInstance> instances;
};
