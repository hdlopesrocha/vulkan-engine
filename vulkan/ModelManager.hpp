#pragma once

#include "../utils/Model3D.hpp"
#include "vulkan.hpp"
#include <vector>
#include <glm/glm.hpp>

#include "ModelInstance.hpp"

// Forward declaration
struct MaterialProperties;


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
