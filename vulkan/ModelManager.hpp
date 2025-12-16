#pragma once

#include "../math/Mesh3D.hpp"
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
    void addInstance(Mesh3D* model, const VertexBufferObject& vbo, const glm::mat4& transform, 
                    VkDescriptorSet descriptorSet, Buffer* uniformBuffer,
                    VkDescriptorSet shadowDescriptorSet, Buffer* shadowUniformBuffer);
    
    // Clear all instances
    void clear();
    
    // Get all instances for rendering
    std::vector<ModelInstance>& getInstances() { return instances; }
    
    // Get number of instances
    size_t getInstanceCount() const { return instances.size(); }
    
private:
    std::vector<ModelInstance> instances;
};
