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
    void addInstance(const VertexBufferObject& vbo, const glm::mat4& transform);
    
    // Clear all instances
    void clear();
    
    // Get all instances for rendering
    const std::vector<ModelInstance>& getInstances() const { return instances; }
    
    // Get number of instances
    size_t getInstanceCount() const { return instances.size(); }
    
private:
    std::vector<ModelInstance> instances;
};
