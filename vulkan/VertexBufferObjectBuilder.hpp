#pragma once

#include "vulkan.hpp"
#include "../utils/Model3D.hpp"

class VertexBufferObjectBuilder {
public:
    // Create VBO from Model3D using VulkanApp; returns VertexBufferObject
    static VertexBufferObject create(VulkanApp* app, const Model3D& model);
};
