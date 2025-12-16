#pragma once

#include "vulkan.hpp"
#include "../math/Mesh3D.hpp"

class VertexBufferObjectBuilder {
public:
    // Create VBO from Model3D using VulkanApp; returns VertexBufferObject
    static VertexBufferObject create(VulkanApp* app, const Mesh3D& model);
};
