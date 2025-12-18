#pragma once

#include "vulkan.hpp"
#include "../math/Geometry.hpp"

class VertexBufferObjectBuilder {
public:
    // Create VBO from Model3D using VulkanApp; returns VertexBufferObject
    static VertexBufferObject create(VulkanApp* app, const Geometry& model);
};
