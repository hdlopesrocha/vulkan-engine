#pragma once
#include "vulkan.hpp"

class PlaneMesh {
private:
    VertexBufferObject vbo;

public:
    void build(VulkanApp* app, float width = 20.0f, float height = 20.0f, float texIndex = 0.0f);
    const VertexBufferObject& getVBO() const { return vbo; }
    void cleanup(VulkanApp* app);
};