#pragma once

#include "vulkan.hpp"

class CubeMesh {
public:
    CubeMesh() = default;
    // build GPU buffers for the cube using provided VulkanApp helper methods
    // faceTexIndices: optional vector of 6 floats (one per cube face) to assign texIndex per face
    void build(VulkanApp* app, const std::vector<float>& faceTexIndices = {});

    // CubeMesh owns its buffers; destroy them with this call
    void destroy(VkDevice device) { vbo.destroy(device); }

    const VertexBufferObject& getVBO() const { return vbo; }

private:
    VertexBufferObject vbo{};
};
