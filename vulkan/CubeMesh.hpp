#pragma once

#include "Model3D.hpp"

class CubeMesh : public Model3D {
public:
    CubeMesh() = default;
    
    // Build GPU buffers for the cube using provided VulkanApp helper methods
    // faceTexIndices: optional vector of 6 floats (one per cube face) to assign texIndex per face
    void build(VulkanApp* app, const std::vector<float>& faceTexIndices = {});
};
