#pragma once

#include "Model3D.hpp"
#include <vector>

class CubeMesh : public Model3D {
public:
    CubeMesh() = default;
    
    // Build geometry (no Vulkan dependency)
    void build(const std::vector<float>& faceTexIndices = {});
};
