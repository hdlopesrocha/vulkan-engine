#pragma once

#include "Mesh3D.hpp"
#include <vector>

class CubeModel : public Mesh3D {
public:
    CubeModel() = default;
    
    // Build geometry (no Vulkan dependency)
    void build(const std::vector<float>& faceTexIndices = {});
};
