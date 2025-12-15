#pragma once

#include "Model3D.hpp"
#include <vector>

class CubeModel : public Model3D {
public:
    CubeModel() = default;
    
    // Build geometry (no Vulkan dependency)
    void build(const std::vector<float>& faceTexIndices = {});
};
