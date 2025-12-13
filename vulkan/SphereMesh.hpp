#pragma once

#include "Model3D.hpp"

class SphereMesh : public Model3D {
public:
    SphereMesh() = default;
    // Build sphere vertex/index buffers. Slices=longitudinal segments, Stacks=latitudinal segments
    void build(VulkanApp* app, float radius = 1.0f, int slices = 32, int stacks = 16, float texIndex = 0.0f);
};
