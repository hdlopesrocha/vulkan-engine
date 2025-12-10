#pragma once
#include "Model3D.hpp"

class PlaneMesh : public Model3D {
public:
    void build(VulkanApp* app, float width = 20.0f, float height = 20.0f, float texIndex = 0.0f);
};