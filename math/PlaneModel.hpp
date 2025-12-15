#pragma once

#include "Model3D.hpp"

class PlaneModel : public Model3D {
public:
    PlaneModel() = default;
    void build(float width = 1.0f, float height = 1.0f, int hSegments = 1, int vSegments = 1, float texIndex = 0.0f);
};
