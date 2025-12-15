#pragma once

#include "Model3D.hpp"

class SphereModel : public Model3D {
public:
    SphereModel() = default;
    void build(float radius = 0.5f, int longitudes = 16, int latitudes = 12, float texIndex = 0.0f);
};
