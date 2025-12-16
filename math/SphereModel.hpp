#pragma once

#include "Mesh3D.hpp"

class SphereModel : public Mesh3D {
public:
    SphereModel() = default;
    void build(float radius = 0.5f, int longitudes = 16, int latitudes = 12, float texIndex = 0.0f);
};
