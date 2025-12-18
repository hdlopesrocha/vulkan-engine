#pragma once

#include "Geometry.hpp"

class SphereModel : public Geometry {
public:
    SphereModel(float radius = 0.5f, int longitudes = 16, int latitudes = 12, float texIndex = 0.0f);
};
