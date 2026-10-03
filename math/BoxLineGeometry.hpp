#pragma once
#include "Geometry.hpp"
#include "BoundingBox.hpp"

class BoxLineGeometry : public Geometry {
public:
    BoxLineGeometry(const BoundingBox &box);
};

 
