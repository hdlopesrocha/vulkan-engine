#pragma once
#include "Geometry.hpp"
#include "BoundingBox.hpp"

class BoxLineGeometry : public Geometry {
public:
    BoxLineGeometry(const BoundingBox &box);
    void addTriangle(glm::vec3 a,glm::vec3 b, glm::vec3 c);
};

 
