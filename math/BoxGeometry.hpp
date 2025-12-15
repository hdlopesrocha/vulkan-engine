#pragma once
#include "Geometry.hpp"
#include "BoundingBox.hpp"

class BoxGeometry : public Geometry {
public:
    BoxGeometry(const BoundingBox &box);
    void addTriangle(glm::vec3 a,glm::vec3 b, glm::vec3 c);
};

 
