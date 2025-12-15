#pragma once
#include <glm/glm.hpp>
#include "BoundingBox.hpp"

class HeightFunction {
public:
    virtual ~HeightFunction() {}
    virtual float getHeightAt(float x, float z) const = 0;
    glm::vec3 getNormal(float x, float z, float delta) const;
};

 
