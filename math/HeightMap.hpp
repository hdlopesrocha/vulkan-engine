#pragma once
#include "BoundingBox.hpp"
#include "HeightFunction.hpp"

class HeightMap: public BoundingBox  {
public:
    using BoundingBox::BoundingBox;
private:
    float step;
    const HeightFunction &func;
public:
    HeightMap(const HeightFunction &func_, BoundingBox box, float step_);
    float distance(const glm::vec3 p) const;
};

 
