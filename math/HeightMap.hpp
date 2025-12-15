#ifndef HEIGHT_MAP_HPP
#define HEIGHT_MAP_HPP

#include "BoundingBox.hpp"
#include "HeightFunction.hpp"

class HeightMap: public BoundingBox  {
public:
    using BoundingBox::BoundingBox;
private:
    float step;
    const HeightFunction &func;
public:
    HeightMap(const HeightFunction &func, BoundingBox box, float step);
    float distance(const glm::vec3 p) const;
};

#endif
