#include "HeightMap.hpp"
#include "HeightFunction.hpp"
#include "Math.hpp"

HeightMap::HeightMap(const HeightFunction &func, BoundingBox box, float step)  : BoundingBox(box), step(step), func(func){

}

float HeightMap::distance(const glm::vec3 p) const {
    float surfaceY = func.getHeightAt(p.x, p.z);
    return p.y - surfaceY;
}
