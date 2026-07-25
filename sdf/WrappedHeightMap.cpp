#include "WrappedHeightMap.hpp"
#include "../math/HeightMap.hpp"

WrappedHeightMap::WrappedHeightMap(HeightMapDistanceFunction * function_, float bias)
    : WrappedSignedDistanceFunction(function_)
    , box(getBox(bias)) {

}

WrappedHeightMap::~WrappedHeightMap() {

}

BoundingBox WrappedHeightMap::getBox(float bias) const {
    HeightMapDistanceFunction * f = (HeightMapDistanceFunction*) function;
    return BoundingBox(f->map->getMin()-glm::vec3(bias), f->map->getMax()+glm::vec3(bias));
}
    
ContainmentType WrappedHeightMap::check(const BoundingCube &cube) const {
    return box.test(cube);
};

bool WrappedHeightMap::isContained(const BoundingCube &cube) const {
    return cube.contains(box);
};


const char* WrappedHeightMap::getLabel() const {
    return "Height Map";
}