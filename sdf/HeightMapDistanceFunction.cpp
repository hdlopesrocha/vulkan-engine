#include "HeightMapDistanceFunction.hpp"
#include "SDF.hpp"
#include "../math/HeightMap.hpp"

HeightMapDistanceFunction::HeightMapDistanceFunction(HeightMap * map_, float bias)
    : SignedDistanceFunction(SdfType::HEIGHTMAP), map(map_)
    , box(getBox(bias)) {}

float HeightMapDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) const {
    glm::vec3 len = map->getLength()*0.5f;
    glm::vec3 pos = p - map->getCenter();

    float sdf = map->distance(p);

    float d = SDF::opIntersection(
        SDF::box(pos+model.translate, len),
        sdf
    );

    return d;
}

glm::vec3 HeightMapDistanceFunction::getCenter(const Transformation &model) const {
    return this->map->getCenter();
}

BoundingBox HeightMapDistanceFunction::getBox(float bias) const {
    return BoundingBox(map->getMin()-glm::vec3(bias), map->getMax()+glm::vec3(bias));
}

ContainmentType HeightMapDistanceFunction::check(const BoundingCube &cube) const {
    return box.test(cube);
}

bool HeightMapDistanceFunction::isContained(const BoundingCube &cube) const {
    return cube.contains(box);
}

const char* HeightMapDistanceFunction::getLabel() const {
    return "Height Map";
}
