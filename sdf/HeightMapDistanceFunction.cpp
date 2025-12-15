#include "HeightMapDistanceFunction.hpp"
#include "SDF.hpp"
#include "../math/HeightMap.hpp"


HeightMapDistanceFunction::HeightMapDistanceFunction(HeightMap * map) {
    this->map = map;
}

float HeightMapDistanceFunction::distance(const glm::vec3 &p, const Transformation &model)  {
    glm::vec3 len = map->getLength()*0.5f;
    glm::vec3 pos = p - map->getCenter();

    float sdf = map->distance(p);


    float d = SDF::opIntersection(
        SDF::box(pos+model.translate, len),
        sdf
    );

    return d;
}

SdfType HeightMapDistanceFunction::getType() const {
    return SdfType::HEIGHTMAP; 
}

glm::vec3 HeightMapDistanceFunction::getCenter(const Transformation &model) const {
    return this->map->getCenter();
}


