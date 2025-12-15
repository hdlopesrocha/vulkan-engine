#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>

class HeightMap;

class HeightMapDistanceFunction : public SignedDistanceFunction {
public:
    HeightMap * map;
    HeightMapDistanceFunction(HeightMap * map);
    virtual ~HeightMapDistanceFunction() = default;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override;
    glm::vec3 getCenter(const Transformation &model) const override;
};

 