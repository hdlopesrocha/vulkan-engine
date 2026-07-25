#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/BoundingBox.hpp"

class HeightMap;

class HeightMapDistanceFunction : public SignedDistanceFunction {
public:
    HeightMap * map;
    HeightMapDistanceFunction(HeightMap * map_, float bias);
    virtual ~HeightMapDistanceFunction() = default;
    float distance(const glm::vec3 &p, const Transformation &model) const override;
    glm::vec3 getCenter(const Transformation &model) const override;
    BoundingBox getBox(float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;

private:
    BoundingBox box;
};
