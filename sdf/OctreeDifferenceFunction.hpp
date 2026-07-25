#pragma once
#include "SignedDistanceFunction.hpp"
#include "../math/BoundingBox.hpp"
#include "../math/Transformation.hpp"
#include "../space/Octree.hpp"
#include "SDF.hpp"

class OctreeDifferenceFunction : public SignedDistanceFunction {
    public:
    Octree * tree;
    BoundingBox box;
	float bias;
    OctreeDifferenceFunction(Octree * tree_, BoundingBox box_, float bias_);
    float distance(const glm::vec3 &p, const Transformation &model) override;
    glm::vec3 getCenter(const Transformation &model) const override;
    BoundingBox getBox(float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;

private:
    BoundingBox m_box;
};
