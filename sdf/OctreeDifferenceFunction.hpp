#pragma once
#include "SignedDistanceFunction.hpp"
#include "WrappedSignedDistanceFunction.hpp"
#include "../math/BoundingBox.hpp"
#include "../math/Transformation.hpp"
#include "../space/Octree.hpp"
#include "SDF.hpp"

class OctreeDifferenceFunction : public SignedDistanceFunction {
    public:
    Octree * tree;
    BoundingBox box;
	float bias;
    OctreeDifferenceFunction(Octree * tree, BoundingBox box, float bias);
    float distance(const glm::vec3 &p, const Transformation &model) override;
	SdfType getType() const override;
	glm::vec3 getCenter(const Transformation &model) const override;

};
