#pragma once
#include "../sdf/SignedDistanceFunction.hpp"
#include "../sdf/WrappedSignedDistanceFunction.hpp"
#include "../math/BoundingBox.hpp"
#include "../math/Transformation.hpp"
#include "../space/Octree.hpp"
#include "../sdf/SDF.hpp"

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
