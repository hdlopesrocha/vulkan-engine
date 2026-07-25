#pragma once
#include "WrappedSignedDistanceFunction.hpp"
#include "OctreeDifferenceFunction.hpp"

class WrappedOctreeDifference : public WrappedSignedDistanceFunction {
private:
    BoundingBox box;
    public:
    WrappedOctreeDifference(OctreeDifferenceFunction * function_, float bias = 0.0f);

    ~WrappedOctreeDifference();

    BoundingBox getBox(float bias) const override;
        
    ContainmentType check(const BoundingCube &cube) const override;

    bool isContained(const BoundingCube &cube) const override;

    glm::vec3 getCenter(const Transformation &model) const override;


	const char* getLabel() const override;
};
