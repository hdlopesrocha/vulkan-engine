#pragma once
#include "../sdf/WrappedSignedDistanceFunction.hpp"
#include "OctreeDifferenceFunction.hpp"

class WrappedOctreeDifference : public WrappedSignedDistanceFunction {
    public:
    WrappedOctreeDifference(OctreeDifferenceFunction * function);

    ~WrappedOctreeDifference();

    BoundingBox getBox(float bias) const;
        
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;

    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;

    glm::vec3 getCenter(const Transformation &model) const override;
	float getLength(const Transformation &model, float bias) const override;

    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;

	const char* getLabel() const override;
};
