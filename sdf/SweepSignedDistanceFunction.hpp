#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/Transformation.hpp"
#include "../math/BoundingSphere.hpp"
#include "../math/BoundingBox.hpp"

template<typename T>
class SweepSignedDistanceFunction : public SignedDistanceFunction {
    T function1;
    T function2;
    glm::vec3 posA;
    glm::vec3 posB;
    BoundingSphere sphere;
public:
    SweepSignedDistanceFunction(const T &f1, const T &f2,
                                const Transformation &model, float bias);
    virtual ~SweepSignedDistanceFunction() = default;

    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    BoundingBox getBox(float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;
    glm::vec3 getCenter() const override;
};
