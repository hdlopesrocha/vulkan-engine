#pragma once
#include "WrappedSignedDistanceFunction.hpp"
#include "TriangleStripDistanceFunction.hpp"
#include "../math/BoundingSphere.hpp"

class WrappedTriangleStrip : public WrappedSignedDistanceFunction {
public:
    WrappedTriangleStrip(TriangleStripDistanceFunction* function_,
                         const glm::vec3& sphereCenter, float sphereRadius,
                         const Transformation &model, float bias);
    ~WrappedTriangleStrip();
    float distance(const glm::vec3 &p, const Transformation &model) override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;

private:
    glm::vec3 m_sphereCenter;
    float m_sphereRadius;
    BoundingSphere sphere;
};
