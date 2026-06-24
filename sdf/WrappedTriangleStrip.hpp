#pragma once
#include "WrappedSignedDistanceFunction.hpp"
#include "TriangleStripDistanceFunction.hpp"
#include "../math/BoundingSphere.hpp"

class WrappedTriangleStrip : public WrappedSignedDistanceFunction {
public:
    WrappedTriangleStrip(TriangleStripDistanceFunction* function,
                         const glm::vec3& sphereCenter, float sphereRadius);
    ~WrappedTriangleStrip();
    float distance(const glm::vec3 &p, const Transformation &model) override;
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;
    float getLength(const Transformation &model, float bias) const override;
    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;
    const char* getLabel() const override;

private:
    glm::vec3 m_sphereCenter;
    float m_sphereRadius;
};
