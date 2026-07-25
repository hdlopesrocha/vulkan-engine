#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/Transformation.hpp"
#include "../math/BoundingSphere.hpp"
#include "SDF.hpp"

class TriangleStripDistanceFunction : public SignedDistanceFunction {
public:
    glm::vec3 v0, v1, v2, v3;
    float halfThick;
    TriangleStripDistanceFunction(const glm::vec3& v0_, const glm::vec3& v1_,
                                  const glm::vec3& v2_, const glm::vec3& v3_,
                                  float halfThick_,
                                  const glm::vec3& sphereCenter = glm::vec3(0.0f), float sphereRadius = 0.0f,
                                  const Transformation &model = Transformation(), float bias = 0.0f);
    virtual ~TriangleStripDistanceFunction() = default;
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
