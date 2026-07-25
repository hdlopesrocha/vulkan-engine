#pragma once
#include "SignedDistanceFunction.hpp"
#include "RoadSpline.hpp"
#include <glm/glm.hpp>
#include "../math/BoundingSphere.hpp"

class RoadDistanceFunction : public SignedDistanceFunction {
public:
    RoadDistanceFunction(RoadSpline* spline, float width, float height,
                         float tMin = 0.0f, float tMax = 1.0f,
                         bool applyStartCap = false, bool applyEndCap = false,
                         const glm::vec3& sphereCenter = glm::vec3(0.0f), float sphereRadius = 0.0f,
                         const Transformation &model = Transformation(), float bias = 0.0f);
    virtual ~RoadDistanceFunction() = default;
    float distance(const glm::vec3 &p) const override;
    glm::vec3 getCenter() const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;

    RoadSpline* getSpline() const { return m_spline; }
    float getWidth() const { return m_width; }
    float getHeight() const { return m_height; }
    float getTMin() const { return m_tMin; }
    float getTMax() const { return m_tMax; }
    bool hasStartCap() const { return m_applyStartCap; }
    bool hasEndCap() const { return m_applyEndCap; }

private:
    RoadSpline* m_spline;
    float m_width;
    float m_height;
    float m_tMin, m_tMax;
    bool m_applyStartCap, m_applyEndCap;
    glm::vec3 m_center;
    glm::vec3 m_sphereCenter;
    float m_sphereRadius;
    BoundingSphere sphere;
};
