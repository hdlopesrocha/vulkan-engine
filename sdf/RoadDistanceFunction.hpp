#pragma once
#include "SignedDistanceFunction.hpp"
#include "RoadSpline.hpp"
#include <glm/glm.hpp>

class RoadDistanceFunction : public SignedDistanceFunction {
public:
    RoadDistanceFunction(RoadSpline* spline, float width, float height,
                         float tMin = 0.0f, float tMax = 1.0f,
                         bool applyStartCap = false, bool applyEndCap = false);
    virtual ~RoadDistanceFunction() = default;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override;
    glm::vec3 getCenter(const Transformation &model) const override;

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
};
