#include "RoadDistanceFunction.hpp"
#include "SdfType.hpp"
#include <glm/glm.hpp>
#include <algorithm>

RoadDistanceFunction::RoadDistanceFunction(RoadSpline* spline, float width, float height,
                                            float tMin, float tMax,
                                            bool applyStartCap, bool applyEndCap)
    : SignedDistanceFunction(SdfType::ROAD)
    , m_spline(spline)
    , m_width(width)
    , m_height(height)
    , m_tMin(tMin)
    , m_tMax(tMax)
    , m_applyStartCap(applyStartCap)
    , m_applyEndCap(applyEndCap)
    , m_center(0.0f)
{
    const int numSamples = 32;
    glm::vec3 aabbMin(std::numeric_limits<float>::max());
    glm::vec3 aabbMax(std::numeric_limits<float>::lowest());
    for (int i = 0; i <= numSamples; ++i) {
        float t = m_tMin + (m_tMax - m_tMin) * ((float)i / (float)numSamples);
        glm::vec3 pt = m_spline->position(t);
        aabbMin = glm::min(aabbMin, pt);
        aabbMax = glm::max(aabbMax, pt);
    }
    m_center = (aabbMin + aabbMax) * 0.5f;
}

float RoadDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) {
    float t;
    glm::vec3 c, T, U;
    m_spline->closestPointInRange(p, m_tMin, m_tMax, t, c, T, U);

    glm::vec3 R = glm::cross(T, U);
    float rLen = glm::length(R);
    if (rLen < 1e-6f) {
        glm::vec3 ref = std::abs(T.y) < 0.9f
            ? glm::vec3(0.0f, 1.0f, 0.0f)
            : glm::vec3(1.0f, 0.0f, 0.0f);
        R = glm::normalize(glm::cross(T, ref));
        U = glm::normalize(glm::cross(R, T));
    } else {
        R /= rLen;
    }

    glm::vec3 q = p - c;
    float lateral  = glm::dot(q, R);
    float vertical = glm::dot(q, U);

    glm::vec2 d = glm::abs(glm::vec2(lateral, vertical))
                - glm::vec2(m_width * 0.5f, m_height * 0.5f);

    float road = glm::length(glm::max(d, glm::vec2(0.0f)))
               + std::min(std::max(d.x, d.y), 0.0f);

    if (m_applyStartCap) {
        float capStart = glm::length(p - m_spline->startPoint()) - m_width * 0.5f;
        road = std::max(road, capStart);
    }
    if (m_applyEndCap) {
        float capEnd   = glm::length(p - m_spline->endPoint())   - m_width * 0.5f;
        road = std::max(road, capEnd);
    }

    return road;
}

glm::vec3 RoadDistanceFunction::getCenter(const Transformation &model) const {
    return m_center;
}
