#include "RoadSpline.hpp"
#include <glm/gtc/constants.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

RoadSpline::RoadSpline(const std::vector<ControlPoint>& controlPoints)
    : m_points(controlPoints)
    , m_totalLength(0.0f)
    , m_center(0.0f)
    , m_boundingRadius(0.0f)
{
    m_numSegments = std::max(0, (int)m_points.size() - 1);

    for (const auto& cp : m_points)
        m_center += cp.position;
    if (!m_points.empty())
        m_center /= (float)m_points.size();

    for (const auto& cp : m_points)
        m_boundingRadius = std::max(m_boundingRadius,
            glm::distance(cp.position, m_center));

    const int lengthSamples = 1000;
    glm::vec3 prev = position(0.0f);
    for (int i = 1; i <= lengthSamples; ++i) {
        float t = (float)i / (float)lengthSamples;
        glm::vec3 curr = position(t);
        m_totalLength += glm::distance(curr, prev);
        prev = curr;
    }
}

void RoadSpline::decomposeT(float t, int& segIndex, float& segT) const {
    if (m_numSegments < 1) {
        segIndex = 0;
        segT = 0.0f;
        return;
    }
    float raw = t * (float)m_numSegments;
    if (raw >= (float)m_numSegments) {
        segIndex = m_numSegments - 1;
        segT = 1.0f;
    } else {
        segIndex = (int)raw;
        segT = raw - (float)segIndex;
    }
}

glm::vec3 RoadSpline::catmullRomPos(const glm::vec3& p0, const glm::vec3& p1,
                                     const glm::vec3& p2, const glm::vec3& p3,
                                     float t) const {
    float t2 = t * t;
    float t3 = t2 * t;
    return 0.5f * ((2.0f * p1) +
                   (-p0 + p2) * t +
                   (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
                   (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
}

glm::vec3 RoadSpline::catmullRomTan(const glm::vec3& p0, const glm::vec3& p1,
                                     const glm::vec3& p2, const glm::vec3& p3,
                                     float t) const {
    float t2 = t * t;
    return 0.5f * ((-p0 + p2) +
                   2.0f * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t +
                   3.0f * (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t2);
}

glm::vec3 RoadSpline::slerp(const glm::vec3& a, const glm::vec3& b, float t) const {
    float cosOmega = glm::clamp(glm::dot(a, b), -1.0f, 1.0f);
    float omega = std::acos(cosOmega);
    if (omega < 1e-6f)
        return glm::mix(a, b, t);
    float sinOmega = std::sin(omega);
    return std::sin((1.0f - t) * omega) / sinOmega * a +
           std::sin(t * omega) / sinOmega * b;
}

glm::vec3 RoadSpline::position(float t) const {
    if (m_numSegments < 1)
        return m_points.empty() ? glm::vec3(0.0f) : m_points[0].position;

    int segIndex;
    float segT;
    decomposeT(t, segIndex, segT);

    int i0 = std::max(0, segIndex - 1);
    int i1 = segIndex;
    int i2 = segIndex + 1;
    int i3 = std::min(segIndex + 2, (int)m_points.size() - 1);

    return catmullRomPos(m_points[i0].position, m_points[i1].position,
                         m_points[i2].position, m_points[i3].position, segT);
}

glm::vec3 RoadSpline::tangent(float t) const {
    if (m_numSegments < 1)
        return glm::vec3(1.0f, 0.0f, 0.0f);

    int segIndex;
    float segT;
    decomposeT(t, segIndex, segT);

    int i0 = std::max(0, segIndex - 1);
    int i1 = segIndex;
    int i2 = segIndex + 1;
    int i3 = std::min(segIndex + 2, (int)m_points.size() - 1);

    glm::vec3 tan = catmullRomTan(m_points[i0].position, m_points[i1].position,
                                   m_points[i2].position, m_points[i3].position, segT);
    float len = glm::length(tan);
    return len > 1e-8f ? tan / len : glm::vec3(1.0f, 0.0f, 0.0f);
}

glm::vec3 RoadSpline::up(float t) const {
    if (m_numSegments < 1)
        return m_points.empty() ? glm::vec3(0.0f, 1.0f, 0.0f) : m_points[0].up;

    int segIndex;
    float segT;
    decomposeT(t, segIndex, segT);

    int i1 = segIndex;
    int i2 = std::min(segIndex + 1, (int)m_points.size() - 1);

    if (i1 == i2)
        return glm::normalize(m_points[i1].up);

    return glm::normalize(slerp(m_points[i1].up, m_points[i2].up, segT));
}

void RoadSpline::closestPoint(const glm::vec3& p, float& outT, glm::vec3& outC,
                               glm::vec3& outTan, glm::vec3& outUp) const {
    closestPointInRange(p, 0.0f, 1.0f, outT, outC, outTan, outUp);
}

void RoadSpline::closestPointInRange(const glm::vec3& p, float tMin, float tMax,
                                      float& outT, glm::vec3& outC,
                                      glm::vec3& outTan, glm::vec3& outUp) const {
    const int numSamples = 64;
    float bestT = tMin;
    float bestDist = std::numeric_limits<float>::max();

    for (int i = 0; i <= numSamples; ++i) {
        float t = tMin + (tMax - tMin) * ((float)i / (float)numSamples);
        glm::vec3 pt = position(t);
        glm::vec3 diff = p - pt;
        float d = glm::dot(diff, diff);
        if (d < bestDist) {
            bestDist = d;
            bestT = t;
        }
    }

    float range = tMax - tMin;
    float a = std::max(tMin, bestT - range / (float)numSamples);
    float b = std::min(tMax, bestT + range / (float)numSamples);

    const float phi = 1.618033988749895f;
    const float invPhi = 1.0f / phi;

    float t1 = b - (b - a) * invPhi;
    float t2 = a + (b - a) * invPhi;
    auto dist2 = [&](float t) {
        glm::vec3 diff = p - position(t);
        return glm::dot(diff, diff);
    };
    float d1 = dist2(t1);
    float d2 = dist2(t2);

    for (int iter = 0; iter < 12; ++iter) {
        if (d1 < d2) {
            b = t2;
            t2 = t1;
            d2 = d1;
            t1 = b - (b - a) * invPhi;
            d1 = dist2(t1);
        } else {
            a = t1;
            t1 = t2;
            d1 = d2;
            t2 = a + (b - a) * invPhi;
            d2 = dist2(t2);
        }
    }

    float finalT = (a + b) * 0.5f;
    outT = finalT;
    outC = position(finalT);
    outTan = tangent(finalT);
    outUp = up(finalT);
}

BoundingSphere RoadSpline::boundingSphereInRange(float tMin, float tMax, float halfDiag) const {
    const int numSamples = 32;
    glm::vec3 aabbMin(std::numeric_limits<float>::max());
    glm::vec3 aabbMax(std::numeric_limits<float>::lowest());

    for (int i = 0; i <= numSamples; ++i) {
        float t = tMin + (tMax - tMin) * ((float)i / (float)numSamples);
        glm::vec3 pt = position(t);
        aabbMin = glm::min(aabbMin, pt);
        aabbMax = glm::max(aabbMax, pt);
    }

    aabbMin -= glm::vec3(halfDiag);
    aabbMax += glm::vec3(halfDiag);

    glm::vec3 center = (aabbMin + aabbMax) * 0.5f;
    float radius = glm::distance(center, aabbMax);

    return BoundingSphere(center, radius);
}
