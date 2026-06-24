#pragma once
#include <vector>
#include <glm/glm.hpp>
#include "../math/BoundingSphere.hpp"

class RoadSpline {
public:
    struct ControlPoint {
        glm::vec3 position;
        glm::vec3 up;
        ControlPoint() : position(0.0f), up(0.0f, 1.0f, 0.0f) {}
        ControlPoint(const glm::vec3& pos, const glm::vec3& upVec)
            : position(pos), up(upVec) {}
    };

    RoadSpline(const std::vector<ControlPoint>& controlPoints);

    glm::vec3 position(float t) const;
    glm::vec3 tangent(float t) const;
    glm::vec3 up(float t) const;

    void closestPoint(const glm::vec3& p, float& outT, glm::vec3& outC,
                      glm::vec3& outTan, glm::vec3& outUp) const;

    void closestPointInRange(const glm::vec3& p, float tMin, float tMax,
                             float& outT, glm::vec3& outC,
                             glm::vec3& outTan, glm::vec3& outUp) const;

    BoundingSphere boundingSphereInRange(float tMin, float tMax, float halfDiag) const;

    float totalLength() const { return m_totalLength; }
    glm::vec3 startPoint() const { return m_points.front().position; }
    glm::vec3 endPoint() const { return m_points.back().position; }
    glm::vec3 center() const { return m_center; }
    float boundingRadius() const { return m_boundingRadius; }

private:
    std::vector<ControlPoint> m_points;
    int m_numSegments;
    float m_totalLength;
    glm::vec3 m_center;
    float m_boundingRadius;

    glm::vec3 catmullRomPos(const glm::vec3& p0, const glm::vec3& p1,
                            const glm::vec3& p2, const glm::vec3& p3, float t) const;
    glm::vec3 catmullRomTan(const glm::vec3& p0, const glm::vec3& p1,
                            const glm::vec3& p2, const glm::vec3& p3, float t) const;
    glm::vec3 slerp(const glm::vec3& a, const glm::vec3& b, float t) const;
    void decomposeT(float t, int& segIndex, float& segT) const;
};
