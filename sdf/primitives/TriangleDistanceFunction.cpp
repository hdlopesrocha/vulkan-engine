#include "TriangleDistanceFunction.hpp"

#include <algorithm>
#include <cmath>

float TriangleDistanceFunction::segmentDistance(const glm::vec3 &p, const glm::vec3 &a, const glm::vec3 &b) {
    glm::vec3 pa = p - a;
    glm::vec3 ba = b - a;
    float t = glm::clamp(glm::dot(pa, ba) / glm::dot(ba, ba), 0.0f, 1.0f);
    return glm::length(pa - ba * t);
}

float TriangleDistanceFunction::triangleDistance(const glm::vec3 &p, const glm::vec3 &a,
                                                 const glm::vec3 &b, const glm::vec3 &c) {
    // Plane normal.
    glm::vec3 ab = b - a;
    glm::vec3 ac = c - a;
    glm::vec3 n = glm::cross(ab, ac);
    float nlen2 = glm::dot(n, n);
    if (nlen2 < 1e-12f) {
        // Degenerate triangle -> fall back to the edge distances.
        return std::min({segmentDistance(p, a, b), segmentDistance(p, b, c), segmentDistance(p, c, a)});
    }
    glm::vec3 nn = n / glm::sqrt(nlen2);
    float distPlane = glm::dot(p - a, nn);
    glm::vec3 proj = p - distPlane * nn;

    // Inside-triangle test (same side of every edge).
    glm::vec3 ap = proj - a;
    glm::vec3 bp = proj - b;
    glm::vec3 cp = proj - c;
    if (glm::dot(glm::cross(ab, ap), n) >= 0.0f &&
        glm::dot(glm::cross(c - b, bp), n) >= 0.0f &&
        glm::dot(glm::cross(a - c, cp), n) >= 0.0f) {
        return std::abs(distPlane); // perpendicular foot lies inside the triangle
    }

    // Otherwise the closest point is on one of the edges.
    return std::min({segmentDistance(p, a, b), segmentDistance(p, b, c), segmentDistance(p, c, a)});
}

float TriangleDistanceFunction::thickTriangleDistance(const glm::vec3 &p, const glm::vec3 &a,
                                                      const glm::vec3 &b, const glm::vec3 &c,
                                                      float halfThick) {
    glm::vec3 n = glm::cross(b - a, c - a);
    float nLen = glm::length(n);
    if (nLen < 1e-10f)
        return segmentDistance(p, a, b) - halfThick;
    n /= nLen;
    float planeDist = glm::dot(p - a, n);
    glm::vec3 proj = p - planeDist * n;
    glm::vec3 v0 = glm::cross(b - a, proj - a);
    glm::vec3 v1 = glm::cross(c - b, proj - b);
    glm::vec3 v2 = glm::cross(a - c, proj - c);
    float d0 = glm::dot(v0, n), d1 = glm::dot(v1, n), d2 = glm::dot(v2, n);
    bool inside = (d0 >= 0.0f && d1 >= 0.0f && d2 >= 0.0f) ||
                  (d0 <= 0.0f && d1 <= 0.0f && d2 <= 0.0f);
    if (inside)
        return glm::abs(planeDist) - halfThick;
    float dEdge = glm::min(segmentDistance(proj, a, b),
                  glm::min(segmentDistance(proj, b, c), segmentDistance(proj, c, a)));
    float dPlane = glm::max(glm::abs(planeDist) - halfThick, 0.0f);
    return glm::sqrt(dEdge * dEdge + dPlane * dPlane);
}
