#include "TetrahedronDistanceFunction.hpp"
#include "../TriangleDistanceFunction.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
// Regular tetrahedron in the conventional base-anchored orientation: an
// equilateral base on y=0 and the apex at y=1. The base circumradius is
// 1/sqrt(2), so every vertex stays within the unit sphere around the base
// centre (the model translate).
std::array<glm::vec3, 4> unitTetrahedronVertices() {
    const float r = 1.0f / std::sqrt(2.0f);
    const float s = std::sqrt(3.0f) * 0.5f * r;
    return {
        glm::vec3(0.0f, 1.0f, 0.0f),   // apex
        glm::vec3(0.0f, 0.0f, r),      // base
        glm::vec3(-s,   0.0f, -0.5f * r),
        glm::vec3( s,   0.0f, -0.5f * r)
    };
}
} // namespace

TetrahedronDistanceFunction::TetrahedronDistanceFunction(const Transformation &model, float bias)
    : SignedDistanceFunction(model.translate, model) {
    const std::array<glm::vec3, 4> local = unitTetrahedronVertices();
    std::array<glm::vec3, 4> world;
    for (size_t i = 0; i < local.size(); ++i) {
        world[i] = m_model.translate + m_model.quaternion * (local[i] * m_model.scale);
    }

    static constexpr int kFaces[4][3] = {
        {0, 1, 2}, {0, 2, 3}, {0, 3, 1}, {1, 2, 3}
    };
    centroid_ = (world[0] + world[1] + world[2] + world[3]) / 4.0f;
    for (int f = 0; f < 4; ++f) {
        triangles_[f] = {world[kFaces[f][0]], world[kFaces[f][1]], world[kFaces[f][2]]};
        glm::vec3 n = glm::cross(triangles_[f][1] - triangles_[f][0],
                                 triangles_[f][2] - triangles_[f][0]);
        const float len = glm::length(n);
        n = (len > 1e-20f) ? n / len : glm::vec3(0.0f, 1.0f, 0.0f);
        if (glm::dot(n, centroid_ - triangles_[f][0]) > 0.0f) n = -n;
        faceNormals_[f] = n;
    }

    m_sphere = getSphere(model, bias);
}

float TetrahedronDistanceFunction::distance(const glm::vec3 &p) const {
    float dist = std::numeric_limits<float>::max();
    for (const Triangle &t : triangles_) {
        dist = std::min(dist,
                        TriangleDistanceFunction::triangleDistance(p, t[0], t[1], t[2]));
    }

    bool inside = true;
    for (int f = 0; f < 4 && inside; ++f) {
        inside = glm::dot(p - triangles_[f][0], faceNormals_[f]) <= 0.0f;
    }
    return inside ? -dist : dist;
}

BoundingSphere TetrahedronDistanceFunction::getSphere(const Transformation &model, float bias) const {
    // The base circumradius is < 1 and the apex is at distance 1, so the
    // transformed vertices are within |scale| of the base centre under any
    // rotation and non-uniform scale.
    return BoundingSphere(getCenter(), glm::length(model.scale) + bias);
}
