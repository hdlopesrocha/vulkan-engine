#include "PyramidDistanceFunction.hpp"
#include "../TriangleDistanceFunction.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
// The unit pyramid, in local space, before the model transform:
//   height 1, square base half-extent sqrt(0.5) (base area 1).
// Every vertex lies on the unit sphere centred at the local origin, which is
// what makes the conservative bounding sphere below safe for non-uniform scale.
std::array<glm::vec3, 5> unitPyramidVertices() {
    const float a = std::sqrt(0.5f);
    return {
        glm::vec3(0.0f, 1.0f, 0.0f),   // apex
        glm::vec3(-a, 0.0f, -a),
        glm::vec3(a, 0.0f, -a),
        glm::vec3(a, 0.0f, a),
        glm::vec3(-a, 0.0f, a)
    };
}
} // namespace

PyramidDistanceFunction::PyramidDistanceFunction(const Transformation &model, float bias)
    : SignedDistanceFunction(model.translate, model) {
    const std::array<glm::vec3, 5> local = unitPyramidVertices();
    std::array<glm::vec3, 5> world;
    for (size_t i = 0; i < local.size(); ++i) {
        // Inverse of the old local evaluation: translate + rotation * (local * scale).
        world[i] = m_model.translate + m_model.quaternion * (local[i] * m_model.scale);
    }

    // The five vertices (apex + four base corners) form four side triangles
    // and a square base split into two triangles.
    static constexpr int kFaces[6][3] = {
        {0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 1}, // sides
        {1, 2, 3}, {3, 4, 1}                         // base
    };
    centroid_ = (world[0] + world[1] + world[2] + world[3] + world[4]) / 5.0f;
    for (int f = 0; f < 6; ++f) {
        triangles_[f] = {world[kFaces[f][0]], world[kFaces[f][1]], world[kFaces[f][2]]};
        glm::vec3 n = glm::cross(triangles_[f][1] - triangles_[f][0],
                                 triangles_[f][2] - triangles_[f][0]);
        const float len = glm::length(n);
        n = (len > 1e-20f) ? n / len : glm::vec3(0.0f, 1.0f, 0.0f);
        // The vertex centroid is inside the convex pyramid; flip every face
        // normal that points toward it.
        if (glm::dot(n, centroid_ - triangles_[f][0]) > 0.0f) n = -n;
        faceNormals_[f] = n;
    }

    m_sphere = getSphere(model, bias);
}

float PyramidDistanceFunction::distance(const glm::vec3 &p) const {
    // Exact Euclidean distance to the actual world-space face triangles. The
    // old implementation evaluated the unit pyramid in scaled local space and
    // multiplied by min(scale), which is only a lower bound for non-uniform
    // scale; that made the field non-Euclidean and could under-subdivide or
    // clip the pyramid's apex.
    float dist = std::numeric_limits<float>::max();
    for (const Triangle &t : triangles_) {
        dist = std::min(dist,
                        TriangleDistanceFunction::triangleDistance(p, t[0], t[1], t[2]));
    }

    bool inside = true;
    for (int f = 0; f < 6 && inside; ++f) {
        inside = glm::dot(p - triangles_[f][0], faceNormals_[f]) <= 0.0f;
    }
    return inside ? -dist : dist;
}

BoundingSphere PyramidDistanceFunction::getSphere(const Transformation &model, float bias) const {
    // All unit-pyramid vertices lie on the unit sphere, so the transformed
    // vertices are within |scale| of the model centre even under non-uniform
    // scale and rotation. The previous isotropicSphere(|scale| * sqrt(0.5))
    // radius was too small for tall pyramids (large Y scale) and could make
    // check() report Disjoint for cells containing the apex.
    return BoundingSphere(getCenter(), glm::length(model.scale) + bias);
}
