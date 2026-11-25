#include "TetrahedronDistanceFunction.hpp"
#include "TriangleDistanceFunction.hpp"

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

// Faces are listed with the winding used by the convex inside test.
static constexpr int kFaces[4][3] = {
    {0, 1, 2}, {0, 2, 3}, {0, 3, 1}, {1, 2, 3}
};

// The three faces incident to each vertex. A vertex of the inset tetrahedron
// is the intersection of the three inset face planes incident to it.
static constexpr int kVertexFaces[4][3] = {
    {0, 1, 2}, // vertex 0 (apex)
    {0, 2, 3}, // vertex 1
    {0, 1, 3}, // vertex 2
    {1, 2, 3}  // vertex 3
};
} // namespace

TetrahedronDistanceFunction::TetrahedronDistanceFunction(const Transformation &model, float bias)
    : SignedDistanceFunction(model.translate, model) {
    const std::array<glm::vec3, 4> local = unitTetrahedronVertices();
    std::array<glm::vec3, 4> world;
    const glm::vec3 absScale = glm::abs(m_model.scale);
    const float minScale = glm::min(glm::min(absScale.x, absScale.y), absScale.z);
    for (size_t i = 0; i < local.size(); ++i) {
        world[i] = m_model.translate + m_model.quaternion * (local[i] * m_model.scale);
    }

    std::array<glm::vec3, 4> faceNormals;
    std::array<float, 4> faceOffsets;
    const glm::vec3 centroid = (world[0] + world[1] + world[2] + world[3]) * 0.25f;
    float inradius = std::numeric_limits<float>::max();
    for (int f = 0; f < 4; ++f) {
        const glm::vec3 &a = world[kFaces[f][0]];
        const glm::vec3 &b = world[kFaces[f][1]];
        const glm::vec3 &c = world[kFaces[f][2]];
        glm::vec3 n = glm::cross(b - a, c - a);
        const float len = glm::length(n);
        n = (len > 1e-20f) ? n / len : glm::vec3(0.0f, 1.0f, 0.0f);
        if (glm::dot(n, centroid - a) > 0.0f) n = -n;
        faceNormals[f] = n;
        faceOffsets[f] = glm::dot(n, a);
        inradius = std::min(inradius, -glm::dot(centroid - a, n));
    }

    // The octree stores the field at cell corners and reconstructs it with
    // trilinear interpolation. A perfectly sharp tetrahedron has a gradient
    // discontinuity along every edge, which the trilinear field cannot
    // represent: the reconstructed edge scallops by roughly one cell and the
    // interpolated normal can flip at grazing angles, so the ray march looks
    // stair-stepped. Bevel the edges by half the caller's sampling bias
    // (clamped to a small fraction of the shape) while keeping the four face
    // planes exactly where they are. The field then has a C1 edge of radius
    // `bevel`, which the cell grid can resolve; the four face planes stay
    // put and only the edges/vertices move inward, and bevel == 0 reproduces
    // the sharp exact tetrahedron (bias 0 is used by callers that do not
    // sample on a grid).
    //
    // One full sampling cell is the minimum rounding a trilinear cell can
    // resolve: a half-cell bevel still leaves the edge between two corner
    // samples, which is what produces the visible facets. The 15% of scale
    // and half-inradius caps keep the inset tetrahedron well-conditioned for
    // small/flat/non-uniform transforms.
    const float requested = std::max(bias, 0.0f);
    float bevel = std::min(requested, 0.15f * minScale);
    bevel = std::min(bevel, 0.5f * std::max(inradius, 0.0f));
    bevel = std::max(bevel, 0.0f);

    std::array<glm::vec3, 4> inset = world;
    if (bevel > 0.0f) {
        std::array<glm::vec3, 4> candidate;
        bool ok = true;
        for (int v = 0; v < 4 && ok; ++v) {
            const glm::vec3 &n0 = faceNormals[kVertexFaces[v][0]];
            const glm::vec3 &n1 = faceNormals[kVertexFaces[v][1]];
            const glm::vec3 &n2 = faceNormals[kVertexFaces[v][2]];
            // Solve [n0;n1;n2] * x = offset - bevel for the inset vertex.
            // GLM's mat3 constructor takes columns, so transpose to put the
            // three plane normals in the rows of the system.
            const glm::mat3 A = glm::transpose(glm::mat3(n0, n1, n2));
            const float det = glm::determinant(A);
            if (std::abs(det) < 1e-8f) {
                ok = false;
                break;
            }
            const glm::vec3 b(faceOffsets[kVertexFaces[v][0]] - bevel,
                              faceOffsets[kVertexFaces[v][1]] - bevel,
                              faceOffsets[kVertexFaces[v][2]] - bevel);
            candidate[v] = glm::inverse(A) * b;
        }
        if (ok) {
            inset = candidate;
        } else {
            bevel = 0.0f;
        }
    }

    for (int f = 0; f < 4; ++f) {
        triangles_[f] = {inset[kFaces[f][0]], inset[kFaces[f][1]], inset[kFaces[f][2]]};
    }
    faceNormals_ = faceNormals;
    bevel_ = bevel;

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
    const float d = inside ? -dist : dist;
    // The zero set is the inset tetrahedron offset outward by the edge bevel:
    // its faces are the original face planes and its edges/vertices are
    // rounded by `bevel_`.
    return d - bevel_;
}

BoundingSphere TetrahedronDistanceFunction::getSphere(const Transformation &model, float bias) const {
    // The base circumradius is < 1 and the apex is at distance 1, so the
    // transformed vertices are within |scale| of the base centre under any
    // rotation and non-uniform scale. The bevel only moves the surface
    // inward, so the sharp tetrahedron's bound stays conservative.
    return BoundingSphere(getCenter(), glm::length(model.scale) + bias);
}
