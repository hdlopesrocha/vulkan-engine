#include "StarPolyhedronDistanceFunction.hpp"
#include "TriangleDistanceFunction.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace {
// Base face prepared in the unit mesh's local space.
struct StarFace {
    std::vector<uint32_t> indices;
    glm::vec3 center{0.0f};
    glm::vec3 apex{0.0f};
};
} // namespace

StarPolyhedronDistanceFunction::StarPolyhedronDistanceFunction(
    const Transformation &model, float bias, const ConvexPolyhedronMesh &mesh,
    float spikeHeight)
    : SignedDistanceFunction(model.translate, model) {
    if (mesh.vertices.size() < 4 || mesh.faces.empty()) {
        m_sphere = getSphere(model, bias);
        return;
    }

    const float spike = std::max(spikeHeight, 0.0f);

    // Build the star in the base mesh's local space first: one apex directly
    // above every face centroid. The mesh is centred on the origin, so the
    // centroid also orients the face normal outward.
    std::vector<StarFace> faces;
    faces.reserve(mesh.faces.size());
    float maxRadius = 0.0f;
    for (const glm::vec3 &v : mesh.vertices) {
        maxRadius = std::max(maxRadius, glm::length(v));
    }

    for (const std::vector<uint32_t> &face : mesh.faces) {
        if (face.size() < 3) continue;

        bool valid = true;
        glm::vec3 center(0.0f);
        for (uint32_t index : face) {
            if (index >= mesh.vertices.size()) {
                valid = false;
                break;
            }
            center += mesh.vertices[index];
        }
        if (!valid) continue;
        center /= static_cast<float>(face.size());

        glm::vec3 normal(0.0f);
        const glm::vec3 &first = mesh.vertices[face[0]];
        for (size_t t = 1; t + 1 < face.size(); ++t) {
            const glm::vec3 &b = mesh.vertices[face[t]];
            const glm::vec3 &c = mesh.vertices[face[t + 1]];
            const glm::vec3 candidate = glm::cross(b - first, c - first);
            const float len = glm::length(candidate);
            if (len > 1e-12f) {
                normal = candidate / len;
                break;
            }
        }
        if (normal == glm::vec3(0.0f)) continue;
        if (glm::dot(normal, center) < 0.0f) normal = -normal;

        StarFace prepared;
        prepared.indices = face;
        prepared.center = center;
        prepared.apex = center + normal * spike;
        maxRadius = std::max(maxRadius, glm::length(prepared.apex));
        faces.push_back(std::move(prepared));
    }

    if (faces.empty() || maxRadius < 1e-8f) {
        m_sphere = getSphere(model, bias);
        return;
    }

    // Normalize so the furthest apex lies on the unit sphere: the star then
    // has the same conservative bounding sphere and object footprint as the
    // regular polyhedron primitives regardless of the requested spike length.
    const float invRadius = 1.0f / maxRadius;
    auto toWorld = [&](const glm::vec3 &local) {
        return model.translate + model.quaternion * ((local * invRadius) * model.scale);
    };

    auto addPlane = [](ConvexPiece &piece, const glm::vec3 &a, const glm::vec3 &b,
                       const glm::vec3 &c, const glm::vec3 &interior) {
        glm::vec3 n = glm::cross(b - a, c - a);
        const float len = glm::length(n);
        if (len < 1e-12f) return false;
        n /= len;
        if (glm::dot(n, interior - a) > 0.0f) n = -n;
        piece.planeNormals.push_back(n);
        piece.planePoints.push_back(a);
        return true;
    };

    // The base polyhedron is part of the union: it fills the wedges between
    // adjacent face pyramids and supplies the concave seams.
    ConvexPiece base;
    base.center = model.translate;

    std::vector<glm::vec3> baseVertices(mesh.vertices.size());
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        baseVertices[i] = toWorld(mesh.vertices[i]);
        base.radius = std::max(base.radius, glm::length(baseVertices[i] - base.center));
    }

    for (const StarFace &face : faces) {
        std::vector<glm::vec3> polygon;
        polygon.reserve(face.indices.size());
        for (uint32_t index : face.indices) polygon.push_back(baseVertices[index]);

        // One half-space per polygonal face (the fan triangles share it).
        bool hasPlane = false;
        for (size_t t = 1; t + 1 < polygon.size() && !hasPlane; ++t) {
            hasPlane = addPlane(base, polygon[0], polygon[t], polygon[t + 1], base.center);
        }
        for (size_t t = 1; t + 1 < polygon.size(); ++t) {
            base.triangles.push_back({polygon[0], polygon[t], polygon[t + 1]});
        }
    }

    pieces_.reserve(1 + (spike > 0.0f ? faces.size() : 0));
    pieces_.push_back(std::move(base));

    if (spike > 0.0f) {
        for (const StarFace &face : faces) {
            std::vector<glm::vec3> polygon;
            polygon.reserve(face.indices.size());
            for (uint32_t index : face.indices) polygon.push_back(baseVertices[index]);

            const glm::vec3 apex = toWorld(face.apex);

            ConvexPiece piece;
            piece.center = apex;
            for (const glm::vec3 &v : polygon) piece.center += v;
            piece.center /= static_cast<float>(polygon.size() + 1);

            piece.radius = glm::length(apex - piece.center);
            for (const glm::vec3 &v : polygon) {
                piece.radius = std::max(piece.radius, glm::length(v - piece.center));
            }

            // Lateral faces: one triangle and one half-space per base edge.
            for (size_t i = 0; i < polygon.size(); ++i) {
                const glm::vec3 &b = polygon[i];
                const glm::vec3 &c = polygon[(i + 1) % polygon.size()];
                piece.triangles.push_back({apex, b, c});
                addPlane(piece, apex, b, c, piece.center);
            }

            // Pyramid base: coplanar with the original polyhedron face. Its
            // triangles are internal to the union, but keeping the pyramid a
            // closed convex piece makes its signed field exact on its own.
            bool hasBasePlane = false;
            for (size_t t = 1; t + 1 < polygon.size(); ++t) {
                piece.triangles.push_back({polygon[0], polygon[t], polygon[t + 1]});
                if (!hasBasePlane) {
                    hasBasePlane = addPlane(piece, polygon[0], polygon[t],
                                            polygon[t + 1], piece.center);
                }
            }

            pieces_.push_back(std::move(piece));
        }
    }

    // Bevel the sharp spikes and convex edges by one sampling cell, with the
    // same scale/inradius caps as the convex polyhedron primitives. The base
    // piece is always first, so its face planes give the world inradius.
    const ConvexPiece &basePiece = pieces_.front();
    const glm::vec3 absScale = glm::abs(model.scale);
    const float minScale = glm::min(glm::min(absScale.x, absScale.y), absScale.z);
    float inradius = std::numeric_limits<float>::max();
    for (size_t f = 0; f < basePiece.planeNormals.size(); ++f) {
        inradius = std::min(inradius,
                            glm::dot(basePiece.planeNormals[f],
                                     basePiece.planePoints[f] - basePiece.center));
    }
    if (!std::isfinite(inradius)) inradius = 0.0f;

    float bevel = std::min(std::max(bias, 0.0f), 0.15f * minScale);
    bevel = std::min(bevel, 0.5f * std::max(inradius, 0.0f));
    bevel_ = std::max(bevel, 0.0f);

    m_sphere = getSphere(model, bias);
}

float StarPolyhedronDistanceFunction::distance(const glm::vec3 &p) const {
    if (pieces_.empty()) return std::numeric_limits<float>::max();

    float nearest = std::numeric_limits<float>::max();
    bool inside = false;
    for (const ConvexPiece &piece : pieces_) {
        // Outside the piece its distance is at least `bound`, so its dilated
        // field (distance - bevel) is at least `bound - bevel`. Skip the piece
        // when even that cannot beat the closest boundary found so far. A
        // containing piece has a negative bound, so it is never skipped.
        const float bound = glm::length(p - piece.center) - piece.radius;
        if (bound - bevel_ > nearest) continue;

        bool contained = !piece.planeNormals.empty();
        for (size_t f = 0; f < piece.planeNormals.size() && contained; ++f) {
            contained = glm::dot(p - piece.planePoints[f], piece.planeNormals[f]) <= 0.0f;
        }

        float dist = std::numeric_limits<float>::max();
        for (const Triangle &t : piece.triangles) {
            dist = std::min(dist,
                            TriangleDistanceFunction::triangleDistance(p, t[0], t[1], t[2]));
        }

        // Per-piece bevel: subtracting it dilates the convex piece by one
        // sampling cell, which rounds its spike tip and convex edges. The
        // union is then the minimum over the pieces; taking the smallest
        // magnitude (rather than the signed minimum) keeps the interior
        // conservative, so trilinear reconstruction never pushes the surface
        // outside the star at the concave seams between adjacent spikes.
        const float d = (contained ? -dist : dist) - bevel_;
        inside = inside || d < 0.0f;
        nearest = std::min(nearest, std::abs(d));
    }

    return inside ? -nearest : nearest;
}

BoundingSphere StarPolyhedronDistanceFunction::getSphere(const Transformation &model, float bias) const {
    // The star is normalized so every local vertex lies on the unit sphere;
    // length(scale) remains a conservative bound for non-uniform scale and
    // rotation. The bevel only moves the surface inward.
    return BoundingSphere(getCenter(), glm::length(model.scale) + bias);
}
