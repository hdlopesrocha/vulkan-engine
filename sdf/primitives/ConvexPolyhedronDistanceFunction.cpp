#include "ConvexPolyhedronDistanceFunction.hpp"
#include "TriangleDistanceFunction.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

std::vector<std::vector<uint32_t>> ConvexPolyhedronDistanceFunction::hullFaces(
    const std::vector<glm::vec3> &vertices) {
    std::vector<std::vector<uint32_t>> faces;
    if (vertices.size() < 4) return faces;

    // Regular-polyhedron coordinates are unit-sized, so an absolute tolerance
    // is enough to absorb the float error of the golden-ratio construction.
    constexpr float kPlaneEps = 1e-4f;
    std::vector<glm::vec3> planeNormals;
    std::vector<float> planeOffsets;

    // A supporting plane through three vertices that leaves every other vertex
    // on one side is a face plane of the convex hull. The primitives are
    // centred on the origin (d > 0 selects the outward normal), which also
    // lets the same test work for the pentagonal dodecahedron faces.
    for (size_t i = 0; i < vertices.size(); ++i) {
        for (size_t j = i + 1; j < vertices.size(); ++j) {
            for (size_t k = j + 1; k < vertices.size(); ++k) {
                glm::vec3 n = glm::cross(vertices[j] - vertices[i], vertices[k] - vertices[i]);
                const float len = glm::length(n);
                if (len < 1e-8f) continue;
                n /= len;
                float d = glm::dot(n, vertices[i]);
                if (d < 0.0f) {
                    n = -n;
                    d = -d;
                }

                bool supporting = true;
                for (const glm::vec3 &v : vertices) {
                    if (glm::dot(n, v) > d + kPlaneEps) {
                        supporting = false;
                        break;
                    }
                }
                if (!supporting) continue;

                bool duplicate = false;
                for (size_t m = 0; m < planeNormals.size(); ++m) {
                    if (glm::dot(n, planeNormals[m]) > 1.0f - 1e-5f &&
                        std::abs(d - planeOffsets[m]) < kPlaneEps) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) {
                    planeNormals.push_back(n);
                    planeOffsets.push_back(d);
                }
            }
        }
    }

    // Collect every vertex lying on each face plane and sort it cyclically so
    // the fan triangulation covers the polygon without crossing edges.
    for (size_t m = 0; m < planeNormals.size(); ++m) {
        std::vector<uint32_t> face;
        for (size_t v = 0; v < vertices.size(); ++v) {
            if (std::abs(glm::dot(planeNormals[m], vertices[v]) - planeOffsets[m]) <= kPlaneEps) {
                face.push_back(static_cast<uint32_t>(v));
            }
        }
        if (face.size() < 3) continue;

        glm::vec3 center(0.0f);
        for (uint32_t index : face) center += vertices[index];
        center /= static_cast<float>(face.size());

        glm::vec3 u = vertices[face[0]] - center;
        const float uLen = glm::length(u);
        if (uLen < 1e-8f) continue;
        u /= uLen;
        const glm::vec3 w = glm::normalize(glm::cross(planeNormals[m], u));

        std::sort(face.begin(), face.end(), [&](uint32_t ia, uint32_t ib) {
            const glm::vec3 da = vertices[ia] - center;
            const glm::vec3 db = vertices[ib] - center;
            return std::atan2(glm::dot(da, w), glm::dot(da, u)) <
                   std::atan2(glm::dot(db, w), glm::dot(db, u));
        });
        faces.push_back(std::move(face));
    }
    return faces;
}

ConvexPolyhedronDistanceFunction::ConvexPolyhedronDistanceFunction(
    const Transformation &model, float bias, const ConvexPolyhedronMesh &mesh)
    : SignedDistanceFunction(model.translate, model) {
    const std::vector<glm::vec3> &local = mesh.vertices;
    const std::vector<std::vector<uint32_t>> &faces = mesh.faces;
    if (local.empty() || faces.empty()) {
        m_sphere = getSphere(model, bias);
        return;
    }

    const glm::vec3 absScale = glm::abs(model.scale);
    const float minScale = glm::min(glm::min(absScale.x, absScale.y), absScale.z);

    std::vector<glm::vec3> world(local.size());
    for (size_t i = 0; i < local.size(); ++i) {
        world[i] = model.translate + model.quaternion * (local[i] * model.scale);
        maxLocalRadius_ = std::max(maxLocalRadius_, glm::length(local[i]));
    }

    const size_t faceCount = faces.size();
    faceNormals_.resize(faceCount);
    facePoints_.resize(faceCount);
    std::vector<float> faceOffsets(faceCount, 0.0f);
    float inradius = std::numeric_limits<float>::max();
    for (size_t f = 0; f < faceCount; ++f) {
        const std::vector<uint32_t> &face = faces[f];
        if (face.size() < 3) continue;

        glm::vec3 normal(0.0f);
        for (size_t t = 1; t + 1 < face.size(); ++t) {
            normal = glm::cross(world[face[t]] - world[face[0]],
                                world[face[t + 1]] - world[face[0]]);
            const float len = glm::length(normal);
            if (len > 1e-8f) {
                normal /= len;
                break;
            }
        }
        glm::vec3 faceCenter(0.0f);
        for (uint32_t index : face) faceCenter += world[index];
        faceCenter /= static_cast<float>(face.size());
        if (glm::dot(normal, faceCenter - model.translate) < 0.0f) normal = -normal;

        faceNormals_[f] = normal;
        faceOffsets[f] = glm::dot(normal, world[face[0]]);
        inradius = std::min(inradius, faceOffsets[f] - glm::dot(normal, model.translate));
    }

    // Three incident face planes meet at every vertex of the regular
    // polyhedra here. Solving them for offsets shifted inward by `bevel`
    // produces the inset vertices; the original planes then bound the rounded
    // zero set after `distance()` adds the bevel back.
    std::vector<std::array<int, 3>> vertexFaces(local.size(), {-1, -1, -1});
    std::vector<uint32_t> incidentCount(local.size(), 0u);
    for (size_t f = 0; f < faceCount; ++f) {
        for (uint32_t index : faces[f]) {
            if (incidentCount[index] < 3) vertexFaces[index][incidentCount[index]] = static_cast<int>(f);
            ++incidentCount[index];
        }
    }

    const float requested = std::max(bias, 0.0f);
    float bevel = std::min(requested, 0.15f * minScale);
    bevel = std::min(bevel, 0.5f * std::max(inradius, 0.0f));
    bevel = std::max(bevel, 0.0f);

    std::vector<glm::vec3> inset = world;
    if (bevel > 0.0f) {
        std::vector<glm::vec3> candidate = world;
        bool ok = true;
        for (size_t v = 0; v < world.size() && ok; ++v) {
            if (incidentCount[v] < 3) {
                ok = false;
                break;
            }
            const glm::vec3 &n0 = faceNormals_[vertexFaces[v][0]];
            const glm::vec3 &n1 = faceNormals_[vertexFaces[v][1]];
            const glm::vec3 &n2 = faceNormals_[vertexFaces[v][2]];
            // GLM's mat3 constructor takes columns, so transpose to put the
            // three plane normals in the rows of the system.
            const glm::mat3 A = glm::transpose(glm::mat3(n0, n1, n2));
            const float det = glm::determinant(A);
            if (std::abs(det) < 1e-8f) {
                ok = false;
                break;
            }
            const glm::vec3 b(faceOffsets[vertexFaces[v][0]] - bevel,
                              faceOffsets[vertexFaces[v][1]] - bevel,
                              faceOffsets[vertexFaces[v][2]] - bevel);
            candidate[v] = glm::inverse(A) * b;
        }
        if (ok) {
            inset = std::move(candidate);
        } else {
            bevel = 0.0f;
        }
    }

    size_t triangleCount = 0;
    for (const std::vector<uint32_t> &face : faces) {
        if (face.size() >= 3) triangleCount += face.size() - 2;
    }
    triangles_.clear();
    triangles_.reserve(triangleCount);
    for (size_t f = 0; f < faceCount; ++f) {
        const std::vector<uint32_t> &face = faces[f];
        if (face.size() >= 3) facePoints_[f] = inset[face[0]];
        for (size_t t = 1; t + 1 < face.size(); ++t) {
            triangles_.push_back({inset[face[0]], inset[face[t]], inset[face[t + 1]]});
        }
    }
    bevel_ = bevel;

    m_sphere = getSphere(model, bias);
}

float ConvexPolyhedronDistanceFunction::distance(const glm::vec3 &p) const {
    if (triangles_.empty()) return std::numeric_limits<float>::max();

    float dist = std::numeric_limits<float>::max();
    for (const Triangle &t : triangles_) {
        dist = std::min(dist,
                        TriangleDistanceFunction::triangleDistance(p, t[0], t[1], t[2]));
    }

    // The inset face planes are the original planes shifted inward by the
    // bevel, so the sign is the exact signed distance to the inset polyhedron.
    // Subtracting the bevel then rounds the edges/vertices while leaving the
    // face planes where they were.
    bool inside = true;
    for (size_t f = 0; f < faceNormals_.size() && inside; ++f) {
        inside = glm::dot(p - facePoints_[f], faceNormals_[f]) <= 0.0f;
    }
    const float d = inside ? -dist : dist;
    return d - bevel_;
}

BoundingSphere ConvexPolyhedronDistanceFunction::getSphere(const Transformation &model, float bias) const {
    // `maxLocalRadius_` is <= 1 for the unit regular polyhedra; the bevel only
    // moves the surface inward, so the sharp bounds stay conservative.
    return BoundingSphere(getCenter(), glm::length(model.scale) * maxLocalRadius_ + bias);
}
