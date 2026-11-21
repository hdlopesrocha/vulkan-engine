#include "RegularPolyhedronMeshes.hpp"

#include <cmath>
#include <utility>
#include <vector>

namespace {
// Normalizes the vertex set to unit circumradius and derives the hull faces.
ConvexPolyhedronMesh buildMesh(std::vector<glm::vec3> vertices, float circumradius) {
    for (glm::vec3 &v : vertices) v /= circumradius;
    ConvexPolyhedronMesh mesh;
    mesh.vertices = std::move(vertices);
    mesh.faces = ConvexPolyhedronDistanceFunction::hullFaces(mesh.vertices);
    return mesh;
}
} // namespace

const ConvexPolyhedronMesh &RegularPolyhedronMeshes::tetrahedron() {
    static const ConvexPolyhedronMesh mesh = [] {
        // Half of the cube's alternating corners: a regular tetrahedron with
        // circumradius sqrt(3) before normalization.
        std::vector<glm::vec3> vertices = {
            { 1.0f,  1.0f,  1.0f},
            { 1.0f, -1.0f, -1.0f},
            {-1.0f,  1.0f, -1.0f},
            {-1.0f, -1.0f,  1.0f}
        };
        return buildMesh(std::move(vertices), std::sqrt(3.0f));
    }();
    return mesh;
}

const ConvexPolyhedronMesh &RegularPolyhedronMeshes::hexahedron() {
    static const ConvexPolyhedronMesh mesh = [] {
        std::vector<glm::vec3> vertices;
        vertices.reserve(8);
        for (int sx = -1; sx <= 1; sx += 2) {
            for (int sy = -1; sy <= 1; sy += 2) {
                for (int sz = -1; sz <= 1; sz += 2) {
                    vertices.emplace_back(static_cast<float>(sx),
                                          static_cast<float>(sy),
                                          static_cast<float>(sz));
                }
            }
        }
        return buildMesh(std::move(vertices), std::sqrt(3.0f));
    }();
    return mesh;
}

const ConvexPolyhedronMesh &RegularPolyhedronMeshes::octahedron() {
    static const ConvexPolyhedronMesh mesh = [] {
        // The six axis vertices already lie on the unit sphere.
        std::vector<glm::vec3> vertices = {
            { 1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f},
            { 0.0f, 1.0f, 0.0f}, { 0.0f, -1.0f, 0.0f},
            { 0.0f, 0.0f, 1.0f}, { 0.0f, 0.0f, -1.0f}
        };
        return buildMesh(std::move(vertices), 1.0f);
    }();
    return mesh;
}

const ConvexPolyhedronMesh &RegularPolyhedronMeshes::dodecahedron() {
    static const ConvexPolyhedronMesh mesh = [] {
        // Golden-ratio coordinate set: the eight cube corners (±1,±1,±1) plus
        // the twelve cyclic permutations of (0, ±1/φ, ±φ). Every vertex has
        // radius sqrt(3), so dividing by that puts the circumradius at 1.
        const float phi = (1.0f + std::sqrt(5.0f)) * 0.5f;
        const float invPhi = 1.0f / phi;
        std::vector<glm::vec3> vertices;
        vertices.reserve(20);
        for (int sx = -1; sx <= 1; sx += 2) {
            for (int sy = -1; sy <= 1; sy += 2) {
                for (int sz = -1; sz <= 1; sz += 2) {
                    vertices.emplace_back(static_cast<float>(sx),
                                          static_cast<float>(sy),
                                          static_cast<float>(sz));
                }
            }
        }
        for (int sy = -1; sy <= 1; sy += 2) {
            for (int sz = -1; sz <= 1; sz += 2) {
                vertices.emplace_back(0.0f, static_cast<float>(sy) * invPhi,
                                      static_cast<float>(sz) * phi);
            }
        }
        for (int sx = -1; sx <= 1; sx += 2) {
            for (int sy = -1; sy <= 1; sy += 2) {
                vertices.emplace_back(static_cast<float>(sx) * invPhi,
                                      static_cast<float>(sy) * phi, 0.0f);
            }
        }
        for (int sx = -1; sx <= 1; sx += 2) {
            for (int sz = -1; sz <= 1; sz += 2) {
                vertices.emplace_back(static_cast<float>(sx) * phi, 0.0f,
                                      static_cast<float>(sz) * invPhi);
            }
        }
        return buildMesh(std::move(vertices), std::sqrt(3.0f));
    }();
    return mesh;
}

const ConvexPolyhedronMesh &RegularPolyhedronMeshes::icosahedron() {
    static const ConvexPolyhedronMesh mesh = [] {
        // Twelve cyclic permutations of (0, ±1, ±φ); every vertex has radius
        // sqrt(1 + φ²), so dividing by that puts the circumradius at 1.
        const float phi = (1.0f + std::sqrt(5.0f)) * 0.5f;
        std::vector<glm::vec3> vertices;
        vertices.reserve(12);
        for (int sy = -1; sy <= 1; sy += 2) {
            for (int sz = -1; sz <= 1; sz += 2) {
                vertices.emplace_back(0.0f, static_cast<float>(sy),
                                      static_cast<float>(sz) * phi);
            }
        }
        for (int sx = -1; sx <= 1; sx += 2) {
            for (int sy = -1; sy <= 1; sy += 2) {
                vertices.emplace_back(static_cast<float>(sx),
                                      static_cast<float>(sy) * phi, 0.0f);
            }
        }
        for (int sx = -1; sx <= 1; sx += 2) {
            for (int sz = -1; sz <= 1; sz += 2) {
                vertices.emplace_back(static_cast<float>(sx) * phi, 0.0f,
                                      static_cast<float>(sz));
            }
        }
        return buildMesh(std::move(vertices), std::sqrt(1.0f + phi * phi));
    }();
    return mesh;
}
