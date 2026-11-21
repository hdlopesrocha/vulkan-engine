#include "DodecahedronDistanceFunction.hpp"

#include <cmath>

namespace {
// Regular dodecahedron vertices in the golden-ratio coordinate set: the eight
// cube corners (±1,±1,±1) plus the twelve cyclic permutations of
// (0, ±1/φ, ±φ). Every vertex has radius sqrt(3), so dividing by that puts the
// circumradius at 1 (the model translate is the local origin).
const ConvexPolyhedronMesh &unitDodecahedron() {
    static const ConvexPolyhedronMesh mesh = [] {
        const float phi = (1.0f + std::sqrt(5.0f)) * 0.5f;
        const float invPhi = 1.0f / phi;
        ConvexPolyhedronMesh out;
        auto add = [&out](float x, float y, float z) {
            out.vertices.emplace_back(x, y, z);
        };
        for (int sx = -1; sx <= 1; sx += 2) {
            for (int sy = -1; sy <= 1; sy += 2) {
                for (int sz = -1; sz <= 1; sz += 2) {
                    add(static_cast<float>(sx), static_cast<float>(sy), static_cast<float>(sz));
                }
            }
        }
        for (int sy = -1; sy <= 1; sy += 2) {
            for (int sz = -1; sz <= 1; sz += 2) {
                add(0.0f, static_cast<float>(sy) * invPhi, static_cast<float>(sz) * phi);
            }
        }
        for (int sx = -1; sx <= 1; sx += 2) {
            for (int sy = -1; sy <= 1; sy += 2) {
                add(static_cast<float>(sx) * invPhi, static_cast<float>(sy) * phi, 0.0f);
            }
        }
        for (int sx = -1; sx <= 1; sx += 2) {
            for (int sz = -1; sz <= 1; sz += 2) {
                add(static_cast<float>(sx) * phi, 0.0f, static_cast<float>(sz) * invPhi);
            }
        }
        const float radius = std::sqrt(3.0f);
        for (glm::vec3 &v : out.vertices) v /= radius;
        out.faces = ConvexPolyhedronDistanceFunction::hullFaces(out.vertices);
        return out;
    }();
    return mesh;
}
} // namespace

DodecahedronDistanceFunction::DodecahedronDistanceFunction(const Transformation &model, float bias)
    : ConvexPolyhedronDistanceFunction(model, bias, unitDodecahedron()) {}
