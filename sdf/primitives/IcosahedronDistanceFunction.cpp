#include "IcosahedronDistanceFunction.hpp"

#include <cmath>

namespace {
// Regular icosahedron vertices: the twelve cyclic permutations of
// (0, ±1, ±φ). Every vertex has radius sqrt(1 + φ²), so dividing by that puts
// the circumradius at 1 (the model translate is the local origin).
const ConvexPolyhedronMesh &unitIcosahedron() {
    static const ConvexPolyhedronMesh mesh = [] {
        const float phi = (1.0f + std::sqrt(5.0f)) * 0.5f;
        ConvexPolyhedronMesh out;
        auto add = [&out](float x, float y, float z) {
            out.vertices.emplace_back(x, y, z);
        };
        for (int sx = -1; sx <= 1; sx += 2) {
            for (int sy = -1; sy <= 1; sy += 2) {
                add(0.0f, static_cast<float>(sx), static_cast<float>(sy) * phi);
            }
        }
        for (int sx = -1; sx <= 1; sx += 2) {
            for (int sy = -1; sy <= 1; sy += 2) {
                add(static_cast<float>(sx), static_cast<float>(sy) * phi, 0.0f);
            }
        }
        for (int sx = -1; sx <= 1; sx += 2) {
            for (int sz = -1; sz <= 1; sz += 2) {
                add(static_cast<float>(sx) * phi, 0.0f, static_cast<float>(sz));
            }
        }
        const float radius = std::sqrt(1.0f + phi * phi);
        for (glm::vec3 &v : out.vertices) v /= radius;
        out.faces = ConvexPolyhedronDistanceFunction::hullFaces(out.vertices);
        return out;
    }();
    return mesh;
}
} // namespace

IcosahedronDistanceFunction::IcosahedronDistanceFunction(const Transformation &model, float bias)
    : ConvexPolyhedronDistanceFunction(model, bias, unitIcosahedron()) {}
