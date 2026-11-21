#pragma once
#include "../core/SignedDistanceFunction.hpp"
#include "ConvexPolyhedronDistanceFunction.hpp"
#include <array>
#include <vector>
#include <glm/glm.hpp>
#include "../../math/geometry/Transformation.hpp"
#include "../../math/geometry/BoundingSphere.hpp"

// SDF of a star (stellated) polyhedron: a convex base solid with one pyramid
// raised from every face. `mesh` must be centred on the local origin with all
// vertices on the unit sphere (RegularPolyhedronMeshes provides the five
// Platonic solids); `spikeHeight` is the pyramid height in those same local
// units. The finished star is rescaled so its furthest apex lies on the unit
// sphere, which keeps it inside the same object footprint as the regular
// polyhedron primitives under any affine model transform.
//
// The solid is the union of the base polyhedron and the face pyramids, so the
// field is evaluated piecewise over those convex components in world space.
// With bias = 0 the sharp star is exact outside the surface. A sampling bias
// (the octree leaf size) dilates every component by one cell, which rounds the
// spike tips and the convex edges enough for the trilinear lattice to
// reproduce them instead of faceting them.
class StarPolyhedronDistanceFunction : public SignedDistanceFunction {
public:
    StarPolyhedronDistanceFunction(const Transformation &model, float bias,
                                   const ConvexPolyhedronMesh &mesh,
                                   float spikeHeight = 0.8f);
    virtual ~StarPolyhedronDistanceFunction() = default;
    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;

private:
    using Triangle = std::array<glm::vec3, 3>;

    // One convex piece of the union (the base polyhedron or one face pyramid)
    // with the cached world-space triangles and the outward half-spaces used
    // by distance().
    struct ConvexPiece {
        std::vector<Triangle> triangles;
        std::vector<glm::vec3> planeNormals;
        std::vector<glm::vec3> planePoints;
        glm::vec3 center{0.0f};
        float radius = 0.0f;
    };

    std::vector<ConvexPiece> pieces_;
    float bevel_ = 0.0f;
};
