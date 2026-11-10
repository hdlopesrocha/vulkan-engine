#pragma once
#include "SignedDistanceEffect.hpp"
#include <glm/glm.hpp>
#include "../../math/Math.hpp"
#include "../../math/geometry/BoundingCube.hpp"
#include "../../math/geometry/BoundingSphere.hpp"
#include "../../math/geometry/Transformation.hpp"

class VoronoiCarveDistanceEffect : public SignedDistanceEffect {
    public:
    // Noise-carved base SDF. The returned field is a conservative
    // (1-Lipschitz) world-space SDF: the Voronoi value, its 0.3 Perlin domain
    // warp and the painter's contrast all contribute to the Lipschitz scale.
    // A non-isometric carve has no closed-form exact Euclidean distance, but
    // this bound is safe for sphere tracing and octree classification and keeps
    // the gradient close to unit where the field is well resolved.
    float amplitude;
    float cellSize;
    glm::vec3 offset;
    float brightness;
    float contrast;
    VoronoiCarveDistanceEffect(SignedDistanceFunction &function_, float amplitude_, float cellSize_, glm::vec3 offset_, float brightness_, float contrast_, const Transformation &model, float bias);
    float distance(const glm::vec3 &p) const override;

    private:
    // Deterministic pseudo-random [0,1] from integer coords.
    static float randFromPerlin(const glm::ivec3 &cell, float seed);
    // Safe Voronoi 3D with adjustable cell size.
    static float voronoi3D(const glm::vec3 &p, float cellSize, float seed);
};
