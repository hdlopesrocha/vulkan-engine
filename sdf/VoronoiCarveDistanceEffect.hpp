#pragma once
#include "SignedDistanceEffect.hpp"
#include <glm/glm.hpp>
#include "../math/Math.hpp"
#include "../math/geometry/BoundingCube.hpp"
#include "../math/geometry/BoundingSphere.hpp"
#include "../math/geometry/Transformation.hpp"

class VoronoiCarveDistanceEffect : public SignedDistanceEffect {
    public:
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
