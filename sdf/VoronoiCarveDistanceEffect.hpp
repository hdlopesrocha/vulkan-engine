#pragma once
#include "SignedDistanceEffect.hpp"
#include <glm/glm.hpp>
#include "../math/Math.hpp"
#include "../math/geometry/BoundingCube.hpp"
#include "../math/geometry/BoundingSphere.hpp"
#include "../math/geometry/Transformation.hpp"
#include "SDF.hpp"

class VoronoiCarveDistanceEffect : public SignedDistanceEffect {
    public:
    float amplitude;
    float cellSize;
    glm::vec3 offset;
    float brightness;
    float contrast;
    VoronoiCarveDistanceEffect(SignedDistanceFunction &function_, float amplitude_, float cellSize_, glm::vec3 offset_, float brightness_, float contrast_, const Transformation &model, float bias);
    float distance(const glm::vec3 &p) const override;
};
