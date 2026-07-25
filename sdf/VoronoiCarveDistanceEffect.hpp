#pragma once
#include "SignedDistanceEffect.hpp"
#include <glm/glm.hpp>
#include "../math/Math.hpp"
#include "../math/BoundingCube.hpp"
#include "../math/BoundingSphere.hpp"
#include "../math/Transformation.hpp"
#include "SDF.hpp"

class VoronoiCarveDistanceEffect : public SignedDistanceEffect {
    public:
    float amplitude;
    float cellSize;
    glm::vec3 offset;
    float brightness;
    float contrast;
    VoronoiCarveDistanceEffect(SignedDistanceFunction &function_, float amplitude_, float cellSize_, glm::vec3 offset_, float brightness_, float contrast_, const Transformation &model, float bias);
    ~VoronoiCarveDistanceEffect();
    const char* getLabel() const override;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override { return SdfType::CARVE_VORONOI; }
};
