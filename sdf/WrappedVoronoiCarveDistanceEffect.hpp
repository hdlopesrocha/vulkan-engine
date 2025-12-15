#pragma once
#include "WrappedSignedDistanceEffect.hpp"
#include <glm/glm.hpp>
#include "../math/Math.hpp"
#include "../math/BoundingCube.hpp"
#include "../math/BoundingSphere.hpp"
#include "../math/Transformation.hpp"
#include "SDF.hpp"

class WrappedVoronoiCarveDistanceEffect : public WrappedSignedDistanceEffect {
    public:
    float amplitude;
    float cellSize;
    glm::vec3 offset;
    float brightness;
    float contrast;
    WrappedVoronoiCarveDistanceEffect(WrappedSignedDistanceFunction * function, float amplitude, float cellSize, glm::vec3 offset, float brightness, float contrast);
    ~WrappedVoronoiCarveDistanceEffect();
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    const char* getLabel() const override;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    SdfType getType() const override { return SdfType::CARVE_VORONOI; }
};

 
