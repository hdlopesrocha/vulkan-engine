#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/Transformation.hpp"
#include "SDF.hpp"

class TriangleStripDistanceFunction : public SignedDistanceFunction {
public:
    glm::vec3 v0, v1, v2, v3;
    float halfThick;
    TriangleStripDistanceFunction(const glm::vec3& v0, const glm::vec3& v1,
                                  const glm::vec3& v2, const glm::vec3& v3,
                                  float halfThick);
    virtual ~TriangleStripDistanceFunction() = default;
    float distance(const glm::vec3 &p, const Transformation &model) override;
};
