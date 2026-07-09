#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/Transformation.hpp"
#include "SDF.hpp"

class PyramidDistanceFunction : public SignedDistanceFunction {
public:
    PyramidDistanceFunction();
    virtual ~PyramidDistanceFunction() = default;
    float distance(const glm::vec3 &p, const Transformation &model) override;
};

 
