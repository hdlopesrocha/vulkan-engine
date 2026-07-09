#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/Transformation.hpp"
#include "SDF.hpp"

class BoxDistanceFunction : public SignedDistanceFunction {
public:
    BoxDistanceFunction();
    virtual ~BoxDistanceFunction() = default;
    float distance(const glm::vec3 &p, const Transformation &model) override;
};

 
