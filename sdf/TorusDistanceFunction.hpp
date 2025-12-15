#ifndef SDF_TORUS_DISTANCE_FUNCTION_HPP
#define SDF_TORUS_DISTANCE_FUNCTION_HPP

#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>

class TorusDistanceFunction : public SignedDistanceFunction {
public:
    glm::vec2 radius;
    TorusDistanceFunction(glm::vec2 radius);
    virtual ~TorusDistanceFunction() = default;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override;
    glm::vec3 getCenter(const Transformation &model) const override;
};

#endif
