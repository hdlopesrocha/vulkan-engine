#ifndef SDF_CONE_DISTANCE_FUNCTION_HPP
#define SDF_CONE_DISTANCE_FUNCTION_HPP

#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>

class ConeDistanceFunction : public SignedDistanceFunction {
public:
    ConeDistanceFunction();
    virtual ~ConeDistanceFunction() = default;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override;
    glm::vec3 getCenter(const Transformation &model) const override;
};

#endif