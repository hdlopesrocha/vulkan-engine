#ifndef SDF_CAPSULE_DISTANCE_FUNCTION_HPP
#define SDF_CAPSULE_DISTANCE_FUNCTION_HPP

#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>

class CapsuleDistanceFunction : public SignedDistanceFunction {
public:
    glm::vec3 a;
    glm::vec3 b;
    float radius;
    CapsuleDistanceFunction(glm::vec3 a, glm::vec3 b, float r);
    virtual ~CapsuleDistanceFunction() = default;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override;
    glm::vec3 getCenter(const Transformation &model) const override;
};

#endif
