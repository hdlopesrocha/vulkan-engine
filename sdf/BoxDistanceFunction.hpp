#ifndef SDF_BOX_DISTANCE_FUNCTION_HPP
#define SDF_BOX_DISTANCE_FUNCTION_HPP

#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>

class BoxDistanceFunction : public SignedDistanceFunction {
public:
    BoxDistanceFunction();
    virtual ~BoxDistanceFunction() = default;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override;
    glm::vec3 getCenter(const Transformation &model) const override;
};

#endif
