#ifndef SDF_OCTAHEDRON_DISTANCE_FUNCTION_HPP
#define SDF_OCTAHEDRON_DISTANCE_FUNCTION_HPP

#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/Transformation.hpp"
#include "SDF.hpp"

class OctahedronDistanceFunction : public SignedDistanceFunction {
public:
    OctahedronDistanceFunction();
    virtual ~OctahedronDistanceFunction() = default;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override;
    glm::vec3 getCenter(const Transformation &model) const override;
};

#endif