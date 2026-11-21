#pragma once
#include "StarPolyhedronDistanceFunction.hpp"

// Star (stellated) regular tetrahedron: a tetrahedron with a triangular
// pyramid raised from each of its four faces, normalized to unit
// circumradius. See StarPolyhedronDistanceFunction for the field description
// and the sampling-bias bevel.
class StarTetrahedronDistanceFunction : public StarPolyhedronDistanceFunction {
public:
    StarTetrahedronDistanceFunction(const Transformation &model, float bias,
                                    float spikeHeight = 0.8f);
    virtual ~StarTetrahedronDistanceFunction() = default;
};
