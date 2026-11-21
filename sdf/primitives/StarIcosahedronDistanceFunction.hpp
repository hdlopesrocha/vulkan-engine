#pragma once
#include "StarPolyhedronDistanceFunction.hpp"

// Star (stellated) regular icosahedron: an icosahedron with a triangular
// pyramid raised from each of its twenty faces, normalized to unit
// circumradius. See StarPolyhedronDistanceFunction for the field description
// and the sampling-bias bevel.
class StarIcosahedronDistanceFunction : public StarPolyhedronDistanceFunction {
public:
    StarIcosahedronDistanceFunction(const Transformation &model, float bias,
                                    float spikeHeight = 0.8f);
    virtual ~StarIcosahedronDistanceFunction() = default;
};
