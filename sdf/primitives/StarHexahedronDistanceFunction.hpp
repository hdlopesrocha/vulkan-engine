#pragma once
#include "StarPolyhedronDistanceFunction.hpp"

// Star (stellated) regular hexahedron (cube): a cube with a square pyramid
// raised from each of its six faces, normalized to unit circumradius. See
// StarPolyhedronDistanceFunction for the field description and the
// sampling-bias bevel.
class StarHexahedronDistanceFunction : public StarPolyhedronDistanceFunction {
public:
    StarHexahedronDistanceFunction(const Transformation &model, float bias,
                                   float spikeHeight = 0.8f);
    virtual ~StarHexahedronDistanceFunction() = default;
};
