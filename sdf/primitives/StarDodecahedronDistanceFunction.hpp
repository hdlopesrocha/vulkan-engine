#pragma once
#include "StarPolyhedronDistanceFunction.hpp"

// Star (stellated) regular dodecahedron: a dodecahedron with a pentagonal
// pyramid raised from each of its twelve faces, normalized to unit
// circumradius. See StarPolyhedronDistanceFunction for the field description
// and the sampling-bias bevel.
class StarDodecahedronDistanceFunction : public StarPolyhedronDistanceFunction {
public:
    StarDodecahedronDistanceFunction(const Transformation &model, float bias,
                                     float spikeHeight = 0.8f);
    virtual ~StarDodecahedronDistanceFunction() = default;
};
