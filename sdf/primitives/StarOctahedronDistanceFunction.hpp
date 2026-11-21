#pragma once
#include "StarPolyhedronDistanceFunction.hpp"

// Star (stellated) regular octahedron: an octahedron with a triangular
// pyramid raised from each of its eight faces, normalized to unit
// circumradius. See StarPolyhedronDistanceFunction for the field description
// and the sampling-bias bevel.
class StarOctahedronDistanceFunction : public StarPolyhedronDistanceFunction {
public:
    StarOctahedronDistanceFunction(const Transformation &model, float bias,
                                   float spikeHeight = 0.8f);
    virtual ~StarOctahedronDistanceFunction() = default;
};
