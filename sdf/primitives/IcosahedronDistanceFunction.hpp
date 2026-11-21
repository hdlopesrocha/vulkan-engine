#pragma once
#include "ConvexPolyhedronDistanceFunction.hpp"

// Exact Euclidean SDF of a regular icosahedron with unit circumradius, centred
// on the model origin. The twenty triangular faces are derived from the golden
// ratio vertex set at construction time; edges and vertices are beveled by the
// sampling bias exactly like TetrahedronDistanceFunction.
class IcosahedronDistanceFunction : public ConvexPolyhedronDistanceFunction {
public:
    IcosahedronDistanceFunction(const Transformation &model, float bias);
    virtual ~IcosahedronDistanceFunction() = default;
};
