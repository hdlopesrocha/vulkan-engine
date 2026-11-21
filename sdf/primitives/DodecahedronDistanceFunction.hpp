#pragma once
#include "ConvexPolyhedronDistanceFunction.hpp"

// Exact Euclidean SDF of a regular dodecahedron with unit circumradius, centred
// on the model origin. The twelve pentagonal faces are derived from the golden
// ratio vertex set at construction time; edges and vertices are beveled by the
// sampling bias exactly like TetrahedronDistanceFunction.
class DodecahedronDistanceFunction : public ConvexPolyhedronDistanceFunction {
public:
    DodecahedronDistanceFunction(const Transformation &model, float bias);
    virtual ~DodecahedronDistanceFunction() = default;
};
