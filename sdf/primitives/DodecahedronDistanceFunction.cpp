#include "DodecahedronDistanceFunction.hpp"
#include "RegularPolyhedronMeshes.hpp"

DodecahedronDistanceFunction::DodecahedronDistanceFunction(const Transformation &model, float bias)
    : ConvexPolyhedronDistanceFunction(model, bias, RegularPolyhedronMeshes::dodecahedron()) {}
