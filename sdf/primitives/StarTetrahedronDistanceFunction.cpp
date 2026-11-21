#include "StarTetrahedronDistanceFunction.hpp"
#include "RegularPolyhedronMeshes.hpp"

StarTetrahedronDistanceFunction::StarTetrahedronDistanceFunction(
    const Transformation &model, float bias, float spikeHeight)
    : StarPolyhedronDistanceFunction(model, bias,
                                     RegularPolyhedronMeshes::tetrahedron(),
                                     spikeHeight) {}
