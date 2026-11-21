#include "StarOctahedronDistanceFunction.hpp"
#include "RegularPolyhedronMeshes.hpp"

StarOctahedronDistanceFunction::StarOctahedronDistanceFunction(
    const Transformation &model, float bias, float spikeHeight)
    : StarPolyhedronDistanceFunction(model, bias,
                                     RegularPolyhedronMeshes::octahedron(),
                                     spikeHeight) {}
