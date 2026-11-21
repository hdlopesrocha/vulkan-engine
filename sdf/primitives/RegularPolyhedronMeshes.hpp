#pragma once
#include "ConvexPolyhedronDistanceFunction.hpp"

// Unit-circumradius meshes of the five Platonic solids, centred on the local
// origin. Every vertex lies on the unit sphere, so the conservative bounding
// spheres of the primitives derived from these meshes stay valid under any
// affine model transform (including non-uniform scale and rotation).
//
// The faces are extracted with ConvexPolyhedronDistanceFunction::hullFaces()
// instead of being hand-tabulated, which keeps the pentagonal dodecahedron and
// triangular icosahedron consistent with the golden-ratio vertex sets.
namespace RegularPolyhedronMeshes {
const ConvexPolyhedronMesh &tetrahedron();
const ConvexPolyhedronMesh &hexahedron();
const ConvexPolyhedronMesh &octahedron();
const ConvexPolyhedronMesh &dodecahedron();
const ConvexPolyhedronMesh &icosahedron();
} // namespace RegularPolyhedronMeshes
