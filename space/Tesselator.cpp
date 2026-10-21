#include "Tesselator.hpp"
#include "Octree.hpp"
#include <cmath>


Tesselator::Tesselator(long * count_): OctreeNodeTriangleHandler(count_), geometry() {

}

void Tesselator::handle(Vertex &v0, Vertex &v1, Vertex &v2) {
    if(v0.brushIndex>DISCARD_BRUSH_INDEX &&
        v1.brushIndex>DISCARD_BRUSH_INDEX &&
        v2.brushIndex>DISCARD_BRUSH_INDEX) {

        // Copy the triangle before touching it: the incoming Vertex refs point
        // at OCTREE NODE vertices shared by adjacent cells, and other chunk
        // builds running on the worker pools may reference the same vertices.
        // Writing through the refs would mutate shared octree state.
        Vertex a = v0, b = v1, c = v2;
        // texCoord carries the UV scale the TextureCoordPainter painted (the
        // projection scale for Triplanar/ProjectionX/Y/Z, the authored UV for
        // Custom); the shader applies it. Nothing to force here.
        // Winding is pre-determined by emitSegment via the SDF sign change direction.
        geometry.addTriangle(a, b, c);
        ++(*count);
    }
}
