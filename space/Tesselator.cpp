#include "Tesselator.hpp"
#include "Octree.hpp"
#include <cmath>


Tesselator::Tesselator(long * count_): OctreeNodeTriangleHandler(count_), geometry() {

}

void Tesselator::handle(OctreeNode &o0, OctreeNode &o1, OctreeNode &o2,
                        const CellVertex &v0, const CellVertex &v1, const CellVertex &v2,
                        float eps) {
    if(o0.brushIndex>DISCARD_BRUSH_INDEX &&
        o1.brushIndex>DISCARD_BRUSH_INDEX &&
        o2.brushIndex>DISCARD_BRUSH_INDEX) {

        // The corner point/normal of each node was already resolved by the walk
        // — the GPU-computed vertex in the app, the cell's SDF only on GPU-less
        // paths — so nothing is solved here. The material comes off the node.
        //
        // texCoord carries the UV scale the TextureCoordPainter painted (the
        // projection scale for Triplanar/ProjectionX/Y/Z, the authored UV for
        // Custom); the shader applies it.
        // Winding is pre-determined by emitSegment via the SDF sign change direction.
        (void)eps;

        // The nodes are shared by adjacent cells and by other chunk builds
        // running on the worker pools, so only copies are assembled below.
        Vertex a(v0.position, v0.normal, o0.data, o0.brushIndex);
        a.textureType = o0.textureType;
        a.hsv = o0.hsv;

        Vertex b(v1.position, v1.normal, o1.data, o1.brushIndex);
        b.textureType = o1.textureType;
        b.hsv = o1.hsv;

        Vertex c(v2.position, v2.normal, o2.data, o2.brushIndex);
        c.textureType = o2.textureType;
        c.hsv = o2.hsv;

        geometry.addTriangle(a, b, c);
        ++(*count);
    }
}
