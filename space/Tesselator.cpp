#include "Tesselator.hpp"
#include "Octree.hpp"
#include "../utils/LandBrush.hpp"
#include <cmath>


Tesselator::Tesselator(long * count_): OctreeNodeTriangleHandler(count_), geometry() {

}

void Tesselator::handle(OctreeNode &o0, OctreeNode &o1, OctreeNode &o2,
                        const CellVertex &v0, const CellVertex &v1, const CellVertex &v2,
                        float eps) {
    if(o0.brushIndex>DISCARD_BRUSH_INDEX &&
        o1.brushIndex>DISCARD_BRUSH_INDEX &&
        o2.brushIndex>DISCARD_BRUSH_INDEX) {

        // The corner point/normal of each node was already resolved by the
        // walk (the Blender exporter's SDF solver) — nothing is solved here.
        // The material comes off the node.
        //
        // texCoord carries the UV scale the TextureCoordPainter painted (the
        // projection scale for Triplanar/ProjectionX/Y/Z, the authored UV for
        // Custom); the shader applies it.
        // Winding is pre-determined by emitSegment via the SDF sign change direction.
        (void)eps;

        // The nodes are shared by adjacent cells and by other chunk builds
        // running on the worker pools, so only copies are assembled below.
        // Land/Normal sentinels are classified from the solved point/normal,
        // exactly like OctreeTriangles.comp's nodeResolvedBrush(), so the
        // material space matches the GPU vertex payload.
        const int brush0 = landbrush::resolveVertexMaterial(o0.brushIndex, v0.position, v0.normal);
        const int brush1 = landbrush::resolveVertexMaterial(o1.brushIndex, v1.position, v1.normal);
        const int brush2 = landbrush::resolveVertexMaterial(o2.brushIndex, v2.position, v2.normal);

        Vertex a(v0.position, v0.normal, o0.data, brush0);
        a.textureType = o0.textureType;
        a.hsv = o0.hsv;

        Vertex b(v1.position, v1.normal, o1.data, brush1);
        b.textureType = o1.textureType;
        b.hsv = o1.hsv;

        Vertex c(v2.position, v2.normal, o2.data, brush2);
        c.textureType = o2.textureType;
        c.hsv = o2.hsv;

        geometry.addTriangle(a, b, c);
        ++(*count);
    }
}
