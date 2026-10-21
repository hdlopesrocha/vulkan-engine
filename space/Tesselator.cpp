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
        // UVs are painted by the TexturePainter (Vertex::texCoord / Vertex::uv on
        // the node). Triplanar derives its UVs on the GPU from the world
        // position, so a Triplanar vertex carries (0, 0) and is forced to zero
        // here regardless of what the painter returned. The ProjectionX/Y/Z and
        // Angular modes are likewise evaluated on the GPU; only Custom consumes
        // the painted UV, which is left untouched.
        const int triplanar = static_cast<int>(TextureType::Triplanar);
        if(a.textureType == triplanar) a.texCoord = glm::vec2(0.0f);
        if(b.textureType == triplanar) b.texCoord = glm::vec2(0.0f);
        if(c.textureType == triplanar) c.texCoord = glm::vec2(0.0f);
        // Winding is pre-determined by emitSegment via the SDF sign change direction.
        geometry.addTriangle(a, b, c);
        ++(*count);
    }
}
