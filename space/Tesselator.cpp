#include "Tesselator.hpp"
#include "Octree.hpp"
#include "IteratorHandler.hpp"
#include <cmath>


Tesselator::Tesselator(long * count_): OctreeNodeTriangleHandler(count_), geometry() {

}


int triplanarPlane(glm::vec3 normal) {
    glm::vec3 absNormal = glm::abs(normal);
    if (absNormal.x > absNormal.y && absNormal.x > absNormal.z) {
        return normal.x > 0 ? 0 : 1;
    } else if (absNormal.y > absNormal.x && absNormal.y > absNormal.z) {
        return normal.y > 0 ? 2 : 3;
    } else {
        return normal.z > 0 ? 4 : 5;
    }
}

glm::vec2 triplanarMapping(glm::vec3 position, int plane) {
    switch (plane) {
        case 0: return glm::vec2(-position.z, -position.y);
        case 1: return glm::vec2(position.z, -position.y);
        case 2: return glm::vec2(position.x, position.z);
        case 3: return glm::vec2(position.x, -position.z);
        case 4: return glm::vec2(position.x, -position.y);
        case 5: return glm::vec2(-position.x, -position.y);
        default: return glm::vec2(0.0,0.0);
    }
}

void Tesselator::handle(Vertex &v0, Vertex &v1, Vertex &v2) {
    if(v0.brushIndex>DISCARD_BRUSH_INDEX && 
        v1.brushIndex>DISCARD_BRUSH_INDEX && 
        v2.brushIndex>DISCARD_BRUSH_INDEX) {

        bool triplanar = true;
        float triplanarScale = 0.1f;
        // Copy the triangle before writing UVs: the incoming Vertex refs point
        // at OCTREE NODE vertices shared by adjacent cells, and other chunk
        // builds running on the worker pools may reference the same vertices.
        // Writing texCoord through the refs would mutate shared octree state
        // (a data race with nondeterministic UVs on shared boundary vertices).
        Vertex a = v0, b = v1, c = v2;
        if (triplanar) {
            // Use vertex normal for UV plane selection: it is the SDF gradient direction
            // (authoritative outward direction) and is more reliable than the geometric
            // cross product, especially for coarse LOD cells at curved surface boundaries.
            int plane = triplanarPlane(a.normal);
            a.texCoord = triplanarMapping(a.position, plane)*triplanarScale;
            b.texCoord = triplanarMapping(b.position, plane)*triplanarScale;
            c.texCoord = triplanarMapping(c.position, plane)*triplanarScale;
        }
        // Winding is pre-determined by emitSegment via the SDF sign change direction.
        geometry.addTriangle(a, b, c);
        ++(*count);
    }
}
