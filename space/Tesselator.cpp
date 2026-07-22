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
        if (triplanar) {
            // Use vertex normal for UV plane selection: it is the SDF gradient direction
            // (authoritative outward direction) and is more reliable than the geometric
            // cross product, especially for coarse LOD cells at curved surface boundaries.
            int plane = triplanarPlane(v0.normal);
            v0.texCoord = triplanarMapping(v0.position, plane)*triplanarScale;
            v1.texCoord = triplanarMapping(v1.position, plane)*triplanarScale;
            v2.texCoord = triplanarMapping(v2.position, plane)*triplanarScale;
        }
        // Winding is pre-determined by emitSegment via the SDF sign change direction.
        geometry.addTriangle(v0, v1, v2);
        ++(*count);
    }
}
