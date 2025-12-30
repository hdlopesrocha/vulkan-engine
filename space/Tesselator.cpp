#include "Tesselator.hpp"
#include "Octree.hpp"
#include "IteratorHandler.hpp"
#include <cmath>


Tesselator::Tesselator(long * count, ThreadContext * context): OctreeNodeTriangleHandler(count), context(context), geometry() {

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


void Tesselator::handle(Vertex &v0, Vertex &v1, Vertex &v2, bool reverse) {
    if(v0.texIndex>DISCARD_BRUSH_INDEX && 
        v1.texIndex>DISCARD_BRUSH_INDEX && 
        v2.texIndex>DISCARD_BRUSH_INDEX) {

        bool triplanar = true;
        float triplanarScale = 0.1f;
        glm::vec3 d1 = v1.position - v0.position;
        glm::vec3 d2 = v2.position - v0.position;
        // Ensure normal follows the (v1 - v0) x (v2 - v0) convention to match Geometry::getNormal and the TES face normal
        glm::vec3 n = glm::cross(d1, d2);


        if (triplanar) {
            int plane = triplanarPlane(n);
            v0.texCoord = triplanarMapping(v0.position, plane)*triplanarScale;
            v1.texCoord = triplanarMapping(v1.position, plane)*triplanarScale;
            v2.texCoord = triplanarMapping(v2.position, plane)*triplanarScale;
        }
        // Tangent is computed in the tessellation evaluation shader; no per-vertex storage required here
        geometry.addTriangle(reverse ? v0 : v2, v1, reverse ? v2 : v0);
        ++(*count);
    }
}
