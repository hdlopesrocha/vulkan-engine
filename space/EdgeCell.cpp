#include "EdgeCell.hpp"
#include "OctreeNode.hpp"
#include "../sdf/SDF.hpp"
#include <functional>

bool EdgeCell::isSurface() const {
    // A node typed Surface only because its children contain a surface is NOT
    // a usable ring vertex: its own eight corner samples have no zero crossing,
    // so SDF::getPosition() falls back to the cube centre and the emitted
    // triangle bridges from that centre to the real surface (a spike). Ring
    // polygons must use cells that actually straddle the surface; a
    // non-crossing neighbour is treated exactly like Empty/Solid space and
    // closed by the remaining quadrants.
    return node != NULL
        && node->getType() == SpaceType::Surface
        && SDF::eval(node->sdf) == SpaceType::Surface;
}

bool EdgeCell::ownerLess(const EdgeCell &a, const EdgeCell &b) {
    float al = a.cube.getLengthX();
    float bl = b.cube.getLengthX();
    if(al != bl) return al < bl;

    glm::vec3 amin = a.cube.getMin();
    glm::vec3 bmin = b.cube.getMin();
    if(amin.x != bmin.x) return amin.x < bmin.x;
    if(amin.y != bmin.y) return amin.y < bmin.y;
    if(amin.z != bmin.z) return amin.z < bmin.z;
    return std::less<OctreeNode*>()(a.node, b.node);
}
