#include "EdgeCell.hpp"
#include "OctreeNode.hpp"
#include <functional>

bool EdgeCell::isSurface(int targetLod) const {
    return node != NULL && node->getType() == SpaceType::Surface &&
        node->getLod() == targetLod;
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
