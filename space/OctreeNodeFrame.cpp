#include "OctreeNodeFrame.hpp"
#include <cstring>
#include <cmath>


OctreeNodeFrame::OctreeNodeFrame()
    : node(NULL), iteratedNode(NULL), cube(), type(SpaceType::Empty), level(0), brushIndex(DISCARD_BRUSH_INDEX), chunkCube()
{
    for(int i=0;i<8;++i) sdf[i] = INFINITY;
}

OctreeNodeFrame::OctreeNodeFrame(const OctreeNodeFrame &t)
    : node(t.node), iteratedNode(t.iteratedNode), cube(t.cube), type(t.type), level(t.level), brushIndex(t.brushIndex), chunkCube(t.chunkCube)
{
    std::memcpy(this->sdf, t.sdf, sizeof(this->sdf));
}

OctreeNodeFrame::OctreeNodeFrame(OctreeNode* node, OctreeNode* iteratedNode, BoundingCube cube, SpaceType type, uint level, float * sdf, int brushIndex, BoundingCube chunkCube)
    : node(node), iteratedNode(iteratedNode), cube(cube), type(type), level(level), brushIndex(brushIndex), chunkCube(chunkCube)
{
    if (sdf) std::memcpy(this->sdf, sdf, sizeof(this->sdf)); else for(int i=0;i<8;++i) this->sdf[i]=INFINITY;
}
