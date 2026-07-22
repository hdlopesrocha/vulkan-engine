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

OctreeNodeFrame::OctreeNodeFrame(OctreeNode* node_, OctreeNode* iteratedNode_, const BoundingCube &cube_, SpaceType type_, uint level_, float * sdf_, int brushIndex_, const BoundingCube &chunkCube_)
    : node(node_), iteratedNode(iteratedNode_), cube(cube_), type(type_), level(level_), brushIndex(brushIndex_), chunkCube(chunkCube_)
{
    if (sdf_) std::memcpy(this->sdf, sdf_, sizeof(this->sdf)); else for(int i=0;i<8;++i) this->sdf[i]=INFINITY;
}
