#include "OctreeNodeFrame.hpp"
#include <cstring>
#include <cmath>


OctreeNodeFrame::OctreeNodeFrame()
    : node(NULL), cube(), level(0), brushIndex(DISCARD_BRUSH_INDEX), interpolated(false), chunkCube()
{
    for(int i=0;i<8;++i) sdf[i] = INFINITY;
}

OctreeNodeFrame::OctreeNodeFrame(const OctreeNodeFrame &t)
    : node(t.node), cube(t.cube), level(t.level), brushIndex(t.brushIndex), interpolated(t.interpolated), chunkCube(t.chunkCube)
{
    std::memcpy(this->sdf, t.sdf, sizeof(this->sdf));
}

OctreeNodeFrame::OctreeNodeFrame(OctreeNode* node, BoundingCube cube, uint level, float * sdf, int brushIndex, bool interpolated, BoundingCube chunkCube)
    : node(node), cube(cube), level(level), brushIndex(brushIndex), interpolated(interpolated), chunkCube(chunkCube)
{
    if (sdf) std::memcpy(this->sdf, sdf, sizeof(this->sdf)); else for(int i=0;i<8;++i) this->sdf[i]=INFINITY;
}
