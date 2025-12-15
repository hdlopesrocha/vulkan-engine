#include "OctreeNodeData.hpp"
#include <cstring>
#include <cmath>

OctreeNodeData::OctreeNodeData(uint level, OctreeNode * node, BoundingCube cube, ContainmentType containmentType, void * context, float * sdf)
    : level(level), node(node), cube(cube), containmentType(containmentType), context(context)
{
    if (sdf) std::memcpy(this->sdf, sdf, sizeof(this->sdf)); else for(int i=0;i<8;++i) this->sdf[i]=INFINITY;
}

OctreeNodeData::OctreeNodeData(const OctreeNodeData &data)
    : level(data.level), node(data.node), cube(data.cube), containmentType(data.containmentType), context(data.context)
{
    std::memcpy(this->sdf, data.sdf, sizeof(this->sdf));
}

OctreeNodeData::OctreeNodeData()
    : level(0), node(NULL), cube(), containmentType(ContainmentType::Intersects), context(NULL)
{
    for(int i=0;i<8;++i) sdf[i] = INFINITY;
}






