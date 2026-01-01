#include "OctreeNodeData.hpp"
#include <cstring>
#include <cmath>

OctreeNodeData::OctreeNodeData(uint level, OctreeNode * node, BoundingCube cube, ContainmentType containmentType, void * context)
    : level(level), node(node), cube(cube), containmentType(containmentType), context(context)
{
}

OctreeNodeData::OctreeNodeData(const OctreeNodeData &data)
    : level(data.level), node(data.node), cube(data.cube), containmentType(data.containmentType), context(data.context)
{
}

OctreeNodeData::OctreeNodeData()
    : level(0), node(NULL), cube(), containmentType(ContainmentType::Intersects), context(NULL)
{
}






