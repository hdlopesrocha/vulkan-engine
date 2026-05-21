#include "OctreeNodeData.hpp"
#include <cstring>
#include <cmath>

OctreeNodeData::OctreeNodeData(uint level, OctreeNode * node, BoundingCube cube, void * context)
    : level(level), node(node), cube(cube), context(context)
{
}

OctreeNodeData::OctreeNodeData(const OctreeNodeData &data)
    : level(data.level), node(data.node), cube(data.cube), context(data.context)
{
}

OctreeNodeData::OctreeNodeData()
    : level(0), node(NULL), cube(), context(NULL)
{
}






