#include "OctreeNodeData.hpp"

OctreeNodeData::OctreeNodeData(uint level_, OctreeNode * node_, const BoundingCube &cube_, void * context_)
    : level(level_), node(node_), cube(cube_), context(context_)
{
}

OctreeNodeData::OctreeNodeData()
    : level(0), node(NULL), cube(), context(NULL)
{
}






