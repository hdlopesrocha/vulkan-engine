#include "OctreeNodeData.hpp"

OctreeNodeData::OctreeNodeData(uint level_, OctreeNode * node_, const BoundingCube &cube_)
    : level(level_), node(node_), cube(cube_)
{
}

OctreeNodeData::OctreeNodeData()
    : level(0), node(NULL), cube()
{
}






