#include "OctreeNodeCubeSerialized.hpp"

OctreeNodeCubeSerialized::OctreeNodeCubeSerialized()
    : position(0.0f), normal(0.0f), texCoord(0.0f), brushIndex(DISCARD_BRUSH_INDEX), min(0.0f), bits(0), length(0.0f), level(0)
{
    for(int i=0;i<8;++i) children[i]=UINT_MAX;
}

OctreeNodeCubeSerialized::OctreeNodeCubeSerialized(float * sdf, BoundingCube cube, Vertex vertex, uint bits, uint level)
    : position(vertex.position), normal(vertex.normal), texCoord(vertex.texCoord), brushIndex(vertex.brushIndex), min(cube.getCenter()), bits(bits), length(cube.getLength()), level(level)
{
    for(int i=0;i<8;++i) children[i] = UINT_MAX;
}

