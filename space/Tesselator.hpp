#pragma once

#include "OctreeNodeTriangleHandler.hpp"
#include "ThreadContext.hpp"
#include "../math/Geometry.hpp"
#include "../math/Vertex.hpp"

class Tesselator : public OctreeNodeTriangleHandler{
public:
    Geometry geometry;
    Tesselator(long * count);
    void handle(Vertex &v0, Vertex &v1, Vertex &v2, bool sign) override;
};

 
