// Auto-generated wrapper header for OctreeNodeTriangleHandler
#pragma once
struct Vertex;

class OctreeNodeTriangleHandler {
public:
    long * count;
    OctreeNodeTriangleHandler(long * count);
    virtual void handle(Vertex &v0, Vertex &v1, Vertex &v2, bool sign) = 0;
};