#ifndef SPACE_TESSELATOR_HPP
#define SPACE_TESSELATOR_HPP


#include "OctreeNodeTriangleHandler.hpp"
#include "ThreadContext.hpp"
#include "../math/Geometry.hpp"
#include "../math/Vertex.hpp"

class Tesselator : public OctreeNodeTriangleHandler{
    ThreadContext * context;
public:
    Geometry * geometry;
    Tesselator(long * count, ThreadContext * context);
    void handle(Vertex &v0, Vertex &v1, Vertex &v2, bool sign) override;
};

#endif
