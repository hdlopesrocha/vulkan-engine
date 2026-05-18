#pragma once
#include <vector>
#include "OctreeNodeTriangleHandler.hpp"

class ForwardingHandler : public OctreeNodeTriangleHandler {
public:
    std::vector<OctreeNodeTriangleHandler*> * handlersVec;
    ForwardingHandler(long *count, std::vector<OctreeNodeTriangleHandler*> *h);
    void handle(Vertex &v0, Vertex &v1, Vertex &v2) override;
};
