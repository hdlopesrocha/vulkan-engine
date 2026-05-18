#include "ForwardingHandler.hpp"

ForwardingHandler::ForwardingHandler(long *count, std::vector<OctreeNodeTriangleHandler*> *h)
    : OctreeNodeTriangleHandler(count), handlersVec(h) {
}

void ForwardingHandler::handle(Vertex &v0, Vertex &v1, Vertex &v2) {
    for (auto h : *handlersVec) {
        if (h) h->handle(v0, v1, v2);
    }
}
