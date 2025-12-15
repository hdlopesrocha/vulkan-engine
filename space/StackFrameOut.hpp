#pragma once

#include "OctreeNodeData.hpp"

struct StackFrameOut : public OctreeNodeData {
    bool visited;
    StackFrameOut(const OctreeNodeData &data, bool visited) : OctreeNodeData(data) {
        this->visited = visited;
    }
};
