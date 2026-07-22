#pragma once

#include "OctreeNodeData.hpp"

struct StackFrameOut : public OctreeNodeData {
    bool visited;
    StackFrameOut(const OctreeNodeData &data, bool visited_) : OctreeNodeData(data) {
        this->visited = visited_;
    }
};
